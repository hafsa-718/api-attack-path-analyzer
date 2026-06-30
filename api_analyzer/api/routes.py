"""FastAPI routes for the API Attack Path Analyzer.

Endpoints
---------
  GET  /health                 — liveness / readiness probe
  POST /jobs                   — submit a spec file for analysis (multipart)
  GET  /jobs/{job_id}          — poll job status and retrieve result
  GET  /jobs/{job_id}/report   — fetch the rendered HTML report

Job lifecycle
-------------
  PENDING → RUNNING → COMPLETED
                    ↘ FAILED

Jobs are stored in memory (``_jobs`` dict).  Suitable for single-process
deployments and demos; a production deployment would use Redis or a DB.

Analysis runs in a thread-pool executor so the async event loop is not blocked
by the synchronous Neo4j driver and Anthropic SDK calls.
"""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from api_analyzer import __version__
from api_analyzer._pipeline import PipelineConfig, run_pipeline
from api_analyzer.api.models import (
    AnalyzeRequest,
    HealthResponse,
    JobCreatedResponse,
    JobStatus,
    JobStatusResponse,
)
from api_analyzer.report.generator import generate_report
from api_analyzer.report.sarif import generate_sarif

router = APIRouter()

# ── In-memory job store ────────────────────────────────────────────────────────

type _Job = dict[str, Any]
_jobs: dict[str, _Job] = {}
_jobs_lock = asyncio.Lock()


def _new_job(job_id: str) -> _Job:
    return {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "submitted_at": datetime.now(tz=timezone.utc),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None,
        "html_report": None,
        "sarif": None,
        "spec_filename": "openapi.yaml",
    }


# ── Driver dependency ──────────────────────────────────────────────────────────


def _get_driver(request: Request):
    """Retrieve the Neo4j driver stored in app.state by the lifespan handler."""
    return request.app.state.driver


# ── Background worker ──────────────────────────────────────────────────────────


async def _run_analysis(job_id: str, spec_bytes: bytes, config: PipelineConfig, driver) -> None:
    """Run the pipeline in a thread pool and update the job record."""
    async with _jobs_lock:
        _jobs[job_id]["status"] = JobStatus.RUNNING
        _jobs[job_id]["started_at"] = datetime.now(tz=timezone.utc)

    try:
        # Write spec to a temp file so run_pipeline can ingest it
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp.write(spec_bytes)
            tmp_path = Path(tmp.name)

        # run_pipeline is synchronous — execute in thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline(tmp_path, driver, config=config),
        )
        tmp_path.unlink(missing_ok=True)

        # Generate HTML report to a temp file
        report_tmp = Path(tempfile.mktemp(suffix=".html"))
        await loop.run_in_executor(
            None,
            lambda: generate_report(result, report_tmp, tool_version=__version__),
        )
        html_content = report_tmp.read_text(encoding="utf-8")
        report_tmp.unlink(missing_ok=True)

        # Generate SARIF
        async with _jobs_lock:
            spec_filename = _jobs[job_id].get("spec_filename", "openapi.yaml")
        sarif_dict = await loop.run_in_executor(
            None,
            lambda: generate_sarif(result, spec_filename=spec_filename, tool_version=__version__),
        )

        async with _jobs_lock:
            _jobs[job_id].update(
                status=JobStatus.COMPLETED,
                completed_at=datetime.now(tz=timezone.utc),
                result=result.model_dump(mode="json"),
                html_report=html_content,
                sarif=sarif_dict,
            )

    except Exception as exc:  # noqa: BLE001
        async with _jobs_lock:
            _jobs[job_id].update(
                status=JobStatus.FAILED,
                completed_at=datetime.now(tz=timezone.utc),
                error=str(exc),
            )


# ── Routes ─────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health(driver=Depends(_get_driver)) -> HealthResponse:
    """Return API liveness and Neo4j connectivity status."""
    try:
        driver.verify_connectivity()
        neo4j_ok = True
    except Exception:  # noqa: BLE001
        neo4j_ok = False

    return HealthResponse(
        status="ok",
        version=__version__,
        neo4j_connected=neo4j_ok,
    )


@router.post("/jobs", response_model=JobCreatedResponse, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    spec_file: UploadFile,
    model: str = "claude-sonnet-4-6",
    max_candidates: int = 50,
    confidence_threshold: float = 0.4,
    wipe_before_build: bool = False,
    max_workers: int = 5,
    target_url: str | None = None,
    driver=Depends(_get_driver),
) -> JobCreatedResponse:
    """Submit a spec file for analysis.

    Returns immediately with a ``job_id``.  Poll ``GET /jobs/{job_id}`` for
    status.  The spec must be a valid OpenAPI 3.x or Swagger 2.0 file (YAML or
    JSON).
    """
    spec_bytes = await spec_file.read()
    if not spec_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    spec_filename = spec_file.filename or "openapi.yaml"

    config = PipelineConfig(
        model=model,
        max_candidates=max_candidates,
        confidence_threshold=confidence_threshold,
        wipe_before_build=wipe_before_build,
        max_workers=max_workers,
        target_url=target_url or None,
    )

    job_id = str(uuid.uuid4())
    async with _jobs_lock:
        _jobs[job_id] = _new_job(job_id)
        _jobs[job_id]["spec_filename"] = spec_filename

    background_tasks.add_task(_run_analysis, job_id, spec_bytes, config, driver)

    return JobCreatedResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        submitted_at=_jobs[job_id]["submitted_at"],
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str) -> JobStatusResponse:
    """Poll job status.  Returns the full ``AnalysisResult`` when completed."""
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        submitted_at=job["submitted_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error=job["error"],
        result=job["result"],
    )


@router.get("/jobs/{job_id}/sarif")
async def get_job_sarif(job_id: str) -> JSONResponse:
    """Return the SARIF 2.1.0 findings for a completed job.

    Upload the response to GitHub Advanced Security via
    ``github/codeql-action/upload-sarif`` to get inline PR annotations.
    """
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is {job['status'].value}, not completed.")
    if not job["sarif"]:
        raise HTTPException(status_code=500, detail="SARIF generation failed.")

    return JSONResponse(
        content=job["sarif"],
        headers={"Content-Disposition": f'attachment; filename="results.sarif"'},
    )


@router.get("/jobs/{job_id}/report", response_class=HTMLResponse)
async def get_job_report(job_id: str) -> HTMLResponse:
    """Return the rendered HTML report for a completed job."""
    async with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is {job['status'].value}, not completed.",
        )
    if not job["html_report"]:
        raise HTTPException(status_code=500, detail="Report generation failed.")

    return HTMLResponse(content=job["html_report"])
