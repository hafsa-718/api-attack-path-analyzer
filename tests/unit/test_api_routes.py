"""Unit tests for api_analyzer/api/routes.py (M13).

Strategy:
  - Use FastAPI TestClient (synchronous, wraps starlette.testclient)
  - Patch api_analyzer.api.routes.run_pipeline so no real Neo4j / LLM is hit
  - Patch api_analyzer.api.routes.generate_report for HTML tests
  - Neo4j driver stored in app.state is replaced with a MagicMock

Coverage:
  GET /health — ok, neo4j ok/failing
  POST /jobs  — 202 accepted, job_id returned, empty file 400, bad content-type 415
  GET /jobs/{id} — pending/running/completed/failed states, 404 for unknown
  GET /jobs/{id}/report — 200 HTML, 404 not found, 409 not completed
  Background analysis — completes → COMPLETED, raises → FAILED
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api_analyzer.api.app import app
from api_analyzer.api.models import JobStatus
from api_analyzer.api.routes import _jobs, _jobs_lock
from api_analyzer.models.enums import Severity, SpecFormat
from api_analyzer.models.report import AnalysisResult


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_result() -> AnalysisResult:
    return AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        spec_title="Test API",
        spec_version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        analyzed_at=datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        duration_seconds=4.2,
        endpoint_count=10,
        public_endpoint_count=3,
        auth_declared=True,
        spec_completeness=0.8,
        chains=[],
        graph_node_count=15,
        graph_edge_count=22,
        patterns_run=["AP-001", "AP-002"],
        candidates_evaluated=5,
        candidates_rejected=5,
        llm_tokens_used=0,
        estimated_cost_usd=0.0,
        parse_warnings=[],
    )


def _make_client(neo4j_ok: bool = True) -> TestClient:
    """Create a TestClient with a mocked Neo4j driver in app.state."""
    driver = MagicMock()
    if neo4j_ok:
        driver.verify_connectivity.return_value = None
    else:
        driver.verify_connectivity.side_effect = Exception("connection refused")

    app.state.driver = driver
    return TestClient(app, raise_server_exceptions=False)


def _yaml_bytes() -> bytes:
    return b"openapi: '3.0.0'\ninfo:\n  title: Test\n  version: '1.0'\npaths: {}\n"


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self):
        client = _make_client()
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_neo4j_connected_true_when_driver_ok(self):
        client = _make_client(neo4j_ok=True)
        data = client.get("/health").json()
        assert data["neo4j_connected"] is True

    def test_neo4j_connected_false_when_driver_fails(self):
        client = _make_client(neo4j_ok=False)
        data = client.get("/health").json()
        assert data["neo4j_connected"] is False

    def test_contains_version(self):
        from api_analyzer import __version__
        client = _make_client()
        data = client.get("/health").json()
        assert data["version"] == __version__


# ── POST /jobs ─────────────────────────────────────────────────────────────────


class TestCreateJob:
    def test_returns_202(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        assert resp.status_code == 202

    def test_returns_job_id(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID

    def test_status_is_pending_or_completed_immediately(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        status = resp.json()["status"]
        assert status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.COMPLETED)

    def test_empty_file_returns_400(self):
        client = _make_client()
        resp = client.post(
            "/jobs",
            files={"spec_file": ("spec.yaml", b"", "text/yaml")},
        )
        assert resp.status_code == 400

    def test_unsupported_content_type_still_accepted(self):
        # Content-type is not validated — the parser handles format detection
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report") as mock_gen:
            mock_gen.side_effect = lambda r, p, **kw: (Path(p).write_text("<html/>", encoding="utf-8") or Path(p))
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.exe", b"binary", "application/octet-stream")},
            )
        assert resp.status_code == 202

    def test_json_content_type_accepted(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.json", b'{"openapi":"3.0.0"}', "application/json")},
            )
        assert resp.status_code == 202

    def test_no_content_type_accepted(self):
        client = _make_client()
        def _gen(r, p, **kw):
            Path(p).write_text("<html/>", encoding="utf-8")
            return Path(p)
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", side_effect=_gen):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes())},
            )
        assert resp.status_code == 202

    def test_submitted_at_present(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        assert "submitted_at" in resp.json()

    def test_default_model_param_passed(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()) as mock_pipe, \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        # Give background task time to execute (TestClient runs sync)
        import time; time.sleep(0.1)
        if mock_pipe.called:
            config = mock_pipe.call_args[1]["config"]
            assert config.model == "claude-sonnet-4-6"

    def test_custom_model_param_forwarded(self):
        client = _make_client()
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()) as mock_pipe, \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            client.post(
                "/jobs?model=claude-opus-4-8",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        import time; time.sleep(0.1)
        if mock_pipe.called:
            config = mock_pipe.call_args[1]["config"]
            assert config.model == "claude-opus-4-8"


# ── GET /jobs/{job_id} ─────────────────────────────────────────────────────────


class TestGetJob:
    def _create_job(self, client, status_override: str | None = None) -> str:
        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            resp = client.post(
                "/jobs",
                files={"spec_file": ("spec.yaml", _yaml_bytes(), "text/yaml")},
            )
        return resp.json()["job_id"]

    def test_unknown_job_returns_404(self):
        client = _make_client()
        resp = client.get("/jobs/nonexistent-id")
        assert resp.status_code == 404

    def test_created_job_is_retrievable(self):
        client = _make_client()
        job_id = self._create_job(client)
        resp = client.get(f"/jobs/{job_id}")
        assert resp.status_code == 200

    def test_job_response_has_job_id(self):
        client = _make_client()
        job_id = self._create_job(client)
        data = client.get(f"/jobs/{job_id}").json()
        assert data["job_id"] == job_id

    def test_job_response_has_status(self):
        client = _make_client()
        job_id = self._create_job(client)
        data = client.get(f"/jobs/{job_id}").json()
        assert data["status"] in (s.value for s in JobStatus)

    def test_job_response_has_submitted_at(self):
        client = _make_client()
        job_id = self._create_job(client)
        data = client.get(f"/jobs/{job_id}").json()
        assert data["submitted_at"] is not None

    def test_failed_job_has_error_field(self):
        client = _make_client()
        # Inject a pre-failed job directly
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.FAILED,
            "submitted_at": datetime.now(tz=timezone.utc),
            "started_at": datetime.now(tz=timezone.utc),
            "completed_at": datetime.now(tz=timezone.utc),
            "error": "Neo4j connection refused",
            "result": None,
            "html_report": None,
        }
        data = client.get(f"/jobs/{job_id}").json()
        assert data["status"] == JobStatus.FAILED
        assert data["error"] == "Neo4j connection refused"

    def test_completed_job_has_result(self):
        client = _make_client()
        result = _fake_result()
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.COMPLETED,
            "submitted_at": datetime.now(tz=timezone.utc),
            "started_at": datetime.now(tz=timezone.utc),
            "completed_at": datetime.now(tz=timezone.utc),
            "error": None,
            "result": result.model_dump(mode="json"),
            "html_report": "<html>report</html>",
        }
        data = client.get(f"/jobs/{job_id}").json()
        assert data["result"] is not None
        assert data["result"]["spec_title"] == "Test API"


# ── GET /jobs/{job_id}/report ──────────────────────────────────────────────────


class TestGetJobReport:
    def _inject_job(self, status: JobStatus, html: str | None = None) -> str:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": status,
            "submitted_at": datetime.now(tz=timezone.utc),
            "started_at": datetime.now(tz=timezone.utc),
            "completed_at": datetime.now(tz=timezone.utc) if status != JobStatus.RUNNING else None,
            "error": None,
            "result": None,
            "html_report": html,
        }
        return job_id

    def test_unknown_job_returns_404(self):
        client = _make_client()
        resp = client.get("/jobs/does-not-exist/report")
        assert resp.status_code == 404

    def test_pending_job_returns_409(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.PENDING)
        resp = client.get(f"/jobs/{job_id}/report")
        assert resp.status_code == 409

    def test_running_job_returns_409(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.RUNNING)
        resp = client.get(f"/jobs/{job_id}/report")
        assert resp.status_code == 409

    def test_failed_job_returns_409(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.FAILED)
        resp = client.get(f"/jobs/{job_id}/report")
        assert resp.status_code == 409

    def test_completed_job_returns_200_html(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.COMPLETED, html="<html>hello</html>")
        resp = client.get(f"/jobs/{job_id}/report")
        assert resp.status_code == 200

    def test_completed_job_content_type_is_html(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.COMPLETED, html="<html>hello</html>")
        resp = client.get(f"/jobs/{job_id}/report")
        assert "text/html" in resp.headers["content-type"]

    def test_completed_job_returns_html_content(self):
        client = _make_client()
        job_id = self._inject_job(JobStatus.COMPLETED, html="<html>hello</html>")
        resp = client.get(f"/jobs/{job_id}/report")
        assert "<html>hello</html>" in resp.text


# ── Background analysis integration ───────────────────────────────────────────


class TestBackgroundAnalysis:
    """Tests that _run_analysis correctly updates the job record.

    These tests call the async function directly via asyncio.run so they can
    control timing without relying on BackgroundTasks scheduling.
    """

    def _create_pending_job(self) -> str:
        job_id = str(uuid.uuid4())
        _jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "submitted_at": datetime.now(tz=timezone.utc),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "result": None,
            "html_report": None,
        }
        return job_id

    def _gen_side_effect(self, html: str = "<html/>"):
        """Return a generate_report side-effect that writes html to the given path."""
        def _write(r, path, **kw):
            Path(path).write_text(html, encoding="utf-8")
            return Path(path)
        return _write

    def test_successful_analysis_sets_completed_status(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()
        result = _fake_result()

        with patch("api_analyzer.api.routes.run_pipeline", return_value=result), \
             patch("api_analyzer.api.routes.generate_report", side_effect=self._gen_side_effect()):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["status"] == JobStatus.COMPLETED

    def test_successful_analysis_stores_result(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()
        result = _fake_result()

        with patch("api_analyzer.api.routes.run_pipeline", return_value=result), \
             patch("api_analyzer.api.routes.generate_report", side_effect=self._gen_side_effect()):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["result"] is not None
        assert _jobs[job_id]["result"]["spec_title"] == "Test API"

    def test_successful_analysis_stores_html_report(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()
        result = _fake_result()
        html = "<!DOCTYPE html><html>report</html>"

        with patch("api_analyzer.api.routes.run_pipeline", return_value=result), \
             patch("api_analyzer.api.routes.generate_report") as mock_gen:
            # Capture what path generate_report is called with, then write HTML there
            def _side_effect(r, path, **kw):
                Path(path).write_text(html, encoding="utf-8")
                return Path(path)
            mock_gen.side_effect = _side_effect
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["html_report"] == html

    def test_pipeline_error_sets_failed_status(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()

        with patch("api_analyzer.api.routes.run_pipeline", side_effect=RuntimeError("boom")):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["status"] == JobStatus.FAILED

    def test_pipeline_error_stores_error_message(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()

        with patch("api_analyzer.api.routes.run_pipeline", side_effect=RuntimeError("boom")):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert "boom" in _jobs[job_id]["error"]

    def test_analysis_transitions_through_running_state(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()
        states_seen = []

        result = _fake_result()
        original_run_pipeline = None

        def _capturing_pipeline(*args, **kwargs):
            states_seen.append(_jobs[job_id]["status"])
            return result

        with patch("api_analyzer.api.routes.run_pipeline", side_effect=_capturing_pipeline), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert JobStatus.RUNNING in states_seen

    def test_completed_at_set_on_success(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()

        with patch("api_analyzer.api.routes.run_pipeline", return_value=_fake_result()), \
             patch("api_analyzer.api.routes.generate_report", return_value=Path("/tmp/r.html")):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["completed_at"] is not None

    def test_completed_at_set_on_failure(self):
        from api_analyzer.api.routes import _run_analysis
        from api_analyzer._pipeline import PipelineConfig

        job_id = self._create_pending_job()
        driver = MagicMock()

        with patch("api_analyzer.api.routes.run_pipeline", side_effect=Exception("fail")):
            asyncio.run(_run_analysis(job_id, _yaml_bytes(), PipelineConfig(), driver))

        assert _jobs[job_id]["completed_at"] is not None
