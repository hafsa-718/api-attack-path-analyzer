"""Pydantic request / response models for the FastAPI layer."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyzeRequest(BaseModel):
    """Configuration submitted alongside the spec file upload."""

    model: str = Field(default="claude-sonnet-4-6", description="Claude model ID to use for reasoning")
    max_candidates: int = Field(default=50, ge=1, le=200)
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    wipe_before_build: bool = Field(default=False, description="Wipe previous Neo4j data for this spec")


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: dict[str, Any] | None = Field(
        default=None,
        description="Serialised AnalysisResult, present only when status == completed",
    )


class HealthResponse(BaseModel):
    status: str
    version: str
    neo4j_connected: bool
