"""Pydantic models for runtime probe results."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class ProbeOutcome(str, Enum):
    CONFIRMED     = "CONFIRMED"      # HTTP 2xx — endpoint accessible without auth
    AUTH_ENFORCED = "AUTH_ENFORCED"  # HTTP 401/403 on GET — server enforces auth
    INCONCLUSIVE  = "INCONCLUSIVE"   # POST/PUT/PATCH with no body — 4xx ambiguous
    NOT_FOUND     = "NOT_FOUND"      # HTTP 404 — path doesn't exist
    RATE_LIMITED  = "RATE_LIMITED"   # HTTP 429 — rate limited on first probe
    SERVER_ERROR  = "SERVER_ERROR"   # HTTP 5xx — inconclusive
    UNREACHABLE   = "UNREACHABLE"    # connection/timeout error


class ProbeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    outcome: ProbeOutcome
    probed_url: str
    status_code: int | None = Field(default=None)
    latency_ms: int | None = Field(default=None)
    error: str | None = Field(default=None)
    note: str | None = Field(default=None)  # extra context, e.g. "ID harvested from list endpoint"
