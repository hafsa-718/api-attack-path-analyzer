"""Runtime probe module — fire real HTTP requests to verify validated chains.

Default behaviour (all patterns except AP-001):
  Probe the entry endpoint (step 0) without credentials.
  A 2xx confirms the chain is cold-start reachable.

BOLA two-step probe (AP-001):
  Step 1 — GET the list/collection endpoint (step 0, no path params).
            If 2xx, parse the JSON to harvest a real resource ID.
  Step 2 — GET the detail endpoint (step 1) with the harvested ID
            substituted into path params.
            If 2xx, BOLA is confirmed: an unauthenticated caller accessed
            another user's resource using a real ID from the public list.

Probe outcomes
--------------
CONFIRMED      2xx   — endpoint accessible without auth.
AUTH_ENFORCED  401/403 — server enforces auth.
NOT_FOUND      404   — path doesn't exist on this server (wrong URL / version prefix).
RATE_LIMITED   429   — rate limiting triggered.
SERVER_ERROR   5xx   — server error; inconclusive.
UNREACHABLE    connection/timeout — target not responding.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from api_analyzer.models.chain import ValidatedChain

from api_analyzer.models.probe import ProbeOutcome, ProbeResult

_BOLA_PATTERN = "AP-001"
_PATH_PARAM_RE = re.compile(r"\{[^}]+\}")
_DEFAULT_TIMEOUT = 6.0

# Common ID field names in API JSON responses, in priority order
_ID_FIELDS = [
    "id", "_id", "uuid", "ID",
    "userId", "user_id", "accountId", "account_id",
    "employeeId", "employee_id", "customerId", "customer_id",
    "resourceId", "resource_id", "objectId", "object_id",
]


def _classify(status: int) -> ProbeOutcome:
    if 200 <= status < 300:
        return ProbeOutcome.CONFIRMED
    if status in (401, 403):
        return ProbeOutcome.AUTH_ENFORCED
    if status == 404:
        return ProbeOutcome.NOT_FOUND
    if status == 429:
        return ProbeOutcome.RATE_LIMITED
    if status >= 500:
        return ProbeOutcome.SERVER_ERROR
    return ProbeOutcome.NOT_FOUND


def _fill_params(path: str, value: str) -> str:
    """Replace all {param} placeholders with the given value."""
    return _PATH_PARAM_RE.sub(value, path)


def _extract_first_id(body: str) -> str | None:
    """Pull the first ID-like value from a JSON response body.

    Handles common response shapes:
      - Array at root:             [{"id": "abc"}, ...]
      - Wrapped in data/items/...: {"data": [{"id": "abc"}]}
      - Single object:             {"id": "abc", ...}
    """
    try:
        data: Any = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None

    # Unwrap common envelope keys
    if isinstance(data, dict):
        for key in ("data", "items", "results", "records", "list", "content"):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    # Take first item if we have a list
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]

    if not isinstance(data, dict):
        return None

    # Try known ID field names first
    for field in _ID_FIELDS:
        val = data.get(field)
        if val is not None:
            return str(val)

    # Fall back to any key that ends with 'id' (case-insensitive)
    for key, val in data.items():
        if key.lower().endswith("id") and val is not None:
            return str(val)

    return None


def _get(client: Any, url: str, timeout: float) -> tuple[int, str, int]:
    """Fire a GET request; return (status_code, body, latency_ms)."""
    t0 = time.perf_counter()
    resp = client.get(url, timeout=timeout, follow_redirects=True,
                      headers={"User-Agent": "API-Analyzer-Probe/1.0"})
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return resp.status_code, resp.text, latency_ms


def _probe_bola(
    client: Any,
    base_url: str,
    chain: ValidatedChain,
    timeout: float,
) -> ProbeResult:
    """Two-step BOLA probe: harvest real ID from list, confirm detail access."""
    if len(chain.steps) < 2:
        return _probe_single(client, base_url, chain, timeout)

    list_step   = chain.steps[0]
    detail_step = chain.steps[1]

    # Step 1 — probe the list/collection endpoint
    list_path = _fill_params(list_step.path, "1")  # list endpoints rarely have params
    list_url  = base_url.rstrip("/") + list_path

    try:
        status, body, latency = _get(client, list_url, timeout)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            outcome=ProbeOutcome.UNREACHABLE,
            probed_url=list_url,
            error=str(exc),
        )

    if status not in range(200, 300):
        return ProbeResult(
            outcome=_classify(status),
            probed_url=list_url,
            status_code=status,
            latency_ms=latency,
            note="List endpoint blocked — cannot harvest ID for detail probe",
        )

    # Step 2 — extract a real ID and probe the detail endpoint
    harvested_id = _extract_first_id(body)
    if not harvested_id:
        return ProbeResult(
            outcome=ProbeOutcome.NOT_FOUND,
            probed_url=list_url,
            status_code=status,
            latency_ms=latency,
            note="List endpoint returned 2xx but no ID field found in response — cannot complete BOLA probe",
        )

    detail_path = _fill_params(detail_step.path, harvested_id)
    detail_url  = base_url.rstrip("/") + detail_path

    try:
        d_status, _, d_latency = _get(client, detail_url, timeout)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            outcome=ProbeOutcome.UNREACHABLE,
            probed_url=detail_url,
            error=str(exc),
            note=f"List OK, harvested ID '{harvested_id}', detail probe unreachable",
        )

    return ProbeResult(
        outcome=_classify(d_status),
        probed_url=detail_url,
        status_code=d_status,
        latency_ms=d_latency,
        note=f"ID '{harvested_id}' harvested from {list_url} (HTTP {status})",
    )


def _probe_single(
    client: Any,
    base_url: str,
    chain: ValidatedChain,
    timeout: float,
) -> ProbeResult:
    """Single-step probe: hit entry endpoint without credentials."""
    step = chain.steps[0]
    raw_path = step.path.strip()

    # Skip probe if path is empty or root — would hit the server homepage, not an API endpoint
    if not raw_path or raw_path == "/":
        return ProbeResult(
            outcome=ProbeOutcome.NOT_FOUND,
            probed_url=base_url,
            note="Skipped — step path is empty or root. Check that LLM populated AttackStep.path correctly.",
        )

    path = _fill_params(raw_path, "1")
    url  = base_url.rstrip("/") + path

    _WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

    try:
        t0 = time.perf_counter()
        resp = client.request(
            method=step.method,
            url=url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "API-Analyzer-Probe/1.0"},
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # POST/PUT/PATCH sent without a body — 4xx could mean "missing fields"
        # not "missing auth". Cannot distinguish, so mark inconclusive.
        if step.method.upper() in _WRITE_METHODS and 400 <= resp.status_code < 500:
            return ProbeResult(
                outcome=ProbeOutcome.INCONCLUSIVE,
                probed_url=url,
                status_code=resp.status_code,
                latency_ms=latency_ms,
                note=f"{step.method} probe sent without body — HTTP {resp.status_code} could be validation rejection, not auth enforcement",
            )

        return ProbeResult(
            outcome=_classify(resp.status_code),
            probed_url=url,
            status_code=resp.status_code,
            latency_ms=latency_ms,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            outcome=ProbeOutcome.UNREACHABLE,
            probed_url=url,
            error=str(exc),
        )


def probe_chain(
    base_url: str,
    chain: ValidatedChain,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ProbeResult:
    """Dispatch to the right probe strategy based on pattern_id."""
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return ProbeResult(
            outcome=ProbeOutcome.UNREACHABLE,
            probed_url="",
            error="httpx not installed — run: pip install httpx",
        )

    with httpx.Client() as client:
        if chain.pattern_id == _BOLA_PATTERN:
            return _probe_bola(client, base_url, chain, timeout)
        return _probe_single(client, base_url, chain, timeout)


def probe_all(
    base_url: str,
    chains: list[ValidatedChain],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
    max_workers: int = 5,
) -> dict[str, ProbeResult]:
    """Probe all chains concurrently. Returns {chain.id: ProbeResult}."""
    results: dict[str, ProbeResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(probe_chain, base_url, chain, timeout=timeout): chain.id
            for chain in chains
        }
        for fut in as_completed(futures):
            chain_id = futures[fut]
            try:
                results[chain_id] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[chain_id] = ProbeResult(
                    outcome=ProbeOutcome.UNREACHABLE,
                    probed_url="",
                    error=str(exc),
                )
    return results
