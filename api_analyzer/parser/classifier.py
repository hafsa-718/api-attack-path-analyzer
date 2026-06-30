"""Endpoint classifier and spec completeness scorer.

Takes the structural ParsedSpec produced by the parser (M2) and enriches
it with security-relevant classifications that downstream modules depend on:

  ParsedEndpoint.inferred_function  — AUTH | DATA_READ | DATA_WRITE | ADMIN | WEBHOOK | UNKNOWN
  ParsedEndpoint.sensitivity_class  — PUBLIC | INTERNAL | SENSITIVE | CRITICAL
  ParsedSpec.resources              — InferredResource objects for BOLA/IDOR detection in M8
  ParsedSpec.spec_completeness      — float [~0.01, 1.0] fed into ConfidenceBreakdown

Entry point: ``classify(spec)`` — mutates ParsedSpec in place, returns same object.

Classification order
--------------------
inferred_function is set first because sensitivity_class depends on it
(AUTH and ADMIN functions force CRITICAL sensitivity regardless of other signals).

spec_completeness formula
-------------------------
Weighted geometric mean of four components:

  w=0.15  fraction of endpoints with summary/description
  w=0.50  fraction of endpoints with declared auth scheme (auth_scheme_names != [])
  w=0.20  fraction of parameters with schema_type set
  w=0.15  max(fraction of endpoints with response schemas, RESPONSE_SCHEMA_FLOOR)

Per-component floor of 0.01 prevents log(0) in the weighted geometric mean.
The 0.40 response-schema floor avoids double-penalising real-world APIs that
omit response documentation but are otherwise well-specified.

Zero-auth spec (no endpoint has auth_scheme_names) → completeness ≈ 0.08.
This is intentional: a spec without auth declarations provides almost no
signal for the LLM agent to reason about bypass conditions.

Resource inference
------------------
Groups endpoints by URL structure to find REST resource pairs
(collection GET + detail GET with a path parameter).  Only paths with at
least one parameterised detail endpoint become InferredResource objects.
Non-resource paths (health checks, metrics, API docs) are filtered by
a name blocklist.
"""

from __future__ import annotations

import math
import re
from typing import Any

from api_analyzer.models.enums import (
    EndpointFunction,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SensitivityClass,
)
from api_analyzer.models.spec import (
    InferredResource,
    ParsedEndpoint,
    ParsedSpec,
)

# ── spec_completeness constants ────────────────────────────────────────────────

# Weights must sum to 1.0.
_WEIGHTS: tuple[float, float, float, float] = (0.15, 0.50, 0.20, 0.15)
_COMPONENT_FLOOR: float = 0.01
RESPONSE_SCHEMA_FLOOR: float = 0.40

# ── Function classification patterns ──────────────────────────────────────────

# Patterns match a complete path segment (anchored by leading slash and
# followed by slash or end-of-string) to avoid false matches on substrings
# like /authenticator matching AUTH or /administration matching ADMIN.
_AUTH_PATTERN: re.Pattern[str] = re.compile(
    r"/(auth|login|logout|token|password|register|signup|oauth|session|credentials)(/|$)",
    re.IGNORECASE,
)
_ADMIN_PATTERN: re.Pattern[str] = re.compile(
    r"/(admin|internal|management|system|console|debug|superuser|staff|backstage)(/|$)",
    re.IGNORECASE,
)
_WEBHOOK_PATTERN: re.Pattern[str] = re.compile(
    r"/(webhooks?|hooks?|callback|notify|notification)(/|$)",
    re.IGNORECASE,
)

_ADMIN_TAGS: frozenset[str] = frozenset({"admin", "internal", "management", "superuser"})
_AUTH_TAGS: frozenset[str] = frozenset({"auth", "authentication", "authorization", "login", "security"})

_DATA_WRITE_METHODS: frozenset[HttpMethod] = frozenset({
    HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH, HttpMethod.DELETE,
})
_DATA_READ_METHODS: frozenset[HttpMethod] = frozenset({HttpMethod.GET})

# ── Resource inference constants ───────────────────────────────────────────────

_NON_RESOURCE_NAMES: frozenset[str] = frozenset({
    "health", "metrics", "metric", "version", "ping", "status",
    "ready", "live", "readiness", "liveness", "actuator",
    "swagger", "openapi", "docs", "redoc", "api-docs", "apidocs",
    "favicon", "robots",
})

# Version-like path segments to skip when deriving resource names.
_VERSION_SEGMENT: re.Pattern[str] = re.compile(r"^(api|v\d+)$", re.IGNORECASE)


# ── Public entry point ─────────────────────────────────────────────────────────


def classify(spec: ParsedSpec) -> ParsedSpec:
    """Classify endpoints and compute spec_completeness. Mutates spec in place.

    Call order: inferred_function → sensitivity_class → resources → completeness.
    inferred_function must be set before sensitivity_class because AUTH and ADMIN
    functions force CRITICAL sensitivity.
    """
    for ep in spec.endpoints:
        ep.inferred_function = _classify_function(ep)

    for ep in spec.endpoints:
        ep.sensitivity_class = _classify_sensitivity(ep)

    spec.resources = _infer_resources(spec.endpoints)
    spec.spec_completeness = _compute_completeness(spec)
    return spec


# ── Function classification ────────────────────────────────────────────────────


def _classify_function(ep: ParsedEndpoint) -> EndpointFunction:
    """Classify what the endpoint does based on path patterns, tags, and method."""
    path = ep.path
    tags = {t.lower() for t in ep.tags}

    # ADMIN: path segment or tag signals management/internal access.
    if _ADMIN_PATTERN.search(path) or tags & _ADMIN_TAGS:
        return EndpointFunction.ADMIN

    # AUTH: path segment or tag signals authentication/session management.
    if _AUTH_PATTERN.search(path) or tags & _AUTH_TAGS:
        return EndpointFunction.AUTH

    # WEBHOOK: path signals a callback/notification receiver, or the endpoint
    # accepts a URL parameter (the classic SSRF registration pattern).
    if _WEBHOOK_PATTERN.search(path) or ep.accepts_url_param:
        return EndpointFunction.WEBHOOK

    if ep.method in _DATA_READ_METHODS:
        return EndpointFunction.DATA_READ

    if ep.method in _DATA_WRITE_METHODS:
        return EndpointFunction.DATA_WRITE

    return EndpointFunction.UNKNOWN


# ── Sensitivity classification ─────────────────────────────────────────────────


def _classify_sensitivity(ep: ParsedEndpoint) -> SensitivityClass:
    """Assign a sensitivity class based on function, data signals, and auth status.

    Precedence (highest first):
      CRITICAL — auth/admin function OR role manipulation
      SENSITIVE — PII in request/response OR parameterised resource identifier
      INTERNAL — authenticated but no other elevated signals
      PUBLIC   — unauthenticated and no elevated signals
    """
    if (
        ep.inferred_function == EndpointFunction.ADMIN
        or ep.inferred_function == EndpointFunction.AUTH
        or ep.has_role_param
    ):
        return SensitivityClass.CRITICAL

    if (
        ep.returns_pii
        or ep.accepts_pii
        or ep.path_param_type in {PathParamType.INTEGER, PathParamType.UUID}
    ):
        return SensitivityClass.SENSITIVE

    if not ep.is_public:
        return SensitivityClass.INTERNAL

    return SensitivityClass.PUBLIC


# ── spec_completeness ──────────────────────────────────────────────────────────


def _compute_completeness(spec: ParsedSpec) -> float:
    """Weighted geometric mean of four spec quality components.

    Returns 0.0 when the spec has no endpoints (nothing to score).
    """
    endpoints = spec.endpoints
    total = len(endpoints)
    if total == 0:
        return 0.0

    # Component 1: fraction with descriptions (summary or description field set)
    f_descriptions = sum(
        1 for ep in endpoints if ep.summary and ep.summary.strip()
    ) / total

    # Component 2: fraction with explicitly declared auth scheme.
    # auth_scheme_names is populated by M2 from EITHER op-level or global security,
    # so this correctly credits APIs using global security declarations.
    f_auth = sum(1 for ep in endpoints if ep.auth_scheme_names) / total

    # Component 3: fraction of parameters with schema_type set.
    all_params = [p for ep in endpoints for p in ep.parameters]
    if all_params:
        f_params = sum(1 for p in all_params if p.schema_type) / len(all_params)
    else:
        f_params = 1.0  # no parameters — no penalty for missing type info

    # Component 4: fraction of endpoints with at least one response schema.
    # Floored at RESPONSE_SCHEMA_FLOOR to avoid double-penalising APIs that
    # omit response schemas but are otherwise well-specified.
    f_responses = sum(1 for ep in endpoints if ep.response_schemas) / total
    f_responses = max(f_responses, RESPONSE_SCHEMA_FLOOR)

    return round(
        _weighted_geometric_mean((f_descriptions, f_auth, f_params, f_responses)),
        4,
    )


def _weighted_geometric_mean(components: tuple[float, ...]) -> float:
    """Weighted geometric mean: exp(sum(w_i * log(max(c_i, floor))))."""
    return math.exp(
        sum(w * math.log(max(c, _COMPONENT_FLOOR)) for w, c in zip(_WEIGHTS, components))
    )


# ── Resource inference ─────────────────────────────────────────────────────────


def _infer_resources(endpoints: list[ParsedEndpoint]) -> list[InferredResource]:
    """Group endpoints into InferredResource objects by URL path structure.

    A resource requires at least one detail endpoint (a path ending with a
    path parameter, e.g. /users/{userId}).  Endpoints without parameterised
    siblings (e.g. /health) are skipped.
    """
    # Group all endpoints by the collection path for their resource.
    # _collection_path maps both /users and /users/{userId} to /users.
    by_collection: dict[str, list[ParsedEndpoint]] = {}
    for ep in endpoints:
        key = _collection_path(ep.path)
        by_collection.setdefault(key, []).append(ep)

    resources: list[InferredResource] = []
    for col_path, group in by_collection.items():
        resource = _build_resource(col_path, group)
        if resource is not None:
            resources.append(resource)

    return resources


def _collection_path(path: str) -> str:
    """Map a path to its resource group's collection path.

    /users                          → /users
    /users/{userId}                 → /users
    /users/{userId}/orders          → /users/{userId}/orders
    /users/{userId}/orders/{id}     → /users/{userId}/orders
    """
    segments = path.rstrip("/").split("/")
    if segments and segments[-1].startswith("{") and segments[-1].endswith("}"):
        parent = segments[:-1]
        return "/".join(parent) or "/"
    return path


def _build_resource(
    col_path: str, group: list[ParsedEndpoint]
) -> InferredResource | None:
    """Build an InferredResource from a collection path and its endpoint group.

    Returns None if:
    - The path has no parameterised detail endpoints (e.g. /health).
    - The derived resource name is in the non-resource blocklist.
    """
    # Separate collection-level endpoints from detail endpoints.
    collection_eps = [ep for ep in group if ep.path == col_path]
    detail_eps = [ep for ep in group if ep.path != col_path]

    # A resource must have at least one detail endpoint to enable BOLA analysis.
    if not detail_eps:
        return None

    name = _resource_name(col_path)
    if name is None or name.lower() in _NON_RESOURCE_NAMES:
        return None

    collection_get = next(
        (ep for ep in collection_eps if ep.method == HttpMethod.GET), None
    )
    detail_get = next(
        (ep for ep in detail_eps if ep.method == HttpMethod.GET), None
    )
    write_eps = [
        ep for ep in group
        if ep.method in {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH}
    ]
    delete_ep = next(
        (ep for ep in detail_eps if ep.method == HttpMethod.DELETE), None
    )

    # Use the detail GET for identifier info; fall back to any detail endpoint.
    ref = detail_get or detail_eps[0]

    return InferredResource(
        name=name,
        path_prefix=col_path,
        parent_resource_name=_parent_resource_name(col_path),
        collection_endpoint_id=collection_get.id if collection_get else None,
        detail_endpoint_id=detail_get.id if detail_get else None,
        write_endpoint_ids=[ep.id for ep in write_eps],
        delete_endpoint_id=delete_ep.id if delete_ep else None,
        identifier_name=ref.path_param_names[-1] if ref.path_param_names else None,
        identifier_type=ref.path_param_type,
    )


def _resource_name(col_path: str) -> str | None:
    """Derive a title-cased singular resource name from a collection path.

    /users                     → "User"
    /api/v1/products           → "Product"
    /users/{userId}/orders     → "Order"
    /{tenantId}                → None  (no static segment)
    """
    static_segments = [
        s for s in col_path.rstrip("/").split("/")
        if s and not (s.startswith("{") and s.endswith("}"))
    ]
    if not static_segments:
        return None

    # Strip version/api prefix segments (v1, v2, api, etc.).
    meaningful = [s for s in static_segments if not _VERSION_SEGMENT.match(s)]
    last = (meaningful or static_segments)[-1]

    # Check blocklist against the raw segment BEFORE singularizing.
    # Singularization mangles words like "status" → "statu", so the check
    # must happen on the original path segment to reliably filter /health,
    # /status, /docs, /robots, etc.
    if last.lower() in _NON_RESOURCE_NAMES:
        return None

    return _singularize(last).title()


def _parent_resource_name(col_path: str) -> str | None:
    """Return the parent resource name for a sub-resource collection path.

    /users/{userId}/orders  → "User"
    /users                  → None
    """
    segments = col_path.rstrip("/").split("/")
    for i in range(len(segments) - 1, 0, -1):
        if segments[i].startswith("{") and segments[i].endswith("}"):
            parent_segment = segments[i - 1]
            if parent_segment and not _VERSION_SEGMENT.match(parent_segment):
                return _singularize(parent_segment).title()
    return None


def _singularize(word: str) -> str:
    """Best-effort English singularization for REST resource name derivation."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:   # categories → category
        return w[:-3] + "y"
    if w.endswith("ses") or w.endswith("xes") or w.endswith("zes"):  # statuses → status
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:  # users → user
        return w[:-1]
    return w
