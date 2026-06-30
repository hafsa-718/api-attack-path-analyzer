"""Cypher query helpers for attack-pattern candidate discovery.

Each function accepts a ``neo4j.Session`` and a ``spec_id``, runs a parameterised
Cypher query against the graph built by M5, and returns a typed list of
candidate dataclasses.  No raw Neo4j Records leak out of this module.

Attack pattern coverage
-----------------------
  AP-001  BOLA/IDOR              find_bola_candidates
  AP-002  Broken Authentication  find_broken_auth_candidates
  AP-003  Privilege Escalation   find_priv_esc_candidates
  AP-004  Mass Assignment        find_mass_assignment_candidates
  AP-005  Excessive Data Expos.  find_excessive_data_candidates
  AP-006  SSRF                   find_ssrf_candidates
  AP-007  Auth-chain             find_auth_chain_candidates  (credential theft → data access)

Lookup helpers
--------------
  get_spec_completeness  — float used as auth_clarity_score in ConfidenceBreakdown
  get_endpoint_count     — total endpoints in spec (for chain density metrics)

Design notes
------------
- All Cypher templates are module-level constants, computed once at import time
  using M4 LABEL_* / PROP_* / REL_* constants.  This mirrors M5's approach and
  ensures a single rename in schema.py propagates everywhere.

- Queries use $spec_id parameter, not string interpolation, for every
  user-supplied value.  Label and property names are M4 module constants
  that are never user-supplied.

- Functions accept a Session (for auto-commit reads) rather than a
  ManagedTransaction so callers do not need to manage transaction scope.
  M9 (traversal) opens sessions; M6 simply runs queries within them.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Session

from api_analyzer.graph.schema import (
    LABEL_API_SPEC,
    LABEL_AUTH_SCHEME,
    LABEL_ENDPOINT,
    LABEL_RESOURCE,
    PROP_ACCEPTS_PII,
    PROP_ACCEPTS_URL_PARAM,
    PROP_HAS_ROLE_PARAM,
    PROP_ID,
    PROP_IDENTIFIER_TYPE,
    PROP_INFERRED_FUNCTION,
    PROP_IS_PUBLIC,
    PROP_METHOD,
    PROP_NAME,
    PROP_PATH,
    PROP_PATH_PREFIX,
    PROP_RETURNS_PII,
    PROP_SENSITIVITY_CLASS,
    PROP_SPEC_COMPLETENESS,
    PROP_SPEC_ID,
    REL_LISTS,
    REL_READS,
    REL_REQUIRES_AUTH,
    REL_WRITES,
)

# ── Return-type dataclasses ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BolaCandidate:
    """BOLA/IDOR candidate (AP-001).

    A resource with a predictable identifier (INTEGER or UUID) where there is
    at least one READS endpoint.  The optional LISTS endpoint provides ID
    enumeration, making exploitation trivial; its absence raises the bar but
    doesn't eliminate the risk.
    """

    resource_name: str
    identifier_type: str          # "INTEGER" or "UUID"
    path_prefix: str
    list_endpoint_id: str | None  # LISTS ep; None if only direct access exists
    list_is_public: bool | None   # None when list_endpoint_id is None
    detail_endpoint_id: str       # READS ep; always present (required for BOLA)
    detail_sensitivity: str
    detail_is_public: bool


@dataclass(frozen=True)
class BrokenAuthCandidate:
    """Broken authentication candidate (AP-002).

    A PUBLIC endpoint with SENSITIVE or CRITICAL sensitivity — an endpoint
    that should require authentication but currently does not.
    """

    endpoint_id: str
    path: str
    method: str
    sensitivity_class: str
    inferred_function: str
    returns_pii: bool


@dataclass(frozen=True)
class PrivEscCandidate:
    """Privilege escalation candidate (AP-003).

    An endpoint that accepts role / permission / group parameters in its
    request body, enabling an attacker to self-assign elevated privileges.
    """

    endpoint_id: str
    path: str
    method: str
    sensitivity_class: str
    inferred_function: str
    is_public: bool


@dataclass(frozen=True)
class MassAssignmentCandidate:
    """Mass assignment candidate (AP-004).

    A POST/PUT/PATCH endpoint connected to an inferred Resource via a WRITES
    relationship.  The graph does not store full request body schemas, but the
    combination of DATA_WRITE function + resource association flags this as a
    candidate for mass assignment analysis by the LLM agent.
    """

    endpoint_id: str
    path: str
    method: str
    sensitivity_class: str
    accepts_pii: bool
    has_role_param: bool
    resource_name: str


@dataclass(frozen=True)
class ExcessiveDataCandidate:
    """Excessive data exposure candidate (AP-005).

    An endpoint that returns PII-classified fields.  Particularly severe when
    combined with is_public=True or low sensitivity_class.
    """

    endpoint_id: str
    path: str
    method: str
    sensitivity_class: str
    is_public: bool
    inferred_function: str


@dataclass(frozen=True)
class SsrfCandidate:
    """SSRF candidate (AP-006).

    An endpoint that accepts a URL-valued parameter, enabling an attacker to
    cause the server to make arbitrary outbound HTTP requests.
    """

    endpoint_id: str
    path: str
    method: str
    is_public: bool
    sensitivity_class: str


@dataclass(frozen=True)
class AuthChainCandidate:
    """Credential-theft → data-access chain candidate (AP-007).

    Pairs an AUTH-function endpoint (the attacker's entry point for credential
    theft) with a SENSITIVE/CRITICAL target endpoint protected by a specific
    auth scheme.  The connection is: compromising the auth mechanism for
    ``scheme_name`` grants access to ``target_endpoint_id``.
    """

    auth_endpoint_id: str
    auth_path: str
    scheme_name: str
    target_endpoint_id: str
    target_path: str
    target_sensitivity: str
    target_function: str


# ── Cypher templates ───────────────────────────────────────────────────────────
# Computed once at import time using M4 constants.  Only $spec_id is
# parameterised at query time — never user-controlled label/property names.

_CQL_SPEC_COMPLETENESS: str = (
    f"MATCH (s:{LABEL_API_SPEC} {{{PROP_ID}: $spec_id}}) "
    f"RETURN s.{PROP_SPEC_COMPLETENESS} AS completeness"
)

_CQL_ENDPOINT_COUNT: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id}}) "
    f"RETURN count(e) AS endpoint_count"
)

# AP-001: BOLA — resource with predictable identifier + READS endpoint
# OPTIONAL MATCH for list_ep so resources without a collection GET are included.
_CQL_BOLA: str = (
    f"MATCH (r:{LABEL_RESOURCE} {{{PROP_SPEC_ID}: $spec_id}}) "
    f"WHERE r.{PROP_IDENTIFIER_TYPE} IN ['INTEGER', 'UUID'] "
    f"OPTIONAL MATCH (list_ep:{LABEL_ENDPOINT})-[:{REL_LISTS}]->(r) "
    f"MATCH (detail_ep:{LABEL_ENDPOINT})-[:{REL_READS}]->(r) "
    f"RETURN r.{PROP_NAME} AS resource_name, "
    f"       r.{PROP_IDENTIFIER_TYPE} AS identifier_type, "
    f"       r.{PROP_PATH_PREFIX} AS path_prefix, "
    f"       list_ep.{PROP_ID} AS list_endpoint_id, "
    f"       list_ep.{PROP_IS_PUBLIC} AS list_is_public, "
    f"       detail_ep.{PROP_ID} AS detail_endpoint_id, "
    f"       detail_ep.{PROP_SENSITIVITY_CLASS} AS detail_sensitivity, "
    f"       detail_ep.{PROP_IS_PUBLIC} AS detail_is_public"
)

# AP-002: Broken auth — public endpoint with elevated sensitivity
_CQL_BROKEN_AUTH: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id, {PROP_IS_PUBLIC}: true}}) "
    f"WHERE e.{PROP_SENSITIVITY_CLASS} IN ['SENSITIVE', 'CRITICAL'] "
    f"RETURN e.{PROP_ID} AS endpoint_id, "
    f"       e.{PROP_PATH} AS path, "
    f"       e.{PROP_METHOD} AS method, "
    f"       e.{PROP_SENSITIVITY_CLASS} AS sensitivity_class, "
    f"       e.{PROP_INFERRED_FUNCTION} AS inferred_function, "
    f"       e.{PROP_RETURNS_PII} AS returns_pii"
)

# AP-003: Privilege escalation — endpoint accepting role/permission params
_CQL_PRIV_ESC: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id, {PROP_HAS_ROLE_PARAM}: true}}) "
    f"RETURN e.{PROP_ID} AS endpoint_id, "
    f"       e.{PROP_PATH} AS path, "
    f"       e.{PROP_METHOD} AS method, "
    f"       e.{PROP_SENSITIVITY_CLASS} AS sensitivity_class, "
    f"       e.{PROP_INFERRED_FUNCTION} AS inferred_function, "
    f"       e.{PROP_IS_PUBLIC} AS is_public"
)

# AP-004: Mass assignment — write endpoint connected to an inferred resource
_CQL_MASS_ASSIGNMENT: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id}})"
    f"-[:{REL_WRITES}]->"
    f"(r:{LABEL_RESOURCE} {{{PROP_SPEC_ID}: $spec_id}}) "
    f"WHERE e.{PROP_METHOD} IN ['POST', 'PUT', 'PATCH'] "
    f"RETURN e.{PROP_ID} AS endpoint_id, "
    f"       e.{PROP_PATH} AS path, "
    f"       e.{PROP_METHOD} AS method, "
    f"       e.{PROP_SENSITIVITY_CLASS} AS sensitivity_class, "
    f"       e.{PROP_ACCEPTS_PII} AS accepts_pii, "
    f"       e.{PROP_HAS_ROLE_PARAM} AS has_role_param, "
    f"       r.{PROP_NAME} AS resource_name"
)

# AP-005: Excessive data exposure — any endpoint returning PII fields
_CQL_EXCESSIVE_DATA: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id, {PROP_RETURNS_PII}: true}}) "
    f"RETURN e.{PROP_ID} AS endpoint_id, "
    f"       e.{PROP_PATH} AS path, "
    f"       e.{PROP_METHOD} AS method, "
    f"       e.{PROP_SENSITIVITY_CLASS} AS sensitivity_class, "
    f"       e.{PROP_IS_PUBLIC} AS is_public, "
    f"       e.{PROP_INFERRED_FUNCTION} AS inferred_function"
)

# AP-006: SSRF — endpoint that accepts a URL-valued parameter
_CQL_SSRF: str = (
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id, {PROP_ACCEPTS_URL_PARAM}: true}}) "
    f"RETURN e.{PROP_ID} AS endpoint_id, "
    f"       e.{PROP_PATH} AS path, "
    f"       e.{PROP_METHOD} AS method, "
    f"       e.{PROP_IS_PUBLIC} AS is_public, "
    f"       e.{PROP_SENSITIVITY_CLASS} AS sensitivity_class"
)

# AP-007: Auth chain — AUTH-function endpoint paired with sensitive protected endpoint.
# The chain link: exploiting the auth mechanism grants the scheme credential
# that protects the target endpoint.  LIMIT 50 prevents O(n²) explosion.
_CQL_AUTH_CHAINS: str = (
    f"MATCH (auth_ep:{LABEL_ENDPOINT} "
    f"       {{{PROP_SPEC_ID}: $spec_id, {PROP_INFERRED_FUNCTION}: 'AUTH'}}) "
    f"MATCH (target:{LABEL_ENDPOINT} {{{PROP_SPEC_ID}: $spec_id}}) "
    f"WHERE target.{PROP_SENSITIVITY_CLASS} IN ['SENSITIVE', 'CRITICAL'] "
    f"  AND target.{PROP_IS_PUBLIC} = false "
    f"  AND target.{PROP_ID} <> auth_ep.{PROP_ID} "
    f"MATCH (target)-[:{REL_REQUIRES_AUTH}]->"
    f"      (scheme:{LABEL_AUTH_SCHEME} {{{PROP_SPEC_ID}: $spec_id}}) "
    f"RETURN auth_ep.{PROP_ID} AS auth_endpoint_id, "
    f"       auth_ep.{PROP_PATH} AS auth_path, "
    f"       scheme.{PROP_NAME} AS scheme_name, "
    f"       target.{PROP_ID} AS target_endpoint_id, "
    f"       target.{PROP_PATH} AS target_path, "
    f"       target.{PROP_SENSITIVITY_CLASS} AS target_sensitivity, "
    f"       target.{PROP_INFERRED_FUNCTION} AS target_function "
    f"LIMIT 50"
)


# ── Lookup helpers ─────────────────────────────────────────────────────────────


def get_spec_completeness(session: Session, spec_id: str) -> float:
    """Return the spec_completeness score for a parsed spec.

    Used by M9 as the ``auth_clarity_score`` base in ``ConfidenceBreakdown``.
    Returns 0.0 if the spec node is not found (graph not yet built).
    """
    result = session.run(_CQL_SPEC_COMPLETENESS, spec_id=spec_id)
    record = result.single()
    if record is None:
        return 0.0
    value = record["completeness"]
    return float(value) if value is not None else 0.0


def get_endpoint_count(session: Session, spec_id: str) -> int:
    """Return the total number of endpoint nodes for a spec."""
    result = session.run(_CQL_ENDPOINT_COUNT, spec_id=spec_id)
    record = result.single()
    if record is None:
        return 0
    return int(record["endpoint_count"])


# ── Attack pattern query functions ─────────────────────────────────────────────


def find_bola_candidates(session: Session, spec_id: str) -> list[BolaCandidate]:
    """Find BOLA/IDOR candidates: resources with predictable identifier types.

    Returns one candidate per (resource, detail_endpoint) pair.  If the
    resource has both a list endpoint and a detail endpoint, the list endpoint
    provides ID enumeration — a higher-severity BOLA.
    """
    result = session.run(_CQL_BOLA, spec_id=spec_id)
    return [
        BolaCandidate(
            resource_name=r["resource_name"],
            identifier_type=r["identifier_type"],
            path_prefix=r["path_prefix"],
            list_endpoint_id=r["list_endpoint_id"],
            list_is_public=r["list_is_public"],
            detail_endpoint_id=r["detail_endpoint_id"],
            detail_sensitivity=r["detail_sensitivity"],
            detail_is_public=r["detail_is_public"],
        )
        for r in result
    ]


def find_broken_auth_candidates(
    session: Session, spec_id: str
) -> list[BrokenAuthCandidate]:
    """Find broken auth candidates: public endpoints with elevated sensitivity."""
    result = session.run(_CQL_BROKEN_AUTH, spec_id=spec_id)
    return [
        BrokenAuthCandidate(
            endpoint_id=r["endpoint_id"],
            path=r["path"],
            method=r["method"],
            sensitivity_class=r["sensitivity_class"],
            inferred_function=r["inferred_function"],
            returns_pii=bool(r["returns_pii"]),
        )
        for r in result
    ]


def find_priv_esc_candidates(
    session: Session, spec_id: str
) -> list[PrivEscCandidate]:
    """Find privilege escalation candidates: endpoints accepting role parameters."""
    result = session.run(_CQL_PRIV_ESC, spec_id=spec_id)
    return [
        PrivEscCandidate(
            endpoint_id=r["endpoint_id"],
            path=r["path"],
            method=r["method"],
            sensitivity_class=r["sensitivity_class"],
            inferred_function=r["inferred_function"],
            is_public=bool(r["is_public"]),
        )
        for r in result
    ]


def find_mass_assignment_candidates(
    session: Session, spec_id: str
) -> list[MassAssignmentCandidate]:
    """Find mass assignment candidates: write endpoints connected to resources."""
    result = session.run(_CQL_MASS_ASSIGNMENT, spec_id=spec_id)
    return [
        MassAssignmentCandidate(
            endpoint_id=r["endpoint_id"],
            path=r["path"],
            method=r["method"],
            sensitivity_class=r["sensitivity_class"],
            accepts_pii=bool(r["accepts_pii"]),
            has_role_param=bool(r["has_role_param"]),
            resource_name=r["resource_name"],
        )
        for r in result
    ]


def find_excessive_data_candidates(
    session: Session, spec_id: str
) -> list[ExcessiveDataCandidate]:
    """Find excessive data exposure candidates: endpoints returning PII fields."""
    result = session.run(_CQL_EXCESSIVE_DATA, spec_id=spec_id)
    return [
        ExcessiveDataCandidate(
            endpoint_id=r["endpoint_id"],
            path=r["path"],
            method=r["method"],
            sensitivity_class=r["sensitivity_class"],
            is_public=bool(r["is_public"]),
            inferred_function=r["inferred_function"],
        )
        for r in result
    ]


def find_ssrf_candidates(session: Session, spec_id: str) -> list[SsrfCandidate]:
    """Find SSRF candidates: endpoints accepting URL-valued parameters."""
    result = session.run(_CQL_SSRF, spec_id=spec_id)
    return [
        SsrfCandidate(
            endpoint_id=r["endpoint_id"],
            path=r["path"],
            method=r["method"],
            is_public=bool(r["is_public"]),
            sensitivity_class=r["sensitivity_class"],
        )
        for r in result
    ]


def find_auth_chain_candidates(
    session: Session, spec_id: str
) -> list[AuthChainCandidate]:
    """Find credential-theft → data-access chain candidates.

    Pairs each AUTH-function endpoint with sensitive protected endpoints that
    share an auth scheme.  The resulting chain represents: exploit the auth
    mechanism → use stolen credentials → access sensitive data.
    """
    result = session.run(_CQL_AUTH_CHAINS, spec_id=spec_id)
    return [
        AuthChainCandidate(
            auth_endpoint_id=r["auth_endpoint_id"],
            auth_path=r["auth_path"],
            scheme_name=r["scheme_name"],
            target_endpoint_id=r["target_endpoint_id"],
            target_path=r["target_path"],
            target_sensitivity=r["target_sensitivity"],
            target_function=r["target_function"],
        )
        for r in result
    ]
