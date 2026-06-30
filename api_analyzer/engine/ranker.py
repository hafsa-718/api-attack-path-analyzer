"""Attack path ranker (M8).

Entry point
-----------
``rank_candidates(**kwargs)`` — accepts lists of M6 candidate dataclasses,
converts each to a ``CandidateChain``, and returns them sorted by
``rank_score`` descending (highest priority first).

Rank score formula
------------------
  rank_score = severity_weight × confidence_base
             × boundary_bonus     (×1.3 if crosses_auth_boundary)
             × sensitivity_bonus  (1.0 + 0.1 × min(sensitivity_delta, 3))

Severity weights:
  CRITICAL → 1.0   HIGH → 0.8   MEDIUM → 0.6   LOW → 0.4   INFO → 0.2

``endpoint_ids`` invariant
--------------------------
``CandidateChain`` requires ``min_length=2`` for ``endpoint_ids``.  Single-
endpoint patterns (AP-002 through AP-006) represent this as
``[endpoint_id, endpoint_id]`` with ``hop_count=1``.  This correctly models a
one-step exploitation (the attacker reaches a single endpoint and the
vulnerability is immediately exploitable) while satisfying the model constraint.

``crosses_auth_boundary``
-------------------------
True only when the chain moves from an unauthenticated context to an
authenticated one.  For AP-007 auth chains this is always True (AUTH entry →
protected target).  For AP-001 BOLA this depends on whether the list endpoint
is public and the detail endpoint is protected.  Single-endpoint patterns that
are themselves public set this to False — there is no boundary to cross when
the endpoint is already unauthenticated.
"""

from __future__ import annotations

import uuid

from api_analyzer.graph.queries import (
    AuthChainCandidate,
    BolaCandidate,
    BrokenAuthCandidate,
    ExcessiveDataCandidate,
    MassAssignmentCandidate,
    PrivEscCandidate,
    SsrfCandidate,
)
from api_analyzer.models.chain import CandidateChain
from api_analyzer.models.enums import Severity, SensitivityClass
from api_analyzer.patterns.loader import AttackPattern, get_pattern

# ── Scoring constants ──────────────────────────────────────────────────────────

_SEVERITY_WEIGHTS: dict[Severity, float] = {
    Severity.CRITICAL: 1.0,
    Severity.HIGH: 0.8,
    Severity.MEDIUM: 0.6,
    Severity.LOW: 0.4,
    Severity.INFO: 0.2,
}

_SENSITIVITY_ORDER: list[SensitivityClass] = list(SensitivityClass)
# Indices: PUBLIC=0, INTERNAL=1, SENSITIVE=2, CRITICAL=3

_BOUNDARY_BONUS: float = 1.3
_SENSITIVITY_BONUS_PER_LEVEL: float = 0.1


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_endpoint_id(endpoint_id: str) -> tuple[str, str]:
    """Split ``'METHOD:path'`` into ``(method, path)``."""
    method, _, path = endpoint_id.partition(":")
    return method, path


def _sensitivity_index(sensitivity_class: str) -> int:
    """Return numeric index for a SensitivityClass string (PUBLIC=0, CRITICAL=3)."""
    return _SENSITIVITY_ORDER.index(SensitivityClass(sensitivity_class))


def _compute_rank_score(
    pattern: AttackPattern,
    crosses_auth_boundary: bool,
    sensitivity_delta: int,
) -> float:
    """Return the composite rank score for a candidate chain.

    All arguments after ``pattern`` are already validated (delta ≥ 0).
    The result is rounded to 4 decimal places.
    """
    score = _SEVERITY_WEIGHTS[pattern.severity] * pattern.confidence_base
    if crosses_auth_boundary:
        score *= _BOUNDARY_BONUS
    score *= 1.0 + _SENSITIVITY_BONUS_PER_LEVEL * min(sensitivity_delta, 3)
    return round(score, 4)


def _pattern(pattern_id: str) -> AttackPattern:
    p = get_pattern(pattern_id)
    if p is None:
        raise RuntimeError(
            f"Attack pattern {pattern_id!r} not found in bundled YAML. "
            "This indicates a packaging error — the patterns data file is missing."
        )
    return p


def _make_chain(
    *,
    pattern: AttackPattern,
    endpoint_ids: list[str],
    crosses_auth_boundary: bool,
    sensitivity_delta: int,
    entry_summary: str,
    exit_summary: str,
) -> CandidateChain:
    delta = max(0, sensitivity_delta)
    return CandidateChain(
        id=str(uuid.uuid4()),
        pattern_id=pattern.id,
        pattern_name=pattern.name,
        owasp_category=pattern.owasp_category,
        mitre_hints=list(pattern.mitre_hints),
        confidence_base=pattern.confidence_base,
        endpoint_ids=endpoint_ids,
        hop_count=len(endpoint_ids) - 1,
        entry_endpoint_id=endpoint_ids[0],
        exit_endpoint_id=endpoint_ids[-1],
        crosses_auth_boundary=crosses_auth_boundary,
        sensitivity_delta=delta,
        rank_score=_compute_rank_score(pattern, crosses_auth_boundary, delta),
        entry_summary=entry_summary,
        exit_summary=exit_summary,
    )


# ── Per-pattern converters ─────────────────────────────────────────────────────


def _bola_to_chain(c: BolaCandidate) -> CandidateChain:
    pattern = _pattern("AP-001")

    # Prefer the list endpoint as entry (provides ID enumeration); fall back to
    # detail when no collection GET exists.
    entry_id = c.list_endpoint_id if c.list_endpoint_id is not None else c.detail_endpoint_id
    entry_method, entry_path = _parse_endpoint_id(entry_id)
    detail_method, detail_path = _parse_endpoint_id(c.detail_endpoint_id)

    if c.list_endpoint_id is not None:
        # Two-hop chain: enumerate IDs → access unauthorized detail record.
        entry_public = bool(c.list_is_public)
        crosses = entry_public and not c.detail_is_public
        entry_si = _sensitivity_index("PUBLIC") if entry_public else _sensitivity_index("INTERNAL")
        delta = max(0, _sensitivity_index(c.detail_sensitivity) - entry_si)
        pub_tag = " (PUBLIC)" if entry_public else ""
        entry_summary = f"{entry_method} {entry_path}{pub_tag}"
    else:
        # Single-hop: direct object access without ID enumeration.
        crosses = False
        delta = 0
        entry_summary = f"{entry_method} {entry_path}"

    exit_summary = f"{detail_method} {detail_path} ({c.detail_sensitivity})"

    return _make_chain(
        pattern=pattern,
        endpoint_ids=[entry_id, c.detail_endpoint_id],
        crosses_auth_boundary=crosses,
        sensitivity_delta=delta,
        entry_summary=entry_summary,
        exit_summary=exit_summary,
    )


def _broken_auth_to_chain(c: BrokenAuthCandidate) -> CandidateChain:
    pattern = _pattern("AP-002")
    method, path = _parse_endpoint_id(c.endpoint_id)
    summary = f"{method} {path} ({c.sensitivity_class})"
    # The endpoint is already public — no boundary to cross.  The vulnerability
    # IS the missing boundary, which the LLM agent will explain in the narrative.
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.endpoint_id, c.endpoint_id],
        crosses_auth_boundary=False,
        sensitivity_delta=_sensitivity_index(c.sensitivity_class),
        entry_summary=summary,
        exit_summary=summary,
    )


def _priv_esc_to_chain(c: PrivEscCandidate) -> CandidateChain:
    pattern = _pattern("AP-003")
    method, path = _parse_endpoint_id(c.endpoint_id)
    summary = f"{method} {path} ({c.sensitivity_class})"
    # Privilege escalation assumes the attacker is already authenticated (low
    # privilege); they elevate within their session, not across an auth boundary.
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.endpoint_id, c.endpoint_id],
        crosses_auth_boundary=False,
        sensitivity_delta=_sensitivity_index(c.sensitivity_class),
        entry_summary=summary,
        exit_summary=summary,
    )


def _mass_assignment_to_chain(c: MassAssignmentCandidate) -> CandidateChain:
    pattern = _pattern("AP-004")
    method, path = _parse_endpoint_id(c.endpoint_id)
    summary = f"{method} {path} → {c.resource_name} ({c.sensitivity_class})"
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.endpoint_id, c.endpoint_id],
        crosses_auth_boundary=False,
        sensitivity_delta=_sensitivity_index(c.sensitivity_class),
        entry_summary=summary,
        exit_summary=summary,
    )


def _excessive_data_to_chain(c: ExcessiveDataCandidate) -> CandidateChain:
    pattern = _pattern("AP-005")
    method, path = _parse_endpoint_id(c.endpoint_id)
    public_tag = " (PUBLIC)" if c.is_public else ""
    summary = f"{method} {path}{public_tag} ({c.sensitivity_class})"
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.endpoint_id, c.endpoint_id],
        crosses_auth_boundary=False,
        sensitivity_delta=_sensitivity_index(c.sensitivity_class),
        entry_summary=summary,
        exit_summary=summary,
    )


def _ssrf_to_chain(c: SsrfCandidate) -> CandidateChain:
    pattern = _pattern("AP-006")
    method, path = _parse_endpoint_id(c.endpoint_id)
    summary = f"{method} {path} ({c.sensitivity_class})"
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.endpoint_id, c.endpoint_id],
        crosses_auth_boundary=False,
        sensitivity_delta=_sensitivity_index(c.sensitivity_class),
        entry_summary=summary,
        exit_summary=summary,
    )


def _auth_chain_to_chain(c: AuthChainCandidate) -> CandidateChain:
    pattern = _pattern("AP-007")
    auth_method, auth_path = _parse_endpoint_id(c.auth_endpoint_id)
    target_method, target_path = _parse_endpoint_id(c.target_endpoint_id)
    # AUTH endpoints are public by definition; they cross directly into the
    # protected target endpoint → auth boundary always crossed.
    # sensitivity_delta: AUTH entry is PUBLIC (index 0) → target sensitivity index.
    delta = _sensitivity_index(c.target_sensitivity)
    return _make_chain(
        pattern=pattern,
        endpoint_ids=[c.auth_endpoint_id, c.target_endpoint_id],
        crosses_auth_boundary=True,
        sensitivity_delta=delta,
        entry_summary=f"{auth_method} {auth_path} (AUTH)",
        exit_summary=f"{target_method} {target_path} ({c.target_sensitivity})",
    )


# ── Public API ─────────────────────────────────────────────────────────────────


def rank_candidates(
    *,
    bola: list[BolaCandidate] | None = None,
    broken_auth: list[BrokenAuthCandidate] | None = None,
    priv_esc: list[PrivEscCandidate] | None = None,
    mass_assignment: list[MassAssignmentCandidate] | None = None,
    excessive_data: list[ExcessiveDataCandidate] | None = None,
    ssrf: list[SsrfCandidate] | None = None,
    auth_chains: list[AuthChainCandidate] | None = None,
) -> list[CandidateChain]:
    """Convert M6 attack candidates into ranked ``CandidateChain`` instances.

    Each candidate type is converted using the corresponding AP-00x pattern from
    the bundled YAML.  The returned list is sorted by ``rank_score`` descending
    (highest priority first).

    All keyword arguments default to ``None``; pass only the candidate types
    your query returned results for.

    Returns:
        Sorted list of ``CandidateChain`` instances.  Empty if all inputs are
        empty or ``None``.

    Raises:
        RuntimeError: If a required attack pattern is missing from the bundled
            YAML (indicates a packaging error, not a user error).
    """
    chains: list[CandidateChain] = []

    for c in bola or []:
        chains.append(_bola_to_chain(c))
    for c in broken_auth or []:
        chains.append(_broken_auth_to_chain(c))
    for c in priv_esc or []:
        chains.append(_priv_esc_to_chain(c))
    for c in mass_assignment or []:
        chains.append(_mass_assignment_to_chain(c))
    for c in excessive_data or []:
        chains.append(_excessive_data_to_chain(c))
    for c in ssrf or []:
        chains.append(_ssrf_to_chain(c))
    for c in auth_chains or []:
        chains.append(_auth_chain_to_chain(c))

    chains.sort(key=lambda ch: ch.rank_score, reverse=True)
    return chains
