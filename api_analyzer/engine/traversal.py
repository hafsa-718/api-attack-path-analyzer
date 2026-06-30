"""Neo4j graph traversal engine (M9).

Entry point
-----------
``traverse(driver, spec_id, *, max_candidates=50) -> TraversalResult``

Orchestrates the full traversal pipeline:
  1. Opens a single Neo4j session for all reads (auto-commit, no transaction overhead)
  2. Fetches spec metadata (completeness score, endpoint count)
  3. Runs all 7 attack-pattern queries (M6) in sequence
  4. Passes all raw candidates into the ranker (M8) → sorted ``CandidateChain`` list
  5. Trims to ``max_candidates`` and returns a ``TraversalResult``

``TraversalResult``
-------------------
Immutable summary of one traversal run.  The reasoning agent (M10) receives
this as input and uses:
  - ``chains``            — the ranked candidates to analyse
  - ``spec_completeness`` — fed into ``ConfidenceBreakdown.auth_clarity_score``
  - ``endpoint_count``    — for chain density logging
  - ``candidate_counts``  — per-pattern breakdown for CLI output / logging
  - ``total_candidates``  — pre-trim count; warns when budget was hit

``max_candidates`` default
--------------------------
50 matches the ``LIMIT 50`` in the auth-chain Cypher query and a conservative
LLM analysis budget.  Pass a higher value (or ``None``-equivalent via a large
int) for exhaustive analysis; pass a lower value for fast preview runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from neo4j import Driver

from api_analyzer.engine.ranker import rank_candidates
from api_analyzer.graph.queries import (
    BolaCandidate,
    ExcessiveDataCandidate,
    find_auth_chain_candidates,
    find_bola_candidates,
    find_broken_auth_candidates,
    find_excessive_data_candidates,
    find_mass_assignment_candidates,
    find_priv_esc_candidates,
    find_ssrf_candidates,
    get_endpoint_count,
    get_spec_completeness,
)
from api_analyzer.models.chain import CandidateChain


@dataclass(frozen=True)
class TraversalResult:
    """Immutable output of one ``traverse()`` call.

    All fields are populated from graph reads and the M8 ranker — no LLM calls
    are made here.  The reasoning agent (M10) receives this and decides which
    chains warrant full analysis.

    ``chains`` is already sorted by ``rank_score`` descending and trimmed to
    ``max_candidates``.  ``total_candidates`` records the pre-trim count so
    callers can detect when the budget was hit.
    """

    spec_id: str
    chains: list[CandidateChain]
    total_candidates: int
    spec_completeness: float
    endpoint_count: int
    candidate_counts: dict[str, int]


# ── Cold-start filters ─────────────────────────────────────────────────────────
#
# A "cold-start" attacker has no credentials and no prior knowledge of resource
# IDs.  These filters drop candidates whose entry endpoint is not reachable from
# that zero-knowledge position — chains that require auth to even begin are not
# cold-startable and produce false positives when the LLM rates them HIGH/CRITICAL.
#
# Patterns NOT filtered here (by design):
#   AP-002 Broken Auth    — query already requires is_public=True
#   AP-003 Priv Esc       — post-auth attack; attacker already has a low-priv session
#   AP-004 Mass Assignment — post-auth attack; same reasoning as AP-003
#   AP-006 SSRF           — high severity even behind auth; LLM notes auth requirement
#   AP-007 Auth Chain     — AUTH endpoints are public by definition


def _cold_start_bola(candidates: list[BolaCandidate]) -> list[BolaCandidate]:
    """Keep only BOLA candidates where the attacker's entry endpoint is public.

    Two-hop chains start at the list endpoint (ID enumeration); single-hop chains
    start at the detail endpoint.  If the entry is not public, the attacker cannot
    reach step 1 without credentials — not cold-startable.
    """
    result = []
    for c in candidates:
        if c.list_endpoint_id is not None:
            if c.list_is_public:
                result.append(c)
        else:
            if c.detail_is_public:
                result.append(c)
    return result


def _cold_start_excessive_data(
    candidates: list[ExcessiveDataCandidate],
) -> list[ExcessiveDataCandidate]:
    """Keep only excessive data candidates that are publicly reachable.

    Over-exposure behind auth is a lower-severity concern (BOLA/BOPA) already
    covered by AP-001.  The standalone AP-005 finding is most actionable when
    unauthenticated callers can receive it.
    """
    return [c for c in candidates if c.is_public]


def traverse(
    driver: Driver,
    spec_id: str,
    *,
    max_candidates: int = 50,
) -> TraversalResult:
    """Run all attack-pattern queries and return a ranked ``TraversalResult``.

    All seven M6 query functions are called within a single Neo4j session.
    The session is auto-commit (read-only); no explicit transaction is opened.
    Candidates from all patterns are ranked together by M8 so that the highest-
    priority findings across all attack types surface first.

    Args:
        driver:         An open ``neo4j.Driver`` instance.
        spec_id:        The stable spec identifier from ``make_spec_id()``.
        max_candidates: Maximum number of ``CandidateChain`` instances to return.
                        Candidates beyond this limit are ranked but discarded.
                        Defaults to 50.

    Returns:
        ``TraversalResult`` with ranked chains, spec metadata, and per-pattern
        candidate counts.

    Raises:
        RuntimeError: If a required attack pattern is missing from the bundled
            YAML (propagated from ``rank_candidates``; indicates a packaging error).
        neo4j.exceptions.ServiceUnavailable: If the Neo4j instance is unreachable.
    """
    with driver.session() as session:
        spec_completeness = get_spec_completeness(session, spec_id)
        endpoint_count = get_endpoint_count(session, spec_id)

        bola = _cold_start_bola(find_bola_candidates(session, spec_id))
        broken_auth = find_broken_auth_candidates(session, spec_id)
        priv_esc = find_priv_esc_candidates(session, spec_id)
        mass_assignment = find_mass_assignment_candidates(session, spec_id)
        excessive_data = _cold_start_excessive_data(find_excessive_data_candidates(session, spec_id))
        ssrf = find_ssrf_candidates(session, spec_id)
        auth_chains = find_auth_chain_candidates(session, spec_id)

    candidate_counts: dict[str, int] = {
        "bola": len(bola),
        "broken_auth": len(broken_auth),
        "priv_esc": len(priv_esc),
        "mass_assignment": len(mass_assignment),
        "excessive_data": len(excessive_data),
        "ssrf": len(ssrf),
        "auth_chains": len(auth_chains),
    }

    all_chains = rank_candidates(
        bola=bola,
        broken_auth=broken_auth,
        priv_esc=priv_esc,
        mass_assignment=mass_assignment,
        excessive_data=excessive_data,
        ssrf=ssrf,
        auth_chains=auth_chains,
    )

    return TraversalResult(
        spec_id=spec_id,
        chains=all_chains[:max_candidates],
        total_candidates=len(all_chains),
        spec_completeness=spec_completeness,
        endpoint_count=endpoint_count,
        candidate_counts=candidate_counts,
    )
