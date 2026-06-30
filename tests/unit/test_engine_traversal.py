"""Unit tests for api_analyzer.engine.traversal (M9).

Neo4j is fully mocked via a mock Driver/session.  The seven M6 query
functions are patched at the traversal module level so each test controls
exactly what candidates each pattern returns.

The M8 ranker runs for real (not mocked) so tests also verify that ranking
and trimming work end-to-end.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from api_analyzer.engine.traversal import TraversalResult, traverse
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

# ── Patch target prefix ────────────────────────────────────────────────────────
_MOD = "api_analyzer.engine.traversal"


# ── Mock driver factory ────────────────────────────────────────────────────────

def _driver(session: MagicMock | None = None) -> MagicMock:
    """Return a mock Driver whose context-manager session yields ``session``."""
    if session is None:
        session = MagicMock()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


# ── Candidate factories (reuse from ranker tests) ─────────────────────────────

def _bola(**kw) -> BolaCandidate:
    defaults = dict(
        resource_name="User", identifier_type="INTEGER", path_prefix="/users",
        list_endpoint_id="GET:/users", list_is_public=True,
        detail_endpoint_id="GET:/users/{id}", detail_sensitivity="SENSITIVE",
        detail_is_public=False,
    )
    return BolaCandidate(**{**defaults, **kw})


def _broken_auth(**kw) -> BrokenAuthCandidate:
    defaults = dict(
        endpoint_id="GET:/admin/report", path="/admin/report", method="GET",
        sensitivity_class="CRITICAL", inferred_function="DATA_READ", returns_pii=True,
    )
    return BrokenAuthCandidate(**{**defaults, **kw})


def _auth_chain(**kw) -> AuthChainCandidate:
    defaults = dict(
        auth_endpoint_id="POST:/auth/login", auth_path="/auth/login",
        scheme_name="BearerAuth", target_endpoint_id="GET:/admin/users",
        target_path="/admin/users", target_sensitivity="CRITICAL", target_function="ADMIN",
    )
    return AuthChainCandidate(**{**defaults, **kw})


def _ssrf(**kw) -> SsrfCandidate:
    defaults = dict(
        endpoint_id="POST:/webhooks", path="/webhooks", method="POST",
        is_public=True, sensitivity_class="PUBLIC",
    )
    return SsrfCandidate(**{**defaults, **kw})


# ── Full-patch context manager ─────────────────────────────────────────────────

def _all_empty_patches():
    """Return a dict of patch objects for all 9 query functions, all returning []."""
    return {
        "get_spec_completeness": patch(f"{_MOD}.get_spec_completeness", return_value=0.8),
        "get_endpoint_count": patch(f"{_MOD}.get_endpoint_count", return_value=10),
        "find_bola_candidates": patch(f"{_MOD}.find_bola_candidates", return_value=[]),
        "find_broken_auth_candidates": patch(f"{_MOD}.find_broken_auth_candidates", return_value=[]),
        "find_priv_esc_candidates": patch(f"{_MOD}.find_priv_esc_candidates", return_value=[]),
        "find_mass_assignment_candidates": patch(f"{_MOD}.find_mass_assignment_candidates", return_value=[]),
        "find_excessive_data_candidates": patch(f"{_MOD}.find_excessive_data_candidates", return_value=[]),
        "find_ssrf_candidates": patch(f"{_MOD}.find_ssrf_candidates", return_value=[]),
        "find_auth_chain_candidates": patch(f"{_MOD}.find_auth_chain_candidates", return_value=[]),
    }


# ── TraversalResult dataclass ──────────────────────────────────────────────────


class TestTraversalResult:
    def _make(self) -> TraversalResult:
        return TraversalResult(
            spec_id="api:1.0",
            chains=[],
            total_candidates=0,
            spec_completeness=0.75,
            endpoint_count=5,
            candidate_counts={"bola": 0},
        )

    def test_frozen_raises_on_assignment(self) -> None:
        r = self._make()
        with pytest.raises((AttributeError, TypeError)):
            r.spec_id = "mutated"  # type: ignore[misc]

    def test_all_fields_accessible(self) -> None:
        r = self._make()
        assert r.spec_id == "api:1.0"
        assert r.chains == []
        assert r.total_candidates == 0
        assert r.spec_completeness == 0.75
        assert r.endpoint_count == 5
        assert isinstance(r.candidate_counts, dict)


# ── Driver / session wiring ────────────────────────────────────────────────────


class TestDriverSessionWiring:
    def test_driver_session_opened(self) -> None:
        driver = _driver()
        patches = _all_empty_patches()
        with patch.multiple(_MOD, **{k: v for k, v in patches.items()}):
            for p in patches.values():
                p.start()
            traverse(driver, "api:1.0")
            for p in patches.values():
                p.stop()
        driver.session.assert_called_once()

    def test_context_manager_used(self) -> None:
        session = MagicMock()
        driver = _driver(session)
        patches = _all_empty_patches()
        for p in patches.values():
            p.start()
        try:
            traverse(driver, "api:1.0")
        finally:
            for p in patches.values():
                p.stop()
        driver.session.return_value.__enter__.assert_called_once()
        driver.session.return_value.__exit__.assert_called_once()


# ── Query function dispatch ────────────────────────────────────────────────────


class TestQueryDispatch:
    """Verify each query function is called exactly once with the right args."""

    def _run(self, spec_id: str = "my-api:2.0", **overrides):
        """Run traverse with all queries patched; return mocks dict."""
        session = MagicMock()
        driver = _driver(session)
        mocks: dict[str, MagicMock] = {}

        with (
            patch(f"{_MOD}.get_spec_completeness", return_value=0.5) as m_sc,
            patch(f"{_MOD}.get_endpoint_count", return_value=0) as m_ec,
            patch(f"{_MOD}.find_bola_candidates",
                  return_value=overrides.get("bola", [])) as m_bola,
            patch(f"{_MOD}.find_broken_auth_candidates",
                  return_value=overrides.get("broken_auth", [])) as m_ba,
            patch(f"{_MOD}.find_priv_esc_candidates",
                  return_value=overrides.get("priv_esc", [])) as m_pe,
            patch(f"{_MOD}.find_mass_assignment_candidates",
                  return_value=overrides.get("mass_assignment", [])) as m_ma,
            patch(f"{_MOD}.find_excessive_data_candidates",
                  return_value=overrides.get("excessive_data", [])) as m_ed,
            patch(f"{_MOD}.find_ssrf_candidates",
                  return_value=overrides.get("ssrf", [])) as m_ssrf,
            patch(f"{_MOD}.find_auth_chain_candidates",
                  return_value=overrides.get("auth_chains", [])) as m_ac,
        ):
            result = traverse(driver, spec_id)
            mocks = dict(
                spec_completeness=m_sc, endpoint_count=m_ec,
                bola=m_bola, broken_auth=m_ba, priv_esc=m_pe,
                mass_assignment=m_ma, excessive_data=m_ed,
                ssrf=m_ssrf, auth_chains=m_ac,
            )

        return result, mocks, session

    def test_get_spec_completeness_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["spec_completeness"].assert_called_once_with(session, "my-api:2.0")

    def test_get_endpoint_count_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["endpoint_count"].assert_called_once_with(session, "my-api:2.0")

    def test_find_bola_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["bola"].assert_called_once_with(session, "my-api:2.0")

    def test_find_broken_auth_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["broken_auth"].assert_called_once_with(session, "my-api:2.0")

    def test_find_priv_esc_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["priv_esc"].assert_called_once_with(session, "my-api:2.0")

    def test_find_mass_assignment_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["mass_assignment"].assert_called_once_with(session, "my-api:2.0")

    def test_find_excessive_data_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["excessive_data"].assert_called_once_with(session, "my-api:2.0")

    def test_find_ssrf_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["ssrf"].assert_called_once_with(session, "my-api:2.0")

    def test_find_auth_chains_called_once(self) -> None:
        _, mocks, session = self._run()
        mocks["auth_chains"].assert_called_once_with(session, "my-api:2.0")

    def test_spec_id_forwarded_to_all_queries(self) -> None:
        spec_id = "service:3.1"
        _, mocks, session = self._run(spec_id=spec_id)
        for name, mock in mocks.items():
            if name in ("spec_completeness", "endpoint_count"):
                mock.assert_called_once_with(session, spec_id)
            else:
                mock.assert_called_once_with(session, spec_id)


# ── TraversalResult fields ─────────────────────────────────────────────────────


class TestTraversalResultFields:
    def _traverse_with(
        self,
        spec_id: str = "api:1.0",
        spec_completeness: float = 0.72,
        endpoint_count: int = 8,
        bola=None, broken_auth=None, priv_esc=None,
        mass_assignment=None, excessive_data=None, ssrf=None, auth_chains=None,
        max_candidates: int = 50,
    ) -> TraversalResult:
        driver = _driver()
        with (
            patch(f"{_MOD}.get_spec_completeness", return_value=spec_completeness),
            patch(f"{_MOD}.get_endpoint_count", return_value=endpoint_count),
            patch(f"{_MOD}.find_bola_candidates", return_value=bola or []),
            patch(f"{_MOD}.find_broken_auth_candidates", return_value=broken_auth or []),
            patch(f"{_MOD}.find_priv_esc_candidates", return_value=priv_esc or []),
            patch(f"{_MOD}.find_mass_assignment_candidates", return_value=mass_assignment or []),
            patch(f"{_MOD}.find_excessive_data_candidates", return_value=excessive_data or []),
            patch(f"{_MOD}.find_ssrf_candidates", return_value=ssrf or []),
            patch(f"{_MOD}.find_auth_chain_candidates", return_value=auth_chains or []),
        ):
            return traverse(driver, spec_id, max_candidates=max_candidates)

    def test_returns_traversal_result(self) -> None:
        assert isinstance(self._traverse_with(), TraversalResult)

    def test_spec_id_propagated(self) -> None:
        r = self._traverse_with(spec_id="shop:2.0")
        assert r.spec_id == "shop:2.0"

    def test_spec_completeness_from_query(self) -> None:
        r = self._traverse_with(spec_completeness=0.88)
        assert r.spec_completeness == pytest.approx(0.88)

    def test_endpoint_count_from_query(self) -> None:
        r = self._traverse_with(endpoint_count=42)
        assert r.endpoint_count == 42

    def test_empty_spec_returns_empty_chains(self) -> None:
        r = self._traverse_with()
        assert r.chains == []
        assert r.total_candidates == 0

    def test_chains_are_candidate_chain_instances(self) -> None:
        r = self._traverse_with(bola=[_bola()])
        assert all(isinstance(c, CandidateChain) for c in r.chains)

    def test_total_candidates_counts_all_types(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
        )
        assert r.total_candidates == 3

    def test_candidate_counts_dict_has_all_keys(self) -> None:
        r = self._traverse_with()
        expected_keys = {
            "bola", "broken_auth", "priv_esc", "mass_assignment",
            "excessive_data", "ssrf", "auth_chains",
        }
        assert set(r.candidate_counts.keys()) == expected_keys

    def test_candidate_counts_reflect_input_lengths(self) -> None:
        r = self._traverse_with(
            bola=[_bola(), _bola(resource_name="Order")],
            ssrf=[_ssrf()],
        )
        assert r.candidate_counts["bola"] == 2
        assert r.candidate_counts["ssrf"] == 1
        assert r.candidate_counts["broken_auth"] == 0

    def test_candidate_counts_sum_equals_total_candidates(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
        )
        assert sum(r.candidate_counts.values()) == r.total_candidates


# ── Ranking and trimming ───────────────────────────────────────────────────────


class TestRankingAndTrimming:
    def _traverse_with(self, **kw) -> TraversalResult:
        driver = _driver()
        with (
            patch(f"{_MOD}.get_spec_completeness", return_value=0.5),
            patch(f"{_MOD}.get_endpoint_count", return_value=5),
            patch(f"{_MOD}.find_bola_candidates", return_value=kw.get("bola", [])),
            patch(f"{_MOD}.find_broken_auth_candidates", return_value=kw.get("broken_auth", [])),
            patch(f"{_MOD}.find_priv_esc_candidates", return_value=kw.get("priv_esc", [])),
            patch(f"{_MOD}.find_mass_assignment_candidates", return_value=kw.get("mass_assignment", [])),
            patch(f"{_MOD}.find_excessive_data_candidates", return_value=kw.get("excessive_data", [])),
            patch(f"{_MOD}.find_ssrf_candidates", return_value=kw.get("ssrf", [])),
            patch(f"{_MOD}.find_auth_chain_candidates", return_value=kw.get("auth_chains", [])),
        ):
            return traverse(driver, "api:1.0", max_candidates=kw.get("max_candidates", 50))

    def test_chains_sorted_descending_by_rank_score(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            ssrf=[_ssrf()],
        )
        scores = [c.rank_score for c in r.chains]
        assert scores == sorted(scores, reverse=True)

    def test_max_candidates_trims_result(self) -> None:
        # 3 candidates, max=2 → only 2 returned
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
            max_candidates=2,
        )
        assert len(r.chains) == 2

    def test_total_candidates_reflects_pre_trim_count(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
            max_candidates=2,
        )
        assert r.total_candidates == 3

    def test_trimmed_chains_are_highest_ranked(self) -> None:
        # Build 3 candidates; the trim keeps the top 2
        r = self._traverse_with(
            bola=[_bola()],
            broken_auth=[_broken_auth(sensitivity_class="CRITICAL")],
            ssrf=[_ssrf(sensitivity_class="PUBLIC")],
            max_candidates=2,
        )
        all_scores_sorted = sorted(
            [c.rank_score for c in r.chains], reverse=True
        )
        assert [c.rank_score for c in r.chains] == all_scores_sorted

    def test_max_candidates_1_returns_single_chain(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            ssrf=[_ssrf()],
            max_candidates=1,
        )
        assert len(r.chains) == 1

    def test_max_candidates_larger_than_count_returns_all(self) -> None:
        r = self._traverse_with(
            bola=[_bola()],
            ssrf=[_ssrf()],
            max_candidates=100,
        )
        assert len(r.chains) == 2
        assert r.total_candidates == 2

    def test_default_max_candidates_is_50(self) -> None:
        # Generate 60 candidates (60 bola with different resource names)
        many_bola = [_bola(resource_name=f"Resource{i}") for i in range(60)]
        r = self._traverse_with(bola=many_bola)
        assert len(r.chains) == 50
        assert r.total_candidates == 60

    def test_auth_chain_critical_appears_before_ssrf_public(self) -> None:
        r = self._traverse_with(
            ssrf=[_ssrf(sensitivity_class="PUBLIC")],
            auth_chains=[_auth_chain(target_sensitivity="CRITICAL")],
        )
        pattern_ids = [c.pattern_id for c in r.chains]
        assert pattern_ids[0] == "AP-007"

    def test_chains_have_unique_ids(self) -> None:
        r = self._traverse_with(
            bola=[_bola(), _bola(resource_name="Order")],
            ssrf=[_ssrf()],
        )
        ids = [c.id for c in r.chains]
        assert len(ids) == len(set(ids))
