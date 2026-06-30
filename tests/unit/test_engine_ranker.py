"""Unit tests for api_analyzer.engine.ranker (M8).

No Neo4j or Anthropic calls.  All candidates are constructed directly from
the M6 dataclasses.  Pattern data comes from the real bundled YAML via
load_patterns() — tests verify the formula with actual pattern values.
"""

from __future__ import annotations

import pytest

from api_analyzer.engine.ranker import (
    _BOUNDARY_BONUS,
    _SENSITIVITY_BONUS_PER_LEVEL,
    _SEVERITY_WEIGHTS,
    _auth_chain_to_chain,
    _bola_to_chain,
    _broken_auth_to_chain,
    _compute_rank_score,
    _excessive_data_to_chain,
    _mass_assignment_to_chain,
    _parse_endpoint_id,
    _priv_esc_to_chain,
    _sensitivity_index,
    _ssrf_to_chain,
    rank_candidates,
)
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
from api_analyzer.models.enums import Severity
from api_analyzer.patterns.loader import get_pattern


# ── Candidate factories ────────────────────────────────────────────────────────


def _bola(
    resource_name: str = "User",
    identifier_type: str = "INTEGER",
    path_prefix: str = "/users",
    list_endpoint_id: str | None = "GET:/users",
    list_is_public: bool | None = True,
    detail_endpoint_id: str = "GET:/users/{id}",
    detail_sensitivity: str = "SENSITIVE",
    detail_is_public: bool = False,
) -> BolaCandidate:
    return BolaCandidate(
        resource_name=resource_name,
        identifier_type=identifier_type,
        path_prefix=path_prefix,
        list_endpoint_id=list_endpoint_id,
        list_is_public=list_is_public,
        detail_endpoint_id=detail_endpoint_id,
        detail_sensitivity=detail_sensitivity,
        detail_is_public=detail_is_public,
    )


def _broken_auth(
    endpoint_id: str = "GET:/admin/report",
    path: str = "/admin/report",
    method: str = "GET",
    sensitivity_class: str = "SENSITIVE",
    inferred_function: str = "DATA_READ",
    returns_pii: bool = True,
) -> BrokenAuthCandidate:
    return BrokenAuthCandidate(
        endpoint_id=endpoint_id,
        path=path,
        method=method,
        sensitivity_class=sensitivity_class,
        inferred_function=inferred_function,
        returns_pii=returns_pii,
    )


def _priv_esc(
    endpoint_id: str = "PATCH:/users/{id}",
    path: str = "/users/{id}",
    method: str = "PATCH",
    sensitivity_class: str = "CRITICAL",
    inferred_function: str = "DATA_WRITE",
    is_public: bool = False,
) -> PrivEscCandidate:
    return PrivEscCandidate(
        endpoint_id=endpoint_id,
        path=path,
        method=method,
        sensitivity_class=sensitivity_class,
        inferred_function=inferred_function,
        is_public=is_public,
    )


def _mass_assignment(
    endpoint_id: str = "PUT:/users/{id}",
    path: str = "/users/{id}",
    method: str = "PUT",
    sensitivity_class: str = "SENSITIVE",
    accepts_pii: bool = True,
    has_role_param: bool = True,
    resource_name: str = "User",
) -> MassAssignmentCandidate:
    return MassAssignmentCandidate(
        endpoint_id=endpoint_id,
        path=path,
        method=method,
        sensitivity_class=sensitivity_class,
        accepts_pii=accepts_pii,
        has_role_param=has_role_param,
        resource_name=resource_name,
    )


def _excessive_data(
    endpoint_id: str = "GET:/users/{id}",
    path: str = "/users/{id}",
    method: str = "GET",
    sensitivity_class: str = "SENSITIVE",
    is_public: bool = False,
    inferred_function: str = "DATA_READ",
) -> ExcessiveDataCandidate:
    return ExcessiveDataCandidate(
        endpoint_id=endpoint_id,
        path=path,
        method=method,
        sensitivity_class=sensitivity_class,
        is_public=is_public,
        inferred_function=inferred_function,
    )


def _ssrf(
    endpoint_id: str = "POST:/webhooks",
    path: str = "/webhooks",
    method: str = "POST",
    is_public: bool = True,
    sensitivity_class: str = "PUBLIC",
) -> SsrfCandidate:
    return SsrfCandidate(
        endpoint_id=endpoint_id,
        path=path,
        method=method,
        is_public=is_public,
        sensitivity_class=sensitivity_class,
    )


def _auth_chain(
    auth_endpoint_id: str = "POST:/auth/login",
    auth_path: str = "/auth/login",
    scheme_name: str = "BearerAuth",
    target_endpoint_id: str = "GET:/admin/users",
    target_path: str = "/admin/users",
    target_sensitivity: str = "CRITICAL",
    target_function: str = "ADMIN",
) -> AuthChainCandidate:
    return AuthChainCandidate(
        auth_endpoint_id=auth_endpoint_id,
        auth_path=auth_path,
        scheme_name=scheme_name,
        target_endpoint_id=target_endpoint_id,
        target_path=target_path,
        target_sensitivity=target_sensitivity,
        target_function=target_function,
    )


# ── _parse_endpoint_id ─────────────────────────────────────────────────────────


class TestParseEndpointId:
    def test_basic_split(self) -> None:
        assert _parse_endpoint_id("GET:/users") == ("GET", "/users")

    def test_path_with_param(self) -> None:
        assert _parse_endpoint_id("DELETE:/users/{id}") == ("DELETE", "/users/{id}")

    def test_post_nested(self) -> None:
        assert _parse_endpoint_id("POST:/auth/login") == ("POST", "/auth/login")

    def test_only_first_colon_splits(self) -> None:
        # Paths that contain colons (unusual but possible) must not be split further
        method, path = _parse_endpoint_id("GET:/a:b")
        assert method == "GET"
        assert path == "/a:b"

    def test_patch_method(self) -> None:
        m, p = _parse_endpoint_id("PATCH:/items/{id}")
        assert m == "PATCH"
        assert p == "/items/{id}"


# ── _sensitivity_index ─────────────────────────────────────────────────────────


class TestSensitivityIndex:
    def test_public_is_zero(self) -> None:
        assert _sensitivity_index("PUBLIC") == 0

    def test_internal_is_one(self) -> None:
        assert _sensitivity_index("INTERNAL") == 1

    def test_sensitive_is_two(self) -> None:
        assert _sensitivity_index("SENSITIVE") == 2

    def test_critical_is_three(self) -> None:
        assert _sensitivity_index("CRITICAL") == 3

    def test_ordering_is_strictly_increasing(self) -> None:
        idx = [_sensitivity_index(c) for c in ("PUBLIC", "INTERNAL", "SENSITIVE", "CRITICAL")]
        assert idx == sorted(idx)
        assert len(set(idx)) == 4  # all distinct


# ── _compute_rank_score ────────────────────────────────────────────────────────


class TestComputeRankScore:
    def _ap007(self):  # CRITICAL severity, confidence_base=0.70
        return get_pattern("AP-007")

    def _ap005(self):  # MEDIUM severity, confidence_base=0.50
        return get_pattern("AP-005")

    def test_no_bonus_formula(self) -> None:
        p = self._ap005()
        assert p is not None
        expected = round(0.6 * 0.50 * 1.0, 4)
        assert _compute_rank_score(p, False, 0) == pytest.approx(expected)

    def test_boundary_bonus_applied(self) -> None:
        p = self._ap007()
        assert p is not None
        base = 1.0 * 0.70
        expected = round(base * _BOUNDARY_BONUS, 4)
        assert _compute_rank_score(p, True, 0) == pytest.approx(expected)

    def test_sensitivity_bonus_applied(self) -> None:
        p = self._ap005()
        assert p is not None
        base = 0.6 * 0.50
        expected = round(base * (1.0 + _SENSITIVITY_BONUS_PER_LEVEL * 2), 4)
        assert _compute_rank_score(p, False, 2) == pytest.approx(expected)

    def test_both_bonuses_combined(self) -> None:
        p = self._ap007()
        assert p is not None
        base = 1.0 * 0.70
        expected = round(base * _BOUNDARY_BONUS * (1.0 + _SENSITIVITY_BONUS_PER_LEVEL * 3), 4)
        assert _compute_rank_score(p, True, 3) == pytest.approx(expected)

    def test_sensitivity_delta_capped_at_3(self) -> None:
        p = self._ap007()
        assert p is not None
        # delta=5 should give same result as delta=3
        assert _compute_rank_score(p, False, 5) == _compute_rank_score(p, False, 3)

    def test_result_is_non_negative(self) -> None:
        for pid in ("AP-001", "AP-002", "AP-003", "AP-004", "AP-005", "AP-006", "AP-007"):
            p = get_pattern(pid)
            assert p is not None
            assert _compute_rank_score(p, False, 0) >= 0.0

    def test_critical_outranks_medium_same_conditions(self) -> None:
        ap007 = get_pattern("AP-007")
        ap005 = get_pattern("AP-005")
        assert ap007 is not None and ap005 is not None
        assert _compute_rank_score(ap007, False, 0) > _compute_rank_score(ap005, False, 0)

    def test_result_rounded_to_4_decimal_places(self) -> None:
        p = self._ap007()
        assert p is not None
        score = _compute_rank_score(p, True, 2)
        assert score == round(score, 4)

    def test_severity_weights_ordering(self) -> None:
        weights = list(_SEVERITY_WEIGHTS.values())
        assert weights == sorted(weights, reverse=True)


# ── _bola_to_chain ─────────────────────────────────────────────────────────────


class TestBolaToChain:
    def test_returns_candidate_chain(self) -> None:
        assert isinstance(_bola_to_chain(_bola()), CandidateChain)

    def test_pattern_id_is_ap001(self) -> None:
        assert _bola_to_chain(_bola()).pattern_id == "AP-001"

    def test_with_list_endpoint_two_different_ids(self) -> None:
        c = _bola_to_chain(_bola(
            list_endpoint_id="GET:/users",
            detail_endpoint_id="GET:/users/{id}",
        ))
        assert c.endpoint_ids[0] == "GET:/users"
        assert c.endpoint_ids[1] == "GET:/users/{id}"
        assert c.endpoint_ids[0] != c.endpoint_ids[1]

    def test_without_list_endpoint_uses_detail_as_entry(self) -> None:
        c = _bola_to_chain(_bola(list_endpoint_id=None, list_is_public=None))
        assert c.endpoint_ids[0] == c.endpoint_ids[1] == "GET:/users/{id}"

    def test_crosses_auth_boundary_when_list_public_detail_protected(self) -> None:
        c = _bola_to_chain(_bola(list_is_public=True, detail_is_public=False))
        assert c.crosses_auth_boundary is True

    def test_no_boundary_when_both_public(self) -> None:
        c = _bola_to_chain(_bola(list_is_public=True, detail_is_public=True))
        assert c.crosses_auth_boundary is False

    def test_no_boundary_without_list_endpoint(self) -> None:
        c = _bola_to_chain(_bola(list_endpoint_id=None, list_is_public=None))
        assert c.crosses_auth_boundary is False

    def test_hop_count_equals_len_endpoint_ids_minus_1(self) -> None:
        c = _bola_to_chain(_bola())
        assert c.hop_count == len(c.endpoint_ids) - 1

    def test_hop_count_is_1(self) -> None:
        assert _bola_to_chain(_bola()).hop_count == 1

    def test_sensitivity_delta_non_negative(self) -> None:
        assert _bola_to_chain(_bola()).sensitivity_delta >= 0

    def test_sensitivity_delta_public_to_sensitive(self) -> None:
        c = _bola_to_chain(_bola(list_is_public=True, detail_sensitivity="SENSITIVE"))
        assert c.sensitivity_delta == 2  # PUBLIC=0 → SENSITIVE=2

    def test_sensitivity_delta_zero_without_list(self) -> None:
        c = _bola_to_chain(_bola(list_endpoint_id=None, list_is_public=None))
        assert c.sensitivity_delta == 0

    def test_entry_and_exit_summaries_non_empty(self) -> None:
        c = _bola_to_chain(_bola())
        assert c.entry_summary.strip()
        assert c.exit_summary.strip()

    def test_exit_summary_contains_sensitivity(self) -> None:
        c = _bola_to_chain(_bola(detail_sensitivity="CRITICAL"))
        assert "CRITICAL" in c.exit_summary

    def test_entry_endpoint_id_is_first(self) -> None:
        c = _bola_to_chain(_bola())
        assert c.entry_endpoint_id == c.endpoint_ids[0]

    def test_exit_endpoint_id_is_last(self) -> None:
        c = _bola_to_chain(_bola())
        assert c.exit_endpoint_id == c.endpoint_ids[-1]

    def test_rank_score_non_negative(self) -> None:
        assert _bola_to_chain(_bola()).rank_score >= 0.0

    def test_mitre_hints_is_list(self) -> None:
        assert isinstance(_bola_to_chain(_bola()).mitre_hints, list)

    def test_uuid_id_generated(self) -> None:
        c1 = _bola_to_chain(_bola())
        c2 = _bola_to_chain(_bola())
        assert c1.id != c2.id  # each call generates a new UUID


# ── _broken_auth_to_chain ──────────────────────────────────────────────────────


class TestBrokenAuthToChain:
    def test_returns_candidate_chain(self) -> None:
        assert isinstance(_broken_auth_to_chain(_broken_auth()), CandidateChain)

    def test_pattern_id_is_ap002(self) -> None:
        assert _broken_auth_to_chain(_broken_auth()).pattern_id == "AP-002"

    def test_endpoint_ids_has_two_elements(self) -> None:
        c = _broken_auth_to_chain(_broken_auth())
        assert len(c.endpoint_ids) == 2

    def test_entry_and_exit_are_same_endpoint(self) -> None:
        c = _broken_auth_to_chain(_broken_auth())
        assert c.entry_endpoint_id == c.exit_endpoint_id

    def test_hop_count_is_1(self) -> None:
        assert _broken_auth_to_chain(_broken_auth()).hop_count == 1

    def test_crosses_auth_boundary_is_false(self) -> None:
        assert _broken_auth_to_chain(_broken_auth()).crosses_auth_boundary is False

    def test_sensitivity_delta_from_sensitive(self) -> None:
        c = _broken_auth_to_chain(_broken_auth(sensitivity_class="SENSITIVE"))
        assert c.sensitivity_delta == 2

    def test_sensitivity_delta_from_critical(self) -> None:
        c = _broken_auth_to_chain(_broken_auth(sensitivity_class="CRITICAL"))
        assert c.sensitivity_delta == 3

    def test_summaries_contain_method_and_path(self) -> None:
        c = _broken_auth_to_chain(_broken_auth(
            endpoint_id="GET:/admin/report", sensitivity_class="SENSITIVE"
        ))
        assert "GET" in c.entry_summary
        assert "/admin/report" in c.entry_summary
        assert "SENSITIVE" in c.entry_summary


# ── _priv_esc_to_chain ─────────────────────────────────────────────────────────


class TestPrivEscToChain:
    def test_pattern_id_is_ap003(self) -> None:
        assert _priv_esc_to_chain(_priv_esc()).pattern_id == "AP-003"

    def test_single_endpoint_chain(self) -> None:
        c = _priv_esc_to_chain(_priv_esc())
        assert c.entry_endpoint_id == c.exit_endpoint_id

    def test_crosses_auth_boundary_false(self) -> None:
        assert _priv_esc_to_chain(_priv_esc()).crosses_auth_boundary is False

    def test_sensitivity_delta_critical(self) -> None:
        c = _priv_esc_to_chain(_priv_esc(sensitivity_class="CRITICAL"))
        assert c.sensitivity_delta == 3

    def test_hop_count_is_1(self) -> None:
        assert _priv_esc_to_chain(_priv_esc()).hop_count == 1


# ── _mass_assignment_to_chain ──────────────────────────────────────────────────


class TestMassAssignmentToChain:
    def test_pattern_id_is_ap004(self) -> None:
        assert _mass_assignment_to_chain(_mass_assignment()).pattern_id == "AP-004"

    def test_summary_contains_resource_name(self) -> None:
        c = _mass_assignment_to_chain(_mass_assignment(resource_name="Order"))
        assert "Order" in c.entry_summary

    def test_single_endpoint_chain(self) -> None:
        c = _mass_assignment_to_chain(_mass_assignment())
        assert c.entry_endpoint_id == c.exit_endpoint_id

    def test_hop_count_is_1(self) -> None:
        assert _mass_assignment_to_chain(_mass_assignment()).hop_count == 1


# ── _excessive_data_to_chain ───────────────────────────────────────────────────


class TestExcessiveDataToChain:
    def test_pattern_id_is_ap005(self) -> None:
        assert _excessive_data_to_chain(_excessive_data()).pattern_id == "AP-005"

    def test_public_tag_in_summary_when_public(self) -> None:
        c = _excessive_data_to_chain(_excessive_data(is_public=True))
        assert "(PUBLIC)" in c.entry_summary

    def test_no_public_tag_when_not_public(self) -> None:
        c = _excessive_data_to_chain(_excessive_data(is_public=False))
        assert "(PUBLIC)" not in c.entry_summary

    def test_single_endpoint_chain(self) -> None:
        c = _excessive_data_to_chain(_excessive_data())
        assert c.entry_endpoint_id == c.exit_endpoint_id


# ── _ssrf_to_chain ─────────────────────────────────────────────────────────────


class TestSsrfToChain:
    def test_pattern_id_is_ap006(self) -> None:
        assert _ssrf_to_chain(_ssrf()).pattern_id == "AP-006"

    def test_single_endpoint_chain(self) -> None:
        c = _ssrf_to_chain(_ssrf())
        assert c.entry_endpoint_id == c.exit_endpoint_id

    def test_crosses_auth_boundary_false(self) -> None:
        assert _ssrf_to_chain(_ssrf()).crosses_auth_boundary is False

    def test_hop_count_is_1(self) -> None:
        assert _ssrf_to_chain(_ssrf()).hop_count == 1

    def test_sensitivity_public(self) -> None:
        c = _ssrf_to_chain(_ssrf(sensitivity_class="PUBLIC"))
        assert c.sensitivity_delta == 0


# ── _auth_chain_to_chain ───────────────────────────────────────────────────────


class TestAuthChainToChain:
    def test_pattern_id_is_ap007(self) -> None:
        assert _auth_chain_to_chain(_auth_chain()).pattern_id == "AP-007"

    def test_two_different_endpoint_ids(self) -> None:
        c = _auth_chain_to_chain(_auth_chain())
        assert c.endpoint_ids[0] != c.endpoint_ids[1]

    def test_entry_is_auth_endpoint(self) -> None:
        c = _auth_chain_to_chain(_auth_chain(auth_endpoint_id="POST:/auth/login"))
        assert c.entry_endpoint_id == "POST:/auth/login"

    def test_exit_is_target_endpoint(self) -> None:
        c = _auth_chain_to_chain(_auth_chain(target_endpoint_id="GET:/admin/users"))
        assert c.exit_endpoint_id == "GET:/admin/users"

    def test_crosses_auth_boundary_always_true(self) -> None:
        assert _auth_chain_to_chain(_auth_chain()).crosses_auth_boundary is True

    def test_hop_count_is_1(self) -> None:
        assert _auth_chain_to_chain(_auth_chain()).hop_count == 1

    def test_sensitivity_delta_critical_target(self) -> None:
        c = _auth_chain_to_chain(_auth_chain(target_sensitivity="CRITICAL"))
        assert c.sensitivity_delta == 3  # PUBLIC(0) → CRITICAL(3)

    def test_sensitivity_delta_sensitive_target(self) -> None:
        c = _auth_chain_to_chain(_auth_chain(target_sensitivity="SENSITIVE"))
        assert c.sensitivity_delta == 2  # PUBLIC(0) → SENSITIVE(2)

    def test_entry_summary_contains_auth(self) -> None:
        c = _auth_chain_to_chain(_auth_chain())
        assert "AUTH" in c.entry_summary

    def test_exit_summary_contains_target_sensitivity(self) -> None:
        c = _auth_chain_to_chain(_auth_chain(target_sensitivity="CRITICAL"))
        assert "CRITICAL" in c.exit_summary


# ── rank_candidates ────────────────────────────────────────────────────────────


class TestRankCandidates:
    def test_empty_inputs_return_empty_list(self) -> None:
        assert rank_candidates() == []

    def test_none_inputs_return_empty_list(self) -> None:
        assert rank_candidates(bola=None, broken_auth=None) == []

    def test_returns_list_of_candidate_chains(self) -> None:
        result = rank_candidates(bola=[_bola()])
        assert isinstance(result, list)
        assert all(isinstance(c, CandidateChain) for c in result)

    def test_single_candidate_returned(self) -> None:
        assert len(rank_candidates(bola=[_bola()])) == 1

    def test_count_matches_total_inputs(self) -> None:
        result = rank_candidates(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            ssrf=[_ssrf()],
        )
        assert len(result) == 3

    def test_sorted_descending_by_rank_score(self) -> None:
        result = rank_candidates(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
            ssrf=[_ssrf()],
            priv_esc=[_priv_esc()],
        )
        scores = [c.rank_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_auth_chain_critical_outranks_excessive_data_medium(self) -> None:
        result = rank_candidates(
            excessive_data=[_excessive_data()],
            auth_chains=[_auth_chain(target_sensitivity="CRITICAL")],
        )
        pattern_ids = [c.pattern_id for c in result]
        assert pattern_ids[0] == "AP-007", "Auth chain (CRITICAL) should rank first"

    def test_all_candidate_types_accepted(self) -> None:
        result = rank_candidates(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            priv_esc=[_priv_esc()],
            mass_assignment=[_mass_assignment()],
            excessive_data=[_excessive_data()],
            ssrf=[_ssrf()],
            auth_chains=[_auth_chain()],
        )
        assert len(result) == 7
        pattern_ids = {c.pattern_id for c in result}
        assert pattern_ids == {
            "AP-001", "AP-002", "AP-003", "AP-004", "AP-005", "AP-006", "AP-007"
        }

    def test_multiple_same_pattern_type_all_included(self) -> None:
        result = rank_candidates(
            bola=[_bola(resource_name="User"), _bola(resource_name="Order")],
        )
        assert len(result) == 2

    def test_each_chain_has_unique_id(self) -> None:
        result = rank_candidates(
            bola=[_bola(), _bola()],
        )
        ids = [c.id for c in result]
        assert len(ids) == len(set(ids))

    def test_all_chains_satisfy_candidate_chain_model(self) -> None:
        result = rank_candidates(
            bola=[_bola()],
            broken_auth=[_broken_auth()],
            auth_chains=[_auth_chain()],
        )
        for chain in result:
            assert chain.hop_count >= 1
            assert len(chain.endpoint_ids) >= 2
            assert chain.sensitivity_delta >= 0
            assert chain.rank_score >= 0.0
            assert chain.hop_count == len(chain.endpoint_ids) - 1
            assert chain.entry_endpoint_id == chain.endpoint_ids[0]
            assert chain.exit_endpoint_id == chain.endpoint_ids[-1]

    def test_mitre_hints_carried_from_pattern(self) -> None:
        result = rank_candidates(auth_chains=[_auth_chain()])
        chain = result[0]
        # AP-007 has T1110, T1528, T1530
        assert "T1110" in chain.mitre_hints

    def test_owasp_category_carried_from_pattern(self) -> None:
        result = rank_candidates(broken_auth=[_broken_auth()])
        assert result[0].owasp_category == "API2:2023"

    def test_confidence_base_carried_from_pattern(self) -> None:
        ap001 = get_pattern("AP-001")
        assert ap001 is not None
        result = rank_candidates(bola=[_bola()])
        assert result[0].confidence_base == ap001.confidence_base
