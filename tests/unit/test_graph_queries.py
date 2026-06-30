"""Unit tests for api_analyzer.graph.queries.

All Neo4j interaction is mocked.  ``session.run()`` is replaced by a
MagicMock whose return value is an iterable of plain Python dicts — dict
bracket access works identically to Neo4j Record access for our purposes.

``session.run(...).single()`` is used only by the two lookup helpers;
those tests set ``session.run.return_value.single.return_value`` accordingly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from api_analyzer.graph.queries import (
    AuthChainCandidate,
    BolaCandidate,
    BrokenAuthCandidate,
    ExcessiveDataCandidate,
    MassAssignmentCandidate,
    PrivEscCandidate,
    SsrfCandidate,
    _CQL_AUTH_CHAINS,
    _CQL_BOLA,
    _CQL_BROKEN_AUTH,
    _CQL_ENDPOINT_COUNT,
    _CQL_EXCESSIVE_DATA,
    _CQL_MASS_ASSIGNMENT,
    _CQL_PRIV_ESC,
    _CQL_SPEC_COMPLETENESS,
    _CQL_SSRF,
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
from api_analyzer.graph.schema import (
    LABEL_AUTH_SCHEME,
    LABEL_ENDPOINT,
    LABEL_RESOURCE,
    PROP_HAS_ROLE_PARAM,
    PROP_IDENTIFIER_TYPE,
    PROP_INFERRED_FUNCTION,
    PROP_IS_PUBLIC,
    PROP_RETURNS_PII,
    PROP_SENSITIVITY_CLASS,
    PROP_SPEC_ID,
    PROP_ACCEPTS_URL_PARAM,
    REL_LISTS,
    REL_READS,
    REL_REQUIRES_AUTH,
    REL_WRITES,
)


# ── Mock helpers ───────────────────────────────────────────────────────────────


def _session(rows: list[dict]) -> MagicMock:
    """Return a mock Session where session.run() yields the given rows."""
    session = MagicMock()
    session.run.return_value = rows
    return session


def _session_single(value: dict | None) -> MagicMock:
    """Return a mock Session where session.run().single() returns the given dict."""
    session = MagicMock()
    result_mock = MagicMock()
    result_mock.single.return_value = value
    session.run.return_value = result_mock
    return session


# ── Dataclass integrity ────────────────────────────────────────────────────────


class TestDataclasses:
    def test_bola_candidate_frozen(self) -> None:
        c = BolaCandidate("User", "INTEGER", "/users", "GET:/users", True,
                          "GET:/users/{id}", "SENSITIVE", False)
        with pytest.raises((AttributeError, TypeError)):
            c.resource_name = "X"  # type: ignore[misc]

    def test_broken_auth_candidate_fields(self) -> None:
        c = BrokenAuthCandidate("GET:/pub", "/pub", "GET", "SENSITIVE", "DATA_READ", True)
        assert c.endpoint_id == "GET:/pub"
        assert c.returns_pii is True

    def test_priv_esc_candidate_fields(self) -> None:
        c = PrivEscCandidate("POST:/users", "/users", "POST", "CRITICAL", "DATA_WRITE", False)
        assert c.is_public is False

    def test_mass_assignment_candidate_fields(self) -> None:
        c = MassAssignmentCandidate("PUT:/items/{id}", "/items/{id}", "PUT",
                                    "SENSITIVE", True, False, "Item")
        assert c.resource_name == "Item"

    def test_excessive_data_candidate_fields(self) -> None:
        c = ExcessiveDataCandidate("GET:/users/{id}", "/users/{id}", "GET",
                                   "SENSITIVE", False, "DATA_READ")
        assert c.inferred_function == "DATA_READ"

    def test_ssrf_candidate_fields(self) -> None:
        c = SsrfCandidate("POST:/proxy", "/proxy", "POST", True, "PUBLIC")
        assert c.is_public is True

    def test_auth_chain_candidate_fields(self) -> None:
        c = AuthChainCandidate("POST:/auth/login", "/auth/login", "BearerAuth",
                               "GET:/admin", "/admin", "CRITICAL", "ADMIN")
        assert c.scheme_name == "BearerAuth"
        assert c.target_sensitivity == "CRITICAL"


# ── Cypher template quality ────────────────────────────────────────────────────


class TestCypherTemplates:
    _ALL_TEMPLATES = [
        _CQL_SPEC_COMPLETENESS, _CQL_ENDPOINT_COUNT,
        _CQL_BOLA, _CQL_BROKEN_AUTH, _CQL_PRIV_ESC,
        _CQL_MASS_ASSIGNMENT, _CQL_EXCESSIVE_DATA,
        _CQL_SSRF, _CQL_AUTH_CHAINS,
    ]

    def test_all_templates_are_strings(self) -> None:
        for t in self._ALL_TEMPLATES:
            assert isinstance(t, str) and t.strip()

    def test_all_templates_use_spec_id_parameter(self) -> None:
        for t in self._ALL_TEMPLATES:
            assert "$spec_id" in t, f"Missing $spec_id parameter: {t[:80]!r}"

    def test_bola_uses_lists_relationship(self) -> None:
        assert REL_LISTS in _CQL_BOLA

    def test_bola_uses_reads_relationship(self) -> None:
        assert REL_READS in _CQL_BOLA

    def test_bola_references_resource_label(self) -> None:
        assert LABEL_RESOURCE in _CQL_BOLA

    def test_bola_identifier_type_filter(self) -> None:
        assert PROP_IDENTIFIER_TYPE in _CQL_BOLA
        assert "INTEGER" in _CQL_BOLA
        assert "UUID" in _CQL_BOLA

    def test_broken_auth_filters_is_public(self) -> None:
        assert PROP_IS_PUBLIC in _CQL_BROKEN_AUTH

    def test_broken_auth_filters_sensitivity(self) -> None:
        assert "SENSITIVE" in _CQL_BROKEN_AUTH
        assert "CRITICAL" in _CQL_BROKEN_AUTH

    def test_priv_esc_filters_has_role_param(self) -> None:
        assert PROP_HAS_ROLE_PARAM in _CQL_PRIV_ESC

    def test_mass_assignment_uses_writes_relationship(self) -> None:
        assert REL_WRITES in _CQL_MASS_ASSIGNMENT

    def test_mass_assignment_filters_method(self) -> None:
        assert "POST" in _CQL_MASS_ASSIGNMENT
        assert "PUT" in _CQL_MASS_ASSIGNMENT
        assert "PATCH" in _CQL_MASS_ASSIGNMENT

    def test_excessive_data_filters_returns_pii(self) -> None:
        assert PROP_RETURNS_PII in _CQL_EXCESSIVE_DATA

    def test_ssrf_filters_accepts_url_param(self) -> None:
        assert PROP_ACCEPTS_URL_PARAM in _CQL_SSRF

    def test_auth_chains_uses_requires_auth_relationship(self) -> None:
        assert REL_REQUIRES_AUTH in _CQL_AUTH_CHAINS

    def test_auth_chains_filters_auth_function(self) -> None:
        assert PROP_INFERRED_FUNCTION in _CQL_AUTH_CHAINS
        assert "AUTH" in _CQL_AUTH_CHAINS

    def test_auth_chains_has_limit(self) -> None:
        assert "LIMIT" in _CQL_AUTH_CHAINS.upper()

    def test_auth_chains_references_auth_scheme_label(self) -> None:
        assert LABEL_AUTH_SCHEME in _CQL_AUTH_CHAINS

    def test_no_template_contains_raw_string_injection(self) -> None:
        """Ensure no template uses string format() or % interpolation on user values."""
        for t in self._ALL_TEMPLATES:
            # No {user_input} style holes — only $param style
            import re
            holes = re.findall(r"\{[^}]+\}", t)
            # All remaining {} pairs should be Cypher map syntax, not format holes
            for hole in holes:
                assert hole.startswith("{") and ":" in hole or hole in ("{}", ""), \
                    f"Possible injection hole {hole!r} in Cypher: {t[:60]!r}"


# ── get_spec_completeness ──────────────────────────────────────────────────────


class TestGetSpecCompleteness:
    def test_returns_float_from_record(self) -> None:
        session = _session_single({"completeness": 0.75})
        result = get_spec_completeness(session, "test:1.0")
        assert result == pytest.approx(0.75)

    def test_returns_zero_when_no_record(self) -> None:
        session = _session_single(None)
        result = get_spec_completeness(session, "test:1.0")
        assert result == 0.0

    def test_returns_zero_when_completeness_is_none(self) -> None:
        session = _session_single({"completeness": None})
        result = get_spec_completeness(session, "test:1.0")
        assert result == 0.0

    def test_spec_id_passed_as_parameter(self) -> None:
        session = _session_single({"completeness": 0.5})
        get_spec_completeness(session, "my-api:2.0")
        session.run.assert_called_once()
        kwargs = session.run.call_args.kwargs
        assert kwargs.get("spec_id") == "my-api:2.0"

    def test_result_is_float_type(self) -> None:
        session = _session_single({"completeness": 1})  # integer from Neo4j
        result = get_spec_completeness(session, "x:1")
        assert isinstance(result, float)


# ── get_endpoint_count ─────────────────────────────────────────────────────────


class TestGetEndpointCount:
    def test_returns_count_from_record(self) -> None:
        session = _session_single({"endpoint_count": 42})
        assert get_endpoint_count(session, "x:1") == 42

    def test_returns_zero_when_no_record(self) -> None:
        session = _session_single(None)
        assert get_endpoint_count(session, "x:1") == 0

    def test_result_is_int_type(self) -> None:
        session = _session_single({"endpoint_count": 7})
        assert isinstance(get_endpoint_count(session, "x:1"), int)

    def test_spec_id_passed_as_parameter(self) -> None:
        session = _session_single({"endpoint_count": 0})
        get_endpoint_count(session, "spec:1.0")
        kwargs = session.run.call_args.kwargs
        assert kwargs.get("spec_id") == "spec:1.0"


# ── find_bola_candidates ───────────────────────────────────────────────────────


class TestFindBolaCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "resource_name": "User",
            "identifier_type": "INTEGER",
            "path_prefix": "/users",
            "list_endpoint_id": "GET:/users",
            "list_is_public": True,
            "detail_endpoint_id": "GET:/users/{userId}",
            "detail_sensitivity": "SENSITIVE",
            "detail_is_public": False,
            **kwargs,
        }

    def test_empty_result_returns_empty_list(self) -> None:
        assert find_bola_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed_correctly(self) -> None:
        session = _session([self._row()])
        results = find_bola_candidates(session, "x:1")
        assert len(results) == 1
        c = results[0]
        assert isinstance(c, BolaCandidate)
        assert c.resource_name == "User"
        assert c.identifier_type == "INTEGER"
        assert c.list_endpoint_id == "GET:/users"
        assert c.list_is_public is True
        assert c.detail_endpoint_id == "GET:/users/{userId}"
        assert c.detail_sensitivity == "SENSITIVE"
        assert c.detail_is_public is False

    def test_list_endpoint_id_can_be_none(self) -> None:
        session = _session([self._row(list_endpoint_id=None, list_is_public=None)])
        results = find_bola_candidates(session, "x:1")
        assert results[0].list_endpoint_id is None
        assert results[0].list_is_public is None

    def test_multiple_candidates_returned(self) -> None:
        rows = [
            self._row(resource_name="User", path_prefix="/users"),
            self._row(resource_name="Order", path_prefix="/orders",
                      identifier_type="UUID"),
        ]
        results = find_bola_candidates(_session(rows), "x:1")
        assert len(results) == 2

    def test_spec_id_passed_to_session(self) -> None:
        session = _session([])
        find_bola_candidates(session, "my-api:1.0")
        kwargs = session.run.call_args.kwargs
        assert kwargs.get("spec_id") == "my-api:1.0"

    def test_uses_bola_cypher_template(self) -> None:
        session = _session([])
        find_bola_candidates(session, "x:1")
        stmt = session.run.call_args.args[0]
        assert stmt == _CQL_BOLA


# ── find_broken_auth_candidates ───────────────────────────────────────────────


class TestFindBrokenAuthCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "endpoint_id": "GET:/sensitive",
            "path": "/sensitive",
            "method": "GET",
            "sensitivity_class": "SENSITIVE",
            "inferred_function": "DATA_READ",
            "returns_pii": True,
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_broken_auth_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_broken_auth_candidates(_session([self._row()]), "x:1")
        assert len(results) == 1
        c = results[0]
        assert isinstance(c, BrokenAuthCandidate)
        assert c.endpoint_id == "GET:/sensitive"
        assert c.sensitivity_class == "SENSITIVE"
        assert c.returns_pii is True

    def test_returns_pii_coerced_to_bool(self) -> None:
        c = find_broken_auth_candidates(_session([self._row(returns_pii=1)]), "x:1")[0]
        assert isinstance(c.returns_pii, bool)
        assert c.returns_pii is True

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_broken_auth_candidates(session, "api:2.0")
        assert session.run.call_args.kwargs.get("spec_id") == "api:2.0"


# ── find_priv_esc_candidates ───────────────────────────────────────────────────


class TestFindPrivEscCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "endpoint_id": "POST:/users",
            "path": "/users",
            "method": "POST",
            "sensitivity_class": "CRITICAL",
            "inferred_function": "DATA_WRITE",
            "is_public": False,
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_priv_esc_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_priv_esc_candidates(_session([self._row()]), "x:1")
        c = results[0]
        assert isinstance(c, PrivEscCandidate)
        assert c.endpoint_id == "POST:/users"
        assert c.is_public is False

    def test_is_public_coerced_to_bool(self) -> None:
        c = find_priv_esc_candidates(_session([self._row(is_public=0)]), "x:1")[0]
        assert isinstance(c.is_public, bool)
        assert c.is_public is False

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_priv_esc_candidates(session, "target:1.0")
        assert session.run.call_args.kwargs.get("spec_id") == "target:1.0"


# ── find_mass_assignment_candidates ───────────────────────────────────────────


class TestFindMassAssignmentCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "endpoint_id": "PUT:/users/{id}",
            "path": "/users/{id}",
            "method": "PUT",
            "sensitivity_class": "SENSITIVE",
            "accepts_pii": True,
            "has_role_param": False,
            "resource_name": "User",
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_mass_assignment_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_mass_assignment_candidates(_session([self._row()]), "x:1")
        c = results[0]
        assert isinstance(c, MassAssignmentCandidate)
        assert c.resource_name == "User"
        assert c.method == "PUT"
        assert c.accepts_pii is True
        assert c.has_role_param is False

    def test_has_role_param_coerced_to_bool(self) -> None:
        c = find_mass_assignment_candidates(
            _session([self._row(has_role_param=1)]), "x:1"
        )[0]
        assert isinstance(c.has_role_param, bool)
        assert c.has_role_param is True

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_mass_assignment_candidates(session, "shop:3.0")
        assert session.run.call_args.kwargs.get("spec_id") == "shop:3.0"


# ── find_excessive_data_candidates ────────────────────────────────────────────


class TestFindExcessiveDataCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "endpoint_id": "GET:/users/{id}",
            "path": "/users/{id}",
            "method": "GET",
            "sensitivity_class": "SENSITIVE",
            "is_public": False,
            "inferred_function": "DATA_READ",
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_excessive_data_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_excessive_data_candidates(_session([self._row()]), "x:1")
        c = results[0]
        assert isinstance(c, ExcessiveDataCandidate)
        assert c.endpoint_id == "GET:/users/{id}"
        assert c.inferred_function == "DATA_READ"

    def test_is_public_coerced_to_bool(self) -> None:
        c = find_excessive_data_candidates(
            _session([self._row(is_public=True)]), "x:1"
        )[0]
        assert isinstance(c.is_public, bool)

    def test_public_pii_endpoint_included(self) -> None:
        c = find_excessive_data_candidates(
            _session([self._row(is_public=True, sensitivity_class="PUBLIC")]), "x:1"
        )[0]
        assert c.is_public is True
        assert c.sensitivity_class == "PUBLIC"

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_excessive_data_candidates(session, "api:v1")
        assert session.run.call_args.kwargs.get("spec_id") == "api:v1"


# ── find_ssrf_candidates ───────────────────────────────────────────────────────


class TestFindSsrfCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "endpoint_id": "POST:/proxy",
            "path": "/proxy",
            "method": "POST",
            "is_public": True,
            "sensitivity_class": "PUBLIC",
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_ssrf_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_ssrf_candidates(_session([self._row()]), "x:1")
        c = results[0]
        assert isinstance(c, SsrfCandidate)
        assert c.endpoint_id == "POST:/proxy"
        assert c.is_public is True

    def test_is_public_coerced_to_bool(self) -> None:
        c = find_ssrf_candidates(_session([self._row(is_public=1)]), "x:1")[0]
        assert isinstance(c.is_public, bool)

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_ssrf_candidates(session, "svc:0.1")
        assert session.run.call_args.kwargs.get("spec_id") == "svc:0.1"

    def test_uses_ssrf_cypher_template(self) -> None:
        session = _session([])
        find_ssrf_candidates(session, "x:1")
        assert session.run.call_args.args[0] == _CQL_SSRF


# ── find_auth_chain_candidates ─────────────────────────────────────────────────


class TestFindAuthChainCandidates:
    def _row(self, **kwargs: object) -> dict:
        return {
            "auth_endpoint_id": "POST:/auth/login",
            "auth_path": "/auth/login",
            "scheme_name": "BearerAuth",
            "target_endpoint_id": "GET:/admin/users",
            "target_path": "/admin/users",
            "target_sensitivity": "CRITICAL",
            "target_function": "ADMIN",
            **kwargs,
        }

    def test_empty_returns_empty(self) -> None:
        assert find_auth_chain_candidates(_session([]), "x:1") == []

    def test_single_candidate_parsed(self) -> None:
        results = find_auth_chain_candidates(_session([self._row()]), "x:1")
        c = results[0]
        assert isinstance(c, AuthChainCandidate)
        assert c.auth_endpoint_id == "POST:/auth/login"
        assert c.scheme_name == "BearerAuth"
        assert c.target_sensitivity == "CRITICAL"
        assert c.target_function == "ADMIN"

    def test_multiple_targets_for_same_auth_ep(self) -> None:
        rows = [
            self._row(target_endpoint_id="GET:/admin/users", target_path="/admin/users"),
            self._row(target_endpoint_id="DELETE:/admin/users/{id}",
                      target_path="/admin/users/{id}"),
        ]
        results = find_auth_chain_candidates(_session(rows), "x:1")
        assert len(results) == 2

    def test_different_schemes_preserved(self) -> None:
        rows = [
            self._row(scheme_name="BearerAuth"),
            self._row(scheme_name="ApiKey", target_endpoint_id="GET:/data"),
        ]
        schemes = {c.scheme_name for c in find_auth_chain_candidates(_session(rows), "x:1")}
        assert "BearerAuth" in schemes
        assert "ApiKey" in schemes

    def test_spec_id_forwarded(self) -> None:
        session = _session([])
        find_auth_chain_candidates(session, "backend:5.0")
        assert session.run.call_args.kwargs.get("spec_id") == "backend:5.0"

    def test_uses_auth_chains_cypher_template(self) -> None:
        session = _session([])
        find_auth_chain_candidates(session, "x:1")
        assert session.run.call_args.args[0] == _CQL_AUTH_CHAINS
