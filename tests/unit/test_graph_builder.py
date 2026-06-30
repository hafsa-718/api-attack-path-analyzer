"""Unit tests for api_analyzer.graph.builder.

All Neo4j interaction is mocked — no running instance required.
The mock driver is wired so that session.execute_write(fn) calls fn(mock_tx),
letting us inspect every tx.run() call made during build_graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from api_analyzer.graph.builder import (
    BuildResult,
    _auth_scheme_props,
    _build_auth_scheme_nodes,
    _build_child_of_rels,
    _build_endpoint_nodes,
    _build_part_of_rels,
    _build_requires_auth_rels,
    _build_resource_endpoint_rels,
    _build_resource_nodes,
    _build_spec_node,
    _endpoint_props,
    _resource_props,
    _spec_props,
    build_graph,
)
from api_analyzer.graph.schema import (
    LABEL_API_SPEC,
    LABEL_AUTH_SCHEME,
    LABEL_ENDPOINT,
    LABEL_RESOURCE,
    PROP_AUTH_TYPE,
    PROP_ID,
    PROP_IDENTIFIER_TYPE,
    PROP_INFERRED_FUNCTION,
    PROP_IS_JWT,
    PROP_IS_PUBLIC,
    PROP_METHOD,
    PROP_NAME,
    PROP_PARENT_RESOURCE_NAME,
    PROP_PATH,
    PROP_PATH_PARAM_TYPE,
    PROP_PATH_PREFIX,
    PROP_SENSITIVITY_CLASS,
    PROP_SPEC_COMPLETENESS,
    PROP_SPEC_FORMAT,
    PROP_SPEC_ID,
    PROP_TITLE,
    PROP_VERSION,
    REL_CHILD_OF,
    REL_DELETES,
    REL_LISTS,
    REL_PART_OF,
    REL_READS,
    REL_REQUIRES_AUTH,
    REL_WRITES,
    make_spec_id,
)
from api_analyzer.models.enums import (
    AuthType,
    EndpointFunction,
    HttpMethod,
    PathParamType,
    SensitivityClass,
    SpecFormat,
)
from api_analyzer.models.spec import (
    AuthScheme,
    InferredResource,
    ParsedEndpoint,
    ParsedSpec,
)
from api_analyzer.parser.ingestor import parse_spec_dict


# ── Test fixtures ──────────────────────────────────────────────────────────────


def _make_mock_driver() -> tuple[MagicMock, MagicMock]:
    """Return (driver, tx) where execute_write calls fn(tx) automatically."""
    tx = MagicMock()
    session = MagicMock()
    session.execute_write.side_effect = lambda fn: fn(tx)
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver, tx


def _ep(
    *,
    path: str = "/users",
    method: HttpMethod = HttpMethod.GET,
    is_public: bool = True,
    auth_scheme_names: list[str] | None = None,
    summary: str | None = None,
    inferred_function: EndpointFunction = EndpointFunction.DATA_READ,
    sensitivity_class: SensitivityClass = SensitivityClass.PUBLIC,
    path_param_type: PathParamType = PathParamType.NONE,
    returns_pii: bool = False,
    accepts_pii: bool = False,
    has_role_param: bool = False,
    accepts_url_param: bool = False,
) -> ParsedEndpoint:
    return ParsedEndpoint(
        id=f"{method}:{path}",
        path=path,
        method=method,
        summary=summary,
        is_public=is_public,
        auth_scheme_names=auth_scheme_names or [],
        inferred_function=inferred_function,
        sensitivity_class=sensitivity_class,
        path_param_type=path_param_type,
        returns_pii=returns_pii,
        accepts_pii=accepts_pii,
        has_role_param=has_role_param,
        accepts_url_param=accepts_url_param,
    )


def _spec(
    endpoints: list[ParsedEndpoint] | None = None,
    resources: list[InferredResource] | None = None,
    auth_schemes: dict[str, AuthScheme] | None = None,
    completeness: float = 0.5,
) -> ParsedSpec:
    return ParsedSpec(
        title="Test API",
        version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        endpoints=endpoints or [],
        resources=resources or [],
        auth_schemes=auth_schemes or {},
        spec_completeness=completeness,
    )


def _auth(name: str = "BearerAuth", is_jwt: bool = True) -> AuthScheme:
    return AuthScheme(name=name, auth_type=AuthType.HTTP_BEARER, is_jwt=is_jwt)


def _resource(
    name: str = "User",
    path_prefix: str = "/users",
    identifier_type: PathParamType = PathParamType.INTEGER,
    collection_ep_id: str | None = "GET:/users",
    detail_ep_id: str | None = "GET:/users/{userId}",
    write_ep_ids: list[str] | None = None,
    delete_ep_id: str | None = None,
    parent_name: str | None = None,
    identifier_name: str | None = "userId",
) -> InferredResource:
    return InferredResource(
        name=name,
        path_prefix=path_prefix,
        identifier_type=identifier_type,
        collection_endpoint_id=collection_ep_id,
        detail_endpoint_id=detail_ep_id,
        write_endpoint_ids=write_ep_ids or [],
        delete_endpoint_id=delete_ep_id,
        parent_resource_name=parent_name,
        identifier_name=identifier_name,
    )


# ── BuildResult ────────────────────────────────────────────────────────────────


class TestBuildResult:
    def test_is_frozen(self) -> None:
        result = BuildResult(spec_id="x", endpoint_count=1,
                             resource_count=0, auth_scheme_count=0, rel_count=1)
        with pytest.raises((AttributeError, TypeError)):
            result.endpoint_count = 99  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        result = BuildResult(spec_id="api:1.0", endpoint_count=5,
                             resource_count=2, auth_scheme_count=1, rel_count=8)
        assert result.spec_id == "api:1.0"
        assert result.endpoint_count == 5
        assert result.resource_count == 2
        assert result.auth_scheme_count == 1
        assert result.rel_count == 8


# ── Property extraction ────────────────────────────────────────────────────────


class TestSpecProps:
    def test_required_fields_present(self) -> None:
        spec = _spec()
        props = _spec_props(spec, "test-api:1.0.0")
        assert props[PROP_ID] == "test-api:1.0.0"
        assert props[PROP_SPEC_ID] == "test-api:1.0.0"
        assert props[PROP_TITLE] == "Test API"
        assert props[PROP_VERSION] == "1.0.0"
        assert props[PROP_SPEC_FORMAT] == str(SpecFormat.OPENAPI3)
        assert props[PROP_SPEC_COMPLETENESS] == 0.5

    def test_spec_id_and_id_are_equal(self) -> None:
        props = _spec_props(_spec(), "my-api:2.0")
        assert props[PROP_ID] == props[PROP_SPEC_ID]

    def test_optional_description_omitted_when_none(self) -> None:
        props = _spec_props(_spec(), "x:1")
        assert "description" not in props

    def test_spec_format_is_plain_string(self) -> None:
        props = _spec_props(_spec(), "x:1")
        assert isinstance(props[PROP_SPEC_FORMAT], str)
        assert props[PROP_SPEC_FORMAT] == "openapi3"


class TestEndpointProps:
    def test_required_fields_present(self) -> None:
        ep = _ep(path="/users", method=HttpMethod.GET, is_public=True)
        props = _endpoint_props(ep, "x:1")
        assert props[PROP_ID] == "GET:/users"
        assert props[PROP_SPEC_ID] == "x:1"
        assert props[PROP_PATH] == "/users"
        assert props[PROP_METHOD] == "GET"
        assert props[PROP_IS_PUBLIC] is True

    def test_enums_serialised_as_strings(self) -> None:
        ep = _ep(
            sensitivity_class=SensitivityClass.CRITICAL,
            inferred_function=EndpointFunction.ADMIN,
            path_param_type=PathParamType.INTEGER,
        )
        props = _endpoint_props(ep, "x:1")
        assert props[PROP_SENSITIVITY_CLASS] == "CRITICAL"
        assert props[PROP_INFERRED_FUNCTION] == "ADMIN"
        assert props[PROP_PATH_PARAM_TYPE] == "INTEGER"

    def test_method_is_plain_string(self) -> None:
        ep = _ep(method=HttpMethod.POST)
        props = _endpoint_props(ep, "x:1")
        assert props[PROP_METHOD] == "POST"
        assert isinstance(props[PROP_METHOD], str)

    def test_summary_omitted_when_none(self) -> None:
        ep = _ep(summary=None)
        props = _endpoint_props(ep, "x:1")
        assert "summary" not in props

    def test_summary_included_when_set(self) -> None:
        ep = _ep(summary="Get all users")
        props = _endpoint_props(ep, "x:1")
        assert props["summary"] == "Get all users"

    def test_auth_scheme_names_is_list(self) -> None:
        ep = _ep(auth_scheme_names=["Bearer", "ApiKey"])
        props = _endpoint_props(ep, "x:1")
        assert props["auth_scheme_names"] == ["Bearer", "ApiKey"]
        assert isinstance(props["auth_scheme_names"], list)

    def test_boolean_signals_present(self) -> None:
        ep = _ep(returns_pii=True, accepts_pii=True,
                 has_role_param=True, accepts_url_param=True)
        props = _endpoint_props(ep, "x:1")
        assert props["returns_pii"] is True
        assert props["accepts_pii"] is True
        assert props["has_role_param"] is True
        assert props["accepts_url_param"] is True


class TestResourceProps:
    def test_required_fields_present(self) -> None:
        r = _resource()
        props = _resource_props(r, "x:1")
        assert props[PROP_NAME] == "User"
        assert props[PROP_PATH_PREFIX] == "/users"
        assert props[PROP_SPEC_ID] == "x:1"
        assert props[PROP_IDENTIFIER_TYPE] == "INTEGER"

    def test_identifier_type_is_plain_string(self) -> None:
        r = _resource(identifier_type=PathParamType.UUID)
        props = _resource_props(r, "x:1")
        assert props[PROP_IDENTIFIER_TYPE] == "UUID"
        assert isinstance(props[PROP_IDENTIFIER_TYPE], str)

    def test_parent_resource_name_omitted_when_none(self) -> None:
        r = _resource(parent_name=None)
        props = _resource_props(r, "x:1")
        assert PROP_PARENT_RESOURCE_NAME not in props

    def test_parent_resource_name_included_when_set(self) -> None:
        r = _resource(parent_name="User")
        props = _resource_props(r, "x:1")
        assert props[PROP_PARENT_RESOURCE_NAME] == "User"

    def test_identifier_name_included_when_set(self) -> None:
        r = _resource(identifier_name="userId")
        props = _resource_props(r, "x:1")
        assert props["identifier_name"] == "userId"


class TestAuthSchemeProps:
    def test_required_fields_present(self) -> None:
        scheme = _auth()
        props = _auth_scheme_props(scheme, "x:1")
        assert props[PROP_NAME] == "BearerAuth"
        assert props[PROP_SPEC_ID] == "x:1"
        assert props[PROP_AUTH_TYPE] == "http_bearer"
        assert props[PROP_IS_JWT] is True

    def test_auth_type_is_plain_string(self) -> None:
        scheme = AuthScheme(name="ApiKey", auth_type=AuthType.API_KEY, is_jwt=False)
        props = _auth_scheme_props(scheme, "x:1")
        assert props[PROP_AUTH_TYPE] == "apiKey"
        assert isinstance(props[PROP_AUTH_TYPE], str)

    def test_is_jwt_false(self) -> None:
        scheme = _auth(is_jwt=False)
        props = _auth_scheme_props(scheme, "x:1")
        assert props[PROP_IS_JWT] is False


# ── Node builders ──────────────────────────────────────────────────────────────


class TestBuildSpecNode:
    def test_calls_tx_run_once(self) -> None:
        tx = MagicMock()
        _build_spec_node(tx, _spec(), "test:1.0")
        tx.run.assert_called_once()

    def test_query_contains_api_spec_label(self) -> None:
        tx = MagicMock()
        _build_spec_node(tx, _spec(), "test:1.0")
        stmt = tx.run.call_args.args[0]
        assert LABEL_API_SPEC in stmt

    def test_query_contains_merge(self) -> None:
        tx = MagicMock()
        _build_spec_node(tx, _spec(), "test:1.0")
        stmt = tx.run.call_args.args[0].upper()
        assert "MERGE" in stmt

    def test_spec_id_passed_as_id_param(self) -> None:
        tx = MagicMock()
        _build_spec_node(tx, _spec(), "my-api:1.0")
        kwargs = tx.run.call_args.kwargs
        assert kwargs.get("id") == "my-api:1.0"

    def test_props_dict_passed(self) -> None:
        tx = MagicMock()
        _build_spec_node(tx, _spec(), "x:1")
        kwargs = tx.run.call_args.kwargs
        assert "props" in kwargs
        assert isinstance(kwargs["props"], dict)


class TestBuildAuthSchemeNodes:
    def test_empty_auth_schemes_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        result = _build_auth_scheme_nodes(tx, _spec(), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_single_scheme_returns_one(self) -> None:
        tx = MagicMock()
        spec = _spec(auth_schemes={"BearerAuth": _auth()})
        result = _build_auth_scheme_nodes(tx, spec, "x:1")
        assert result == 1

    def test_calls_tx_run_once_regardless_of_scheme_count(self) -> None:
        tx = MagicMock()
        schemes = {"A": _auth("A"), "B": _auth("B"), "C": _auth("C")}
        spec = _spec(auth_schemes=schemes)
        _build_auth_scheme_nodes(tx, spec, "x:1")
        tx.run.assert_called_once()

    def test_query_contains_auth_scheme_label(self) -> None:
        tx = MagicMock()
        spec = _spec(auth_schemes={"B": _auth()})
        _build_auth_scheme_nodes(tx, spec, "x:1")
        stmt = tx.run.call_args.args[0]
        assert LABEL_AUTH_SCHEME in stmt

    def test_schemes_list_passed_as_parameter(self) -> None:
        tx = MagicMock()
        spec = _spec(auth_schemes={"Bearer": _auth("Bearer")})
        _build_auth_scheme_nodes(tx, spec, "x:1")
        kwargs = tx.run.call_args.kwargs
        assert "schemes" in kwargs
        assert isinstance(kwargs["schemes"], list)
        assert len(kwargs["schemes"]) == 1


class TestBuildEndpointNodes:
    def test_empty_endpoints_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        result = _build_endpoint_nodes(tx, _spec(endpoints=[]), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_returns_endpoint_count(self) -> None:
        tx = MagicMock()
        eps = [_ep(path=f"/ep{i}") for i in range(5)]
        result = _build_endpoint_nodes(tx, _spec(endpoints=eps), "x:1")
        assert result == 5

    def test_calls_tx_run_once(self) -> None:
        tx = MagicMock()
        spec = _spec(endpoints=[_ep(), _ep(path="/items")])
        _build_endpoint_nodes(tx, spec, "x:1")
        tx.run.assert_called_once()

    def test_query_contains_endpoint_label(self) -> None:
        tx = MagicMock()
        _build_endpoint_nodes(tx, _spec(endpoints=[_ep()]), "x:1")
        stmt = tx.run.call_args.args[0]
        assert LABEL_ENDPOINT in stmt

    def test_endpoints_list_passed_as_parameter(self) -> None:
        tx = MagicMock()
        _build_endpoint_nodes(tx, _spec(endpoints=[_ep()]), "x:1")
        kwargs = tx.run.call_args.kwargs
        assert "endpoints" in kwargs
        assert isinstance(kwargs["endpoints"], list)


class TestBuildResourceNodes:
    def test_empty_resources_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        result = _build_resource_nodes(tx, _spec(resources=[]), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_returns_resource_count(self) -> None:
        tx = MagicMock()
        resources = [_resource("User", "/users"), _resource("Item", "/items")]
        result = _build_resource_nodes(tx, _spec(resources=resources), "x:1")
        assert result == 2

    def test_query_contains_resource_label(self) -> None:
        tx = MagicMock()
        _build_resource_nodes(tx, _spec(resources=[_resource()]), "x:1")
        stmt = tx.run.call_args.args[0]
        assert LABEL_RESOURCE in stmt


# ── Relationship builders ──────────────────────────────────────────────────────


class TestBuildPartOfRels:
    def test_no_endpoints_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        result = _build_part_of_rels(tx, _spec(endpoints=[]), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_returns_endpoint_count(self) -> None:
        tx = MagicMock()
        spec = _spec(endpoints=[_ep(), _ep(path="/items")])
        result = _build_part_of_rels(tx, spec, "x:1")
        assert result == 2

    def test_query_contains_part_of_rel_type(self) -> None:
        tx = MagicMock()
        _build_part_of_rels(tx, _spec(endpoints=[_ep()]), "x:1")
        stmt = tx.run.call_args.args[0]
        assert REL_PART_OF in stmt

    def test_query_contains_api_spec_label(self) -> None:
        tx = MagicMock()
        _build_part_of_rels(tx, _spec(endpoints=[_ep()]), "x:1")
        stmt = tx.run.call_args.args[0]
        assert LABEL_API_SPEC in stmt

    def test_spec_id_passed_as_kwarg(self) -> None:
        tx = MagicMock()
        _build_part_of_rels(tx, _spec(endpoints=[_ep()]), "my-api:1.0")
        kwargs = tx.run.call_args.kwargs
        assert kwargs.get("spec_id") == "my-api:1.0"


class TestBuildRequiresAuthRels:
    def test_no_auth_requirements_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        spec = _spec(endpoints=[_ep(auth_scheme_names=[])])
        result = _build_requires_auth_rels(tx, spec, "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_counts_each_scheme_per_endpoint(self) -> None:
        tx = MagicMock()
        ep1 = _ep(auth_scheme_names=["Bearer"])
        ep2 = _ep(path="/items", auth_scheme_names=["Bearer", "ApiKey"])
        spec = _spec(endpoints=[ep1, ep2])
        result = _build_requires_auth_rels(tx, spec, "x:1")
        assert result == 3  # 1 + 2

    def test_query_contains_requires_auth_rel_type(self) -> None:
        tx = MagicMock()
        spec = _spec(endpoints=[_ep(auth_scheme_names=["Bearer"])])
        _build_requires_auth_rels(tx, spec, "x:1")
        stmt = tx.run.call_args.args[0]
        assert REL_REQUIRES_AUTH in stmt

    def test_rels_parameter_contains_correct_data(self) -> None:
        tx = MagicMock()
        ep = _ep(path="/items", auth_scheme_names=["Bearer"])
        _build_requires_auth_rels(tx, _spec(endpoints=[ep]), "x:1")
        rels = tx.run.call_args.kwargs["rels"]
        assert len(rels) == 1
        assert rels[0]["endpoint_id"] == "GET:/items"
        assert rels[0]["scheme_name"] == "Bearer"


class TestBuildResourceEndpointRels:
    def test_no_resources_returns_zero(self) -> None:
        tx = MagicMock()
        result = _build_resource_endpoint_rels(tx, _spec(resources=[]), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_lists_rel_created_when_collection_ep_exists(self) -> None:
        tx = MagicMock()
        r = _resource(collection_ep_id="GET:/users", detail_ep_id=None)
        _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        stmts = [c.args[0] for c in tx.run.call_args_list]
        assert any(REL_LISTS in s for s in stmts)

    def test_reads_rel_created_when_detail_ep_exists(self) -> None:
        tx = MagicMock()
        r = _resource(collection_ep_id=None, detail_ep_id="GET:/users/{id}")
        _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        stmts = [c.args[0] for c in tx.run.call_args_list]
        assert any(REL_READS in s for s in stmts)

    def test_writes_rels_created_for_write_endpoints(self) -> None:
        tx = MagicMock()
        r = _resource(collection_ep_id=None, detail_ep_id=None,
                      write_ep_ids=["POST:/users", "PUT:/users/{id}"])
        _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        stmts = [c.args[0] for c in tx.run.call_args_list]
        assert any(REL_WRITES in s for s in stmts)

    def test_deletes_rel_created_when_delete_ep_exists(self) -> None:
        tx = MagicMock()
        r = _resource(collection_ep_id=None, detail_ep_id=None,
                      delete_ep_id="DELETE:/users/{id}")
        _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        stmts = [c.args[0] for c in tx.run.call_args_list]
        assert any(REL_DELETES in s for s in stmts)

    def test_counts_all_rels(self) -> None:
        tx = MagicMock()
        r = _resource(
            collection_ep_id="GET:/users",
            detail_ep_id="GET:/users/{id}",
            write_ep_ids=["POST:/users", "PUT:/users/{id}"],
            delete_ep_id="DELETE:/users/{id}",
        )
        result = _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        # 1 LISTS + 1 READS + 2 WRITES + 1 DELETES = 5
        assert result == 5

    def test_missing_endpoints_not_included(self) -> None:
        tx = MagicMock()
        # Only detail endpoint set — only READS should be created.
        r = _resource(collection_ep_id=None, detail_ep_id="GET:/users/{id}",
                      write_ep_ids=[], delete_ep_id=None)
        _build_resource_endpoint_rels(tx, _spec(resources=[r]), "x:1")
        assert tx.run.call_count == 1  # only one READS query
        stmt = tx.run.call_args.args[0]
        assert REL_READS in stmt


class TestBuildChildOfRels:
    def test_no_parents_returns_zero_without_query(self) -> None:
        tx = MagicMock()
        r = _resource(parent_name=None)
        result = _build_child_of_rels(tx, _spec(resources=[r]), "x:1")
        assert result == 0
        tx.run.assert_not_called()

    def test_returns_count_of_resources_with_parent(self) -> None:
        tx = MagicMock()
        r1 = _resource("Order", "/users/{id}/orders", parent_name="User")
        r2 = _resource("User", "/users", parent_name=None)
        result = _build_child_of_rels(tx, _spec(resources=[r1, r2]), "x:1")
        assert result == 1

    def test_query_contains_child_of_rel_type(self) -> None:
        tx = MagicMock()
        r = _resource("Order", "/users/{id}/orders", parent_name="User")
        _build_child_of_rels(tx, _spec(resources=[r]), "x:1")
        stmt = tx.run.call_args.args[0]
        assert REL_CHILD_OF in stmt

    def test_rels_param_has_correct_data(self) -> None:
        tx = MagicMock()
        r = _resource("Order", "/users/{id}/orders", parent_name="User")
        _build_child_of_rels(tx, _spec(resources=[r]), "x:1")
        rels = tx.run.call_args.kwargs["rels"]
        assert len(rels) == 1
        assert rels[0]["child_prefix"] == "/users/{id}/orders"
        assert rels[0]["parent_name"] == "User"


# ── build_graph integration ────────────────────────────────────────────────────


class TestBuildGraph:
    def _make_full_spec(self) -> ParsedSpec:
        ep_list = _ep(path="/users", is_public=True)
        ep_detail = _ep(
            path="/users/{userId}", is_public=False,
            auth_scheme_names=["BearerAuth"],
            sensitivity_class=SensitivityClass.SENSITIVE,
            path_param_type=PathParamType.INTEGER,
        )
        resource = _resource(
            name="User", path_prefix="/users",
            collection_ep_id="GET:/users",
            detail_ep_id="GET:/users/{userId}",
            write_ep_ids=["POST:/users"],
        )
        return _spec(
            endpoints=[ep_list, ep_detail],
            resources=[resource],
            auth_schemes={"BearerAuth": _auth()},
        )

    def test_returns_build_result(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert isinstance(result, BuildResult)

    def test_spec_id_derived_from_title_and_version(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        expected = make_spec_id("Test API", "1.0.0")
        assert result.spec_id == expected

    def test_endpoint_count_matches_spec(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert result.endpoint_count == 2

    def test_resource_count_matches_spec(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert result.resource_count == 1

    def test_auth_scheme_count_matches_spec(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert result.auth_scheme_count == 1

    def test_rel_count_is_positive(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert result.rel_count > 0

    def test_driver_session_opened(self) -> None:
        driver, _ = _make_mock_driver()
        build_graph(self._make_full_spec(), driver)
        driver.session.assert_called_once()

    def test_execute_write_called(self) -> None:
        driver, _ = _make_mock_driver()
        session = driver.session.return_value.__enter__.return_value
        build_graph(self._make_full_spec(), driver)
        session.execute_write.assert_called_once()

    def test_tx_run_called_multiple_times(self) -> None:
        """build_graph issues multiple tx.run() calls — one per node/rel batch."""
        driver, tx = _make_mock_driver()
        build_graph(self._make_full_spec(), driver)
        # Minimum: spec + auth_schemes + endpoints + resources + part_of
        # + requires_auth + lists + reads + writes = 9 calls
        assert tx.run.call_count >= 5

    def test_all_node_labels_appear_in_queries(self) -> None:
        driver, tx = _make_mock_driver()
        build_graph(self._make_full_spec(), driver)
        all_stmts = " ".join(c.args[0] for c in tx.run.call_args_list)
        for label in (LABEL_API_SPEC, LABEL_ENDPOINT, LABEL_RESOURCE, LABEL_AUTH_SCHEME):
            assert label in all_stmts, f"Label {label!r} not found in any query"

    def test_empty_spec_returns_zero_counts(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(_spec(), driver)
        assert result.endpoint_count == 0
        assert result.resource_count == 0
        assert result.auth_scheme_count == 0

    def test_spec_id_is_url_safe(self) -> None:
        driver, _ = _make_mock_driver()
        result = build_graph(self._make_full_spec(), driver)
        assert " " not in result.spec_id
        assert ":" in result.spec_id
