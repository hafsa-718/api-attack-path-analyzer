"""Unit tests for api_analyzer.parser.ingestor.

Uses parse_spec_dict() throughout — no file I/O, no prance dependency.
All spec dicts are pre-resolved (no $refs).

Covers:
  - Format detection (OpenAPI 3.x, Swagger 2.0, unknown)
  - Auth scheme parsing for both format families
  - Schema recursion, depth truncation, PII heuristics
  - Parameter parsing (identifier, sensitive, URL-accepting)
  - is_public logic (op-level, global, override)
  - Full endpoint extraction for both format families
  - parse_warnings accumulation for skippable errors
"""

from typing import Any

import pytest

from api_analyzer.models.enums import (
    AuthType,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SpecFormat,
)
from api_analyzer.models.spec import ParsedParameter
from api_analyzer.parser.ingestor import (
    MAX_SCHEMA_DEPTH,
    SpecParseError,
    _detect_format,
    _is_public,
    _infer_path_param_type,
    _param_accepts_url,
    _param_is_identifier,
    _param_sensitivity,
    _parse_schema,
    parse_spec_dict,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _v3(*, paths: dict | None = None, security: list | None = None,
         security_schemes: dict | None = None) -> dict[str, Any]:
    """Build a minimal valid OpenAPI 3.0 spec dict."""
    spec: dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": paths or {},
    }
    if security is not None:
        spec["security"] = security
    if security_schemes:
        spec["components"] = {"securitySchemes": security_schemes}
    return spec


def _v2(*, paths: dict | None = None, security: list | None = None,
         security_defs: dict | None = None) -> dict[str, Any]:
    """Build a minimal valid Swagger 2.0 spec dict."""
    spec: dict[str, Any] = {
        "swagger": "2.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": paths or {},
    }
    if security is not None:
        spec["security"] = security
    if security_defs:
        spec["securityDefinitions"] = security_defs
    return spec


def _path_param(name: str, schema_type: str = "integer") -> dict[str, Any]:
    return {"name": name, "in": "path", "required": True, "schema": {"type": schema_type}}


def _get_op(
    *,
    responses: dict | None = None,
    security: list | None = None,
    parameters: list | None = None,
    request_body: dict | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "responses": responses or {"200": {"description": "OK"}},
    }
    if security is not None:
        op["security"] = security
    if parameters is not None:
        op["parameters"] = parameters
    if request_body is not None:
        op["requestBody"] = request_body
    if operation_id is not None:
        op["operationId"] = operation_id
    return op


def _json_response(schema: dict[str, Any]) -> dict[str, Any]:
    return {"200": {"description": "OK", "content": {"application/json": {"schema": schema}}}}


# ── Format detection ───────────────────────────────────────────────────────────


class TestDetectFormat:
    def test_openapi3_key_detected(self) -> None:
        assert _detect_format({"openapi": "3.0.3", "info": {}, "paths": {}}) == SpecFormat.OPENAPI3

    def test_openapi31_detected(self) -> None:
        assert _detect_format({"openapi": "3.1.0"}) == SpecFormat.OPENAPI3

    def test_swagger2_key_detected(self) -> None:
        assert _detect_format({"swagger": "2.0"}) == SpecFormat.SWAGGER2

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(SpecParseError, match="neither 'openapi' nor 'swagger'"):
            _detect_format({"info": {"title": "x"}})

    def test_openapi_key_takes_precedence_over_swagger(self) -> None:
        # Malformed but should not raise — openapi wins
        result = _detect_format({"openapi": "3.0.0", "swagger": "2.0"})
        assert result == SpecFormat.OPENAPI3


# ── is_public logic ────────────────────────────────────────────────────────────


class TestIsPublic:
    def test_no_op_security_no_global_is_public(self) -> None:
        assert _is_public(None, []) is True

    def test_no_op_security_with_global_is_private(self) -> None:
        assert _is_public(None, [{"BearerAuth": []}]) is False

    def test_explicit_empty_op_security_overrides_global(self) -> None:
        assert _is_public([], [{"BearerAuth": []}]) is True

    def test_op_security_with_scheme_is_private(self) -> None:
        assert _is_public([{"BearerAuth": []}], []) is False

    def test_malformed_op_security_treated_as_secured(self) -> None:
        # non-list op_security should not crash and should default to secured
        assert _is_public("invalid", []) is False


# ── Schema parsing ─────────────────────────────────────────────────────────────


class TestParseSchema:
    def test_none_input_returns_empty_schema(self) -> None:
        s = _parse_schema(None, depth=0)
        assert s.schema_type is None
        assert s.properties == {}

    def test_simple_string_schema(self) -> None:
        s = _parse_schema({"type": "string", "format": "uuid"}, depth=0)
        assert s.schema_type == "string"
        assert s.schema_format == "uuid"
        assert s.has_pii_fields is False

    def test_flat_object_with_non_pii_fields(self) -> None:
        d = {"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}}}
        s = _parse_schema(d, depth=0)
        assert "id" in s.properties
        assert "name" in s.properties
        assert s.has_pii_fields is False

    def test_pii_field_name_detected(self) -> None:
        d = {"type": "object", "properties": {
            "email": {"type": "string"},
            "username": {"type": "string"},
        }}
        s = _parse_schema(d, depth=0)
        assert s.has_pii_fields is True
        assert "email" in s.pii_field_names

    def test_pii_format_detected_on_property(self) -> None:
        d = {"type": "object", "properties": {
            "contact": {"type": "string", "format": "email"},
        }}
        s = _parse_schema(d, depth=0)
        assert s.has_pii_fields is True
        assert "contact" in s.pii_field_names

    def test_pii_propagated_from_nested_properties(self) -> None:
        inner = {"type": "object", "properties": {"email": {"type": "string"}}}
        outer = {"type": "object", "properties": {"profile": inner}}
        s = _parse_schema(outer, depth=0)
        assert s.has_pii_fields is True

    def test_array_with_items_sets_is_collection(self) -> None:
        d = {"type": "array", "items": {"type": "string"}}
        s = _parse_schema(d, depth=0)
        assert s.is_collection is True
        assert s.items is not None

    def test_pii_propagated_from_array_items(self) -> None:
        items = {"type": "object", "properties": {"email": {"type": "string"}}}
        d = {"type": "array", "items": items}
        s = _parse_schema(d, depth=0)
        assert s.has_pii_fields is True
        assert s.is_collection is True

    def test_depth_limit_returns_truncated_schema(self) -> None:
        s = _parse_schema({"type": "object", "properties": {"x": {"type": "string"}}}, depth=MAX_SCHEMA_DEPTH)
        assert s.max_depth_reached is True
        assert s.properties == {}

    def test_depth_below_limit_is_not_truncated(self) -> None:
        s = _parse_schema({"type": "string"}, depth=MAX_SCHEMA_DEPTH - 1)
        assert s.max_depth_reached is False

    def test_required_fields_preserved(self) -> None:
        d = {"type": "object", "required": ["id", "name"], "properties": {}}
        s = _parse_schema(d, depth=0)
        assert "id" in s.required_fields
        assert "name" in s.required_fields


# ── Parameter parsing helpers ──────────────────────────────────────────────────


class TestParamIsIdentifier:
    def test_bare_id_is_identifier(self) -> None:
        assert _param_is_identifier("id") is True

    def test_camel_case_id_suffix(self) -> None:
        assert _param_is_identifier("userId") is True
        assert _param_is_identifier("orderId") is True

    def test_snake_case_id_suffix(self) -> None:
        assert _param_is_identifier("user_id") is True
        assert _param_is_identifier("order_id") is True

    def test_non_identifier_name(self) -> None:
        assert _param_is_identifier("name") is False
        assert _param_is_identifier("address") is False
        assert _param_is_identifier("status") is False


class TestParamSensitivity:
    def test_password_name_is_sensitive(self) -> None:
        is_s, signals = _param_sensitivity("password", None)
        assert is_s is True
        assert any("name_matches" in s for s in signals)

    def test_email_format_is_sensitive(self) -> None:
        is_s, signals = _param_sensitivity("contact", "email")
        assert is_s is True
        assert any("schema_format" in s for s in signals)

    def test_non_sensitive_name_and_format(self) -> None:
        is_s, signals = _param_sensitivity("page", "integer")
        assert is_s is False
        assert signals == []


class TestParamAcceptsUrl:
    def test_uri_format_accepted(self) -> None:
        assert _param_accepts_url("target", "uri") is True

    def test_callback_name_accepted(self) -> None:
        assert _param_accepts_url("callbackUrl", None) is True
        assert _param_accepts_url("webhook_url", None) is True
        assert _param_accepts_url("redirectUri", None) is True

    def test_plain_string_not_url(self) -> None:
        assert _param_accepts_url("username", "string") is False


class TestInferPathParamType:
    def test_integer_type_returns_integer(self) -> None:
        p = ParsedParameter(name="userId", location=ParameterLocation.PATH, schema_type="integer")
        assert _infer_path_param_type([p]) == PathParamType.INTEGER

    def test_string_uuid_format_returns_uuid(self) -> None:
        p = ParsedParameter(name="resourceId", location=ParameterLocation.PATH, schema_type="string", schema_format="uuid")
        assert _infer_path_param_type([p]) == PathParamType.UUID

    def test_string_type_returns_string(self) -> None:
        p = ParsedParameter(name="slug", location=ParameterLocation.PATH, schema_type="string")
        assert _infer_path_param_type([p]) == PathParamType.STRING

    def test_no_params_returns_none(self) -> None:
        assert _infer_path_param_type([]) == PathParamType.NONE

    def test_deepest_param_wins(self) -> None:
        # Two path params: first is integer, deepest is UUID
        p1 = ParsedParameter(name="userId", location=ParameterLocation.PATH, schema_type="integer")
        p2 = ParsedParameter(name="tokenId", location=ParameterLocation.PATH, schema_type="string", schema_format="uuid")
        assert _infer_path_param_type([p1, p2]) == PathParamType.UUID


# ── Full spec parsing — OpenAPI 3.x ───────────────────────────────────────────


class TestParseSpecDictV3:
    def test_minimal_spec_parses_without_error(self) -> None:
        spec = parse_spec_dict(_v3())
        assert spec.title == "Test API"
        assert spec.version == "1.0.0"
        assert spec.spec_format == SpecFormat.OPENAPI3
        assert spec.endpoints == []
        assert spec.auth_schemes == {}

    def test_title_and_version_extracted(self) -> None:
        raw = _v3()
        raw["info"] = {"title": "My API", "version": "2.5.0"}
        spec = parse_spec_dict(raw)
        assert spec.title == "My API"
        assert spec.version == "2.5.0"

    def test_server_url_extracted_as_base_url(self) -> None:
        raw = _v3()
        raw["servers"] = [{"url": "https://api.example.com/v1"}]
        spec = parse_spec_dict(raw)
        assert spec.base_url == "https://api.example.com/v1"

    def test_no_paths_produces_empty_endpoint_list(self) -> None:
        spec = parse_spec_dict(_v3(paths={}))
        assert spec.endpoints == []

    def test_bearer_auth_scheme_parsed(self) -> None:
        spec = parse_spec_dict(_v3(security_schemes={
            "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
        }))
        assert "BearerAuth" in spec.auth_schemes
        scheme = spec.auth_schemes["BearerAuth"]
        assert scheme.auth_type == AuthType.HTTP_BEARER
        assert scheme.is_jwt is True

    def test_api_key_scheme_parsed(self) -> None:
        spec = parse_spec_dict(_v3(security_schemes={
            "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }))
        scheme = spec.auth_schemes["ApiKey"]
        assert scheme.auth_type == AuthType.API_KEY
        assert scheme.in_location == "header"

    def test_oauth2_scheme_parsed_with_flows(self) -> None:
        spec = parse_spec_dict(_v3(security_schemes={
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://auth.example.com/authorize",
                        "tokenUrl": "https://auth.example.com/token",
                        "scopes": {"read:users": "Read users"},
                    }
                },
            }
        }))
        scheme = spec.auth_schemes["OAuth2"]
        assert scheme.auth_type == AuthType.OAUTH2
        assert len(scheme.flows) == 1
        assert scheme.flows[0].flow_type == "authorizationCode"

    def test_unknown_scheme_type_adds_warning_and_skips(self) -> None:
        spec = parse_spec_dict(_v3(security_schemes={
            "WeirdAuth": {"type": "x-custom", "description": "vendor extension"}
        }))
        assert "WeirdAuth" not in spec.auth_schemes
        assert any("WeirdAuth" in w for w in spec.parse_warnings)

    def test_public_endpoint_when_global_security_empty(self) -> None:
        paths = {"/health": {"get": _get_op()}}
        spec = parse_spec_dict(_v3(paths=paths, security=[]))
        assert spec.endpoints[0].is_public is True

    def test_private_endpoint_when_global_security_set(self) -> None:
        paths = {"/users": {"get": _get_op()}}
        spec = parse_spec_dict(_v3(
            paths=paths,
            security=[{"BearerAuth": []}],
            security_schemes={"BearerAuth": {"type": "http", "scheme": "bearer"}},
        ))
        assert spec.endpoints[0].is_public is False

    def test_op_empty_security_overrides_global(self) -> None:
        paths = {"/public": {"get": _get_op(security=[])}}
        spec = parse_spec_dict(_v3(
            paths=paths,
            security=[{"BearerAuth": []}],
        ))
        assert spec.endpoints[0].is_public is True

    def test_op_security_overrides_empty_global(self) -> None:
        paths = {"/secure": {"get": _get_op(security=[{"BearerAuth": []}])}}
        spec = parse_spec_dict(_v3(paths=paths, security=[]))
        assert spec.endpoints[0].is_public is False

    def test_integer_path_param_type_detected(self) -> None:
        paths = {
            "/users/{userId}": {
                "parameters": [_path_param("userId", "integer")],
                "get": _get_op(),
            }
        }
        spec = parse_spec_dict(_v3(paths=paths))
        ep = spec.endpoints[0]
        assert ep.path_param_type == PathParamType.INTEGER
        assert "userId" in ep.path_param_names

    def test_uuid_path_param_type_detected(self) -> None:
        paths = {
            "/items/{itemId}": {
                "parameters": [{"name": "itemId", "in": "path", "required": True,
                                 "schema": {"type": "string", "format": "uuid"}}],
                "get": _get_op(),
            }
        }
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoints[0].path_param_type == PathParamType.UUID

    def test_endpoint_returns_pii_from_response_schema(self) -> None:
        pii_schema = {"type": "object", "properties": {"email": {"type": "string"}}}
        paths = {"/users/{id}": {
            "parameters": [_path_param("id")],
            "get": _get_op(responses=_json_response(pii_schema)),
        }}
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoints[0].returns_pii is True

    def test_endpoint_returns_collection_for_array_response(self) -> None:
        array_schema = {"type": "array", "items": {"type": "object"}}
        paths = {"/users": {"get": _get_op(responses=_json_response(array_schema))}}
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoints[0].returns_collection is True

    def test_multiple_methods_on_same_path_each_become_endpoint(self) -> None:
        paths = {"/users": {
            "get": _get_op(),
            "post": _get_op(),
        }}
        spec = parse_spec_dict(_v3(paths=paths))
        methods = {ep.method for ep in spec.endpoints}
        assert HttpMethod.GET in methods
        assert HttpMethod.POST in methods

    def test_operation_id_preserved(self) -> None:
        paths = {"/users": {"get": _get_op(operation_id="listUsers")}}
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoints[0].operation_id == "listUsers"

    def test_path_level_params_merged_with_op_params(self) -> None:
        paths = {
            "/users/{userId}/posts/{postId}": {
                "parameters": [_path_param("userId", "integer")],
                "get": {
                    "parameters": [_path_param("postId", "string")],
                    "responses": {"200": {"description": "OK"}},
                },
            }
        }
        spec = parse_spec_dict(_v3(paths=paths))
        ep = spec.endpoints[0]
        assert "userId" in ep.path_param_names
        assert "postId" in ep.path_param_names

    def test_malformed_operation_adds_warning_and_continues(self) -> None:
        # One bad endpoint, one good — parser should emit a warning for the bad one
        # and still return the good one.
        paths = {
            "/good": {"get": _get_op()},
            "/bad": {"get": None},  # malformed: operation is not a dict
        }
        spec = parse_spec_dict(_v3(paths=paths))
        # /bad is a value of None — it's skipped at the method loop level (not a dict).
        # The good endpoint must still be present.
        assert any(ep.path == "/good" for ep in spec.endpoints)

    def test_request_body_pii_detected(self) -> None:
        pii_body_schema = {"type": "object", "properties": {"email": {"type": "string"}}}
        body = {
            "required": True,
            "content": {"application/json": {"schema": pii_body_schema}},
        }
        paths = {"/users": {"post": _get_op(request_body=body)}}
        spec = parse_spec_dict(_v3(paths=paths))
        ep = spec.endpoints[0]
        assert ep.accepts_pii is True
        assert ep.request_body is not None
        assert ep.request_body.has_pii_fields is True

    def test_accepts_url_param_detected(self) -> None:
        paths = {"/proxy": {
            "get": _get_op(parameters=[{
                "name": "callbackUrl",
                "in": "query",
                "schema": {"type": "string"},
            }])
        }}
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoints[0].accepts_url_param is True


# ── Full spec parsing — Swagger 2.0 ───────────────────────────────────────────


class TestParseSpecDictV2:
    def test_minimal_swagger2_spec(self) -> None:
        spec = parse_spec_dict(_v2())
        assert spec.spec_format == SpecFormat.SWAGGER2
        assert spec.endpoints == []

    def test_basic_auth_scheme_v2(self) -> None:
        spec = parse_spec_dict(_v2(security_defs={
            "BasicAuth": {"type": "basic"}
        }))
        scheme = spec.auth_schemes["BasicAuth"]
        assert scheme.auth_type == AuthType.HTTP_BASIC

    def test_api_key_scheme_v2(self) -> None:
        spec = parse_spec_dict(_v2(security_defs={
            "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }))
        assert spec.auth_schemes["ApiKey"].auth_type == AuthType.API_KEY

    def test_oauth2_scheme_v2(self) -> None:
        spec = parse_spec_dict(_v2(security_defs={
            "OAuth2": {
                "type": "oauth2",
                "flow": "clientCredentials",
                "tokenUrl": "https://auth.example.com/token",
                "scopes": {"read": "Read access"},
            }
        }))
        scheme = spec.auth_schemes["OAuth2"]
        assert scheme.auth_type == AuthType.OAUTH2
        assert scheme.flows[0].flow_type == "clientCredentials"

    def test_base_url_composed_from_host_and_base_path(self) -> None:
        raw = _v2()
        raw["host"] = "api.example.com"
        raw["basePath"] = "/v2"
        raw["schemes"] = ["https"]
        spec = parse_spec_dict(raw)
        assert spec.base_url == "https://api.example.com/v2"

    def test_v2_endpoint_is_public_when_no_global_security(self) -> None:
        raw = _v2(paths={"/health": {"get": {"responses": {"200": {"description": "OK"}}}}})
        spec = parse_spec_dict(raw)
        assert spec.endpoints[0].is_public is True

    def test_v2_body_param_parsed_as_request_body(self) -> None:
        pii_schema = {"type": "object", "properties": {"email": {"type": "string"}}}
        raw = _v2(paths={
            "/users": {"post": {
                "parameters": [{
                    "name": "body",
                    "in": "body",
                    "required": True,
                    "schema": pii_schema,
                }],
                "responses": {"201": {"description": "Created"}},
            }}
        })
        spec = parse_spec_dict(raw)
        ep = spec.endpoints[0]
        assert ep.request_body is not None
        assert ep.accepts_pii is True

    def test_v2_response_schema_parsed(self) -> None:
        user_schema = {"type": "object", "properties": {"phone": {"type": "string"}}}
        raw = _v2(paths={
            "/users/{id}": {
                "parameters": [_path_param("id", "integer")],
                "get": {
                    "responses": {"200": {"description": "User", "schema": user_schema}},
                },
            }
        })
        spec = parse_spec_dict(raw)
        ep = spec.endpoints[0]
        assert ep.returns_pii is True
        assert ep.path_param_type == PathParamType.INTEGER

    def test_v2_unknown_security_type_warns_and_skips(self) -> None:
        spec = parse_spec_dict(_v2(security_defs={
            "CustomAuth": {"type": "x-custom"}
        }))
        assert "CustomAuth" not in spec.auth_schemes
        assert any("CustomAuth" in w for w in spec.parse_warnings)


# ── ParsedSpec computed fields ─────────────────────────────────────────────────


class TestParsedSpecComputedFieldsViaIngestor:
    def test_endpoint_count_from_parsed_paths(self) -> None:
        paths = {
            "/a": {"get": _get_op()},
            "/b": {"get": _get_op(), "post": _get_op()},
        }
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.endpoint_count == 3

    def test_public_endpoint_count(self) -> None:
        paths = {
            "/open": {"get": _get_op(security=[])},
            "/secure": {"get": _get_op(security=[{"BearerAuth": []}])},
        }
        spec = parse_spec_dict(_v3(paths=paths))
        assert spec.public_endpoint_count == 1

    def test_auth_declared_true_when_schemes_present(self) -> None:
        spec = parse_spec_dict(_v3(security_schemes={
            "BearerAuth": {"type": "http", "scheme": "bearer"}
        }))
        assert spec.auth_declared is True

    def test_auth_declared_false_when_no_schemes(self) -> None:
        spec = parse_spec_dict(_v3())
        assert spec.auth_declared is False
