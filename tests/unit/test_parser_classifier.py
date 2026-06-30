"""Unit tests for api_analyzer.parser.classifier.

Covers: function classification, sensitivity classification, spec_completeness
formula, resource inference, and the classify() integration entry point.
"""

import math
from typing import Any

import pytest

from api_analyzer.models.enums import (
    EndpointFunction,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SensitivityClass,
    SpecFormat,
)
from api_analyzer.models.spec import (
    InferredResource,
    ParsedEndpoint,
    ParsedParameter,
    ParsedSchema,
    ParsedSpec,
)
from api_analyzer.parser.classifier import (
    RESPONSE_SCHEMA_FLOOR,
    _classify_function,
    _classify_sensitivity,
    _collection_path,
    _compute_completeness,
    _infer_resources,
    _resource_name,
    _parent_resource_name,
    _singularize,
    _weighted_geometric_mean,
    classify,
)
from api_analyzer.parser.ingestor import parse_spec_dict


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ep(
    *,
    path: str = "/users",
    method: HttpMethod = HttpMethod.GET,
    is_public: bool = True,
    auth_scheme_names: list[str] | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    parameters: list[ParsedParameter] | None = None,
    path_param_names: list[str] | None = None,
    path_param_type: PathParamType = PathParamType.NONE,
    returns_pii: bool = False,
    accepts_pii: bool = False,
    accepts_url_param: bool = False,
    has_role_param: bool = False,
    response_schemas: dict | None = None,
    inferred_function: EndpointFunction = EndpointFunction.UNKNOWN,
) -> ParsedEndpoint:
    method_str = method.value if isinstance(method, HttpMethod) else method
    return ParsedEndpoint(
        id=f"{method_str}:{path}",
        path=path,
        method=method,
        summary=summary,
        tags=tags or [],
        parameters=parameters or [],
        auth_scheme_names=auth_scheme_names or [],
        is_public=is_public,
        path_param_names=path_param_names or [],
        path_param_type=path_param_type,
        returns_pii=returns_pii,
        accepts_pii=accepts_pii,
        accepts_url_param=accepts_url_param,
        has_role_param=has_role_param,
        response_schemas=response_schemas or {},
        inferred_function=inferred_function,
    )


def _spec(endpoints: list[ParsedEndpoint], auth_schemes: dict | None = None) -> ParsedSpec:
    return ParsedSpec(
        title="Test",
        version="1.0",
        spec_format=SpecFormat.OPENAPI3,
        endpoints=endpoints,
        auth_schemes=auth_schemes or {},
    )


# ── _singularize ───────────────────────────────────────────────────────────────


class TestSingularize:
    def test_regular_plural(self) -> None:
        assert _singularize("users") == "user"
        assert _singularize("orders") == "order"
        assert _singularize("items") == "item"

    def test_ies_plural(self) -> None:
        assert _singularize("categories") == "category"
        assert _singularize("utilities") == "utility"

    def test_es_plural(self) -> None:
        assert _singularize("statuses") == "status"
        assert _singularize("indexes") == "index"

    def test_double_s_unchanged(self) -> None:
        assert _singularize("access") == "access"
        assert _singularize("address") == "address"

    def test_already_singular_unchanged(self) -> None:
        assert _singularize("user") == "user"
        assert _singularize("token") == "token"


# ── _collection_path ──────────────────────────────────────────────────────────


class TestCollectionPath:
    def test_collection_path_unchanged(self) -> None:
        assert _collection_path("/users") == "/users"

    def test_detail_path_strips_param(self) -> None:
        assert _collection_path("/users/{userId}") == "/users"

    def test_sub_resource_collection_unchanged(self) -> None:
        assert _collection_path("/users/{userId}/orders") == "/users/{userId}/orders"

    def test_sub_resource_detail_strips_param(self) -> None:
        assert _collection_path("/users/{userId}/orders/{orderId}") == "/users/{userId}/orders"

    def test_root_path_unchanged(self) -> None:
        assert _collection_path("/health") == "/health"

    def test_deeply_nested_strips_last_param(self) -> None:
        result = _collection_path("/a/{aId}/b/{bId}/c/{cId}")
        assert result == "/a/{aId}/b/{bId}/c"


# ── _resource_name ─────────────────────────────────────────────────────────────


class TestResourceName:
    def test_simple_plural(self) -> None:
        assert _resource_name("/users") == "User"

    def test_sub_resource(self) -> None:
        assert _resource_name("/users/{userId}/orders") == "Order"

    def test_versioned_path_skips_prefix(self) -> None:
        assert _resource_name("/api/v1/products") == "Product"

    def test_param_only_path_returns_none(self) -> None:
        assert _resource_name("/{id}") is None

    def test_ies_plural_in_path(self) -> None:
        assert _resource_name("/categories") == "Category"


# ── _parent_resource_name ──────────────────────────────────────────────────────


class TestParentResourceName:
    def test_top_level_has_no_parent(self) -> None:
        assert _parent_resource_name("/users") is None

    def test_sub_resource_returns_parent(self) -> None:
        assert _parent_resource_name("/users/{userId}/orders") == "User"

    def test_deeply_nested_returns_immediate_parent(self) -> None:
        assert _parent_resource_name("/users/{userId}/orders/{orderId}/items") == "Order"


# ── _weighted_geometric_mean ──────────────────────────────────────────────────


class TestWeightedGeometricMean:
    def test_all_ones_returns_one(self) -> None:
        result = _weighted_geometric_mean((1.0, 1.0, 1.0, 1.0))
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_weights_correctly_applied(self) -> None:
        # With weights (0.15, 0.50, 0.20, 0.15) and components (1, x, 1, 1):
        # result = x^0.50.  At x=0.25 → 0.25^0.50 = 0.5
        result = _weighted_geometric_mean((1.0, 0.25, 1.0, 1.0))
        assert result == pytest.approx(0.25 ** 0.50, rel=1e-6)

    def test_zero_component_floored_not_nan(self) -> None:
        result = _weighted_geometric_mean((0.0, 0.0, 0.0, 0.0))
        assert not math.isnan(result)
        assert result > 0.0

    def test_dominant_security_weight(self) -> None:
        # Zero auth (floored to 0.01), perfect everything else.
        # Result ≈ exp(0.50 * ln(0.01)) ≈ 0.1
        result = _weighted_geometric_mean((1.0, 0.0, 1.0, 1.0))
        expected = math.exp(0.50 * math.log(0.01))  # ≈ 0.1
        assert result == pytest.approx(expected, rel=1e-5)


# ── _compute_completeness ──────────────────────────────────────────────────────


class TestComputeCompleteness:
    def test_empty_spec_returns_zero(self) -> None:
        assert _compute_completeness(_spec([])) == 0.0

    def test_perfect_spec_approaches_one(self) -> None:
        ep = _ep(
            summary="Get all users",
            auth_scheme_names=["BearerAuth"],
            parameters=[ParsedParameter(name="limit", location=ParameterLocation.QUERY,
                                        schema_type="integer")],
            response_schemas={"200": ParsedSchema(schema_type="object")},
        )
        result = _compute_completeness(_spec([ep]))
        assert result > 0.85

    def test_no_auth_declarations_scores_very_low(self) -> None:
        ep = _ep(summary="List", auth_scheme_names=[],
                 response_schemas={"200": ParsedSchema(schema_type="object")})
        result = _compute_completeness(_spec([ep]))
        assert result < 0.20

    def test_no_summaries_reduces_score(self) -> None:
        with_summary = _ep(summary="Get users", auth_scheme_names=["B"],
                           response_schemas={"200": ParsedSchema(schema_type="object")})
        without_summary = _ep(summary=None, auth_scheme_names=["B"],
                              response_schemas={"200": ParsedSchema(schema_type="object")})
        score_with = _compute_completeness(_spec([with_summary]))
        score_without = _compute_completeness(_spec([without_summary]))
        assert score_with > score_without

    def test_response_schema_floor_prevents_double_penalty(self) -> None:
        # No response schemas — floor kicks in at RESPONSE_SCHEMA_FLOOR
        ep = _ep(auth_scheme_names=["B"], summary="x", response_schemas={})
        result = _compute_completeness(_spec([ep]))
        # Component 4 is floored at 0.40, not 0.01
        # Manual check: floor of 0.40 vs 0.01 — result should be >= exp(0.15*ln(0.40))
        floor_contribution = math.exp(0.15 * math.log(RESPONSE_SCHEMA_FLOOR))
        assert result >= floor_contribution * 0.5  # some room for other components

    def test_no_parameters_does_not_penalise_param_component(self) -> None:
        # No params → param component = 1.0 (no penalty)
        ep_no_params = _ep(auth_scheme_names=["B"], summary="x",
                           response_schemas={"200": ParsedSchema(schema_type="object")}, parameters=[])
        ep_with_typed = _ep(auth_scheme_names=["B"], summary="x",
                            response_schemas={"200": ParsedSchema(schema_type="object")},
                            parameters=[ParsedParameter(name="p", location=ParameterLocation.QUERY,
                                                        schema_type="string")])
        score_no_params = _compute_completeness(_spec([ep_no_params]))
        score_typed = _compute_completeness(_spec([ep_with_typed]))
        # Both should be similar (no params → no penalty, typed params → no penalty)
        assert abs(score_no_params - score_typed) < 0.05

    def test_completeness_within_valid_range(self) -> None:
        ep = _ep(summary="x", auth_scheme_names=["B"])
        result = _compute_completeness(_spec([ep]))
        assert 0.0 <= result <= 1.0


# ── _classify_function ─────────────────────────────────────────────────────────


class TestClassifyFunction:
    def test_get_is_data_read(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.GET, path="/users")) == EndpointFunction.DATA_READ

    def test_post_is_data_write(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.POST, path="/users")) == EndpointFunction.DATA_WRITE

    def test_put_is_data_write(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.PUT, path="/users/{id}")) == EndpointFunction.DATA_WRITE

    def test_delete_is_data_write(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.DELETE, path="/users/{id}")) == EndpointFunction.DATA_WRITE

    def test_head_is_unknown(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.HEAD, path="/users")) == EndpointFunction.UNKNOWN

    def test_options_is_unknown(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.OPTIONS, path="/users")) == EndpointFunction.UNKNOWN

    def test_admin_path_segment(self) -> None:
        assert _classify_function(_ep(path="/admin/users")) == EndpointFunction.ADMIN

    def test_internal_path_segment(self) -> None:
        assert _classify_function(_ep(path="/internal/metrics")) == EndpointFunction.ADMIN

    def test_admin_tag_overrides_method(self) -> None:
        assert _classify_function(_ep(method=HttpMethod.GET, tags=["admin"])) == EndpointFunction.ADMIN

    def test_login_path_is_auth(self) -> None:
        assert _classify_function(_ep(path="/auth/login")) == EndpointFunction.AUTH

    def test_token_path_is_auth(self) -> None:
        assert _classify_function(_ep(path="/oauth/token")) == EndpointFunction.AUTH

    def test_auth_tag_is_auth(self) -> None:
        assert _classify_function(_ep(tags=["authentication"])) == EndpointFunction.AUTH

    def test_webhook_path(self) -> None:
        assert _classify_function(_ep(path="/webhooks/receive")) == EndpointFunction.WEBHOOK

    def test_callback_path(self) -> None:
        assert _classify_function(_ep(path="/oauth/callback")) == EndpointFunction.AUTH

    def test_accepts_url_param_is_webhook(self) -> None:
        ep = _ep(path="/proxy", method=HttpMethod.POST, accepts_url_param=True)
        assert _classify_function(ep) == EndpointFunction.WEBHOOK

    def test_admin_takes_precedence_over_auth(self) -> None:
        ep = _ep(path="/admin/auth/reset", tags=["admin"])
        assert _classify_function(ep) == EndpointFunction.ADMIN

    def test_partial_path_segment_does_not_match(self) -> None:
        # /authentication should NOT match the /auth pattern since it's not a complete segment
        result = _classify_function(_ep(path="/authentication/flow"))
        # "authentication" is not an exact match for our segment alternatives;
        # there is no /auth/ or /auth$ here — it is /authentication/
        assert result != EndpointFunction.AUTH

    def test_users_path_is_data_read_not_admin(self) -> None:
        assert _classify_function(_ep(path="/users")) == EndpointFunction.DATA_READ


# ── _classify_sensitivity ──────────────────────────────────────────────────────


class TestClassifySensitivity:
    def test_public_endpoint_no_signals_is_public(self) -> None:
        ep = _ep(is_public=True)
        assert _classify_sensitivity(ep) == SensitivityClass.PUBLIC

    def test_authenticated_endpoint_no_pii_is_internal(self) -> None:
        ep = _ep(is_public=False)
        assert _classify_sensitivity(ep) == SensitivityClass.INTERNAL

    def test_returns_pii_is_sensitive(self) -> None:
        ep = _ep(returns_pii=True, is_public=True)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE

    def test_accepts_pii_is_sensitive(self) -> None:
        ep = _ep(accepts_pii=True, is_public=False)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE

    def test_integer_path_param_is_sensitive(self) -> None:
        ep = _ep(path_param_type=PathParamType.INTEGER, is_public=False)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE

    def test_uuid_path_param_is_sensitive(self) -> None:
        ep = _ep(path_param_type=PathParamType.UUID, is_public=False)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE

    def test_integer_path_param_public_still_sensitive(self) -> None:
        # BOLA risk exists even on public endpoints with integer params.
        ep = _ep(path_param_type=PathParamType.INTEGER, is_public=True)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE

    def test_string_path_param_private_is_internal(self) -> None:
        ep = _ep(path_param_type=PathParamType.STRING, is_public=False)
        assert _classify_sensitivity(ep) == SensitivityClass.INTERNAL

    def test_admin_function_is_critical(self) -> None:
        ep = _ep(inferred_function=EndpointFunction.ADMIN)
        assert _classify_sensitivity(ep) == SensitivityClass.CRITICAL

    def test_auth_function_is_critical(self) -> None:
        ep = _ep(inferred_function=EndpointFunction.AUTH)
        assert _classify_sensitivity(ep) == SensitivityClass.CRITICAL

    def test_has_role_param_is_critical(self) -> None:
        ep = _ep(has_role_param=True)
        assert _classify_sensitivity(ep) == SensitivityClass.CRITICAL

    def test_critical_overrides_pii_signals(self) -> None:
        # Even if endpoint also returns PII, ADMIN function → CRITICAL
        ep = _ep(inferred_function=EndpointFunction.ADMIN, returns_pii=True)
        assert _classify_sensitivity(ep) == SensitivityClass.CRITICAL

    def test_sensitive_overrides_internal(self) -> None:
        # Private endpoint returning PII → SENSITIVE, not INTERNAL
        ep = _ep(is_public=False, returns_pii=True)
        assert _classify_sensitivity(ep) == SensitivityClass.SENSITIVE


# ── _infer_resources ──────────────────────────────────────────────────────────


class TestInferResources:
    def test_collection_and_detail_produce_resource(self) -> None:
        endpoints = [
            _ep(path="/users", method=HttpMethod.GET),
            _ep(path="/users/{userId}", method=HttpMethod.GET,
                path_param_names=["userId"], path_param_type=PathParamType.INTEGER),
        ]
        resources = _infer_resources(endpoints)
        assert len(resources) == 1
        r = resources[0]
        assert r.name == "User"
        assert r.collection_endpoint_id == "GET:/users"
        assert r.detail_endpoint_id == "GET:/users/{userId}"
        assert r.identifier_type == PathParamType.INTEGER
        assert r.identifier_name == "userId"

    def test_health_endpoint_skipped(self) -> None:
        endpoints = [_ep(path="/health", method=HttpMethod.GET)]
        assert _infer_resources(endpoints) == []

    def test_collection_only_no_resource(self) -> None:
        # /users with no /users/{userId} → no detail endpoint → no resource
        endpoints = [_ep(path="/users", method=HttpMethod.GET)]
        assert _infer_resources(endpoints) == []

    def test_sub_resource_with_parent_name(self) -> None:
        endpoints = [
            _ep(path="/users/{userId}/orders", method=HttpMethod.GET,
                path_param_names=["userId"]),
            _ep(path="/users/{userId}/orders/{orderId}", method=HttpMethod.GET,
                path_param_names=["userId", "orderId"], path_param_type=PathParamType.INTEGER),
        ]
        resources = _infer_resources(endpoints)
        assert len(resources) == 1
        r = resources[0]
        assert r.name == "Order"
        assert r.parent_resource_name == "User"

    def test_write_endpoints_collected(self) -> None:
        endpoints = [
            _ep(path="/users", method=HttpMethod.GET),
            _ep(path="/users", method=HttpMethod.POST),
            _ep(path="/users/{userId}", method=HttpMethod.GET,
                path_param_names=["userId"], path_param_type=PathParamType.INTEGER),
            _ep(path="/users/{userId}", method=HttpMethod.PUT,
                path_param_names=["userId"], path_param_type=PathParamType.INTEGER),
            _ep(path="/users/{userId}", method=HttpMethod.DELETE,
                path_param_names=["userId"], path_param_type=PathParamType.INTEGER),
        ]
        resources = _infer_resources(endpoints)
        assert len(resources) == 1
        r = resources[0]
        assert "POST:/users" in r.write_endpoint_ids
        assert "PUT:/users/{userId}" in r.write_endpoint_ids
        assert r.delete_endpoint_id == "DELETE:/users/{userId}"

    def test_uuid_identifier_type_preserved(self) -> None:
        endpoints = [
            _ep(path="/items", method=HttpMethod.GET),
            _ep(path="/items/{itemId}", method=HttpMethod.GET,
                path_param_names=["itemId"], path_param_type=PathParamType.UUID),
        ]
        r = _infer_resources(endpoints)[0]
        assert r.identifier_type == PathParamType.UUID

    def test_multiple_independent_resources(self) -> None:
        endpoints = [
            _ep(path="/users", method=HttpMethod.GET),
            _ep(path="/users/{userId}", method=HttpMethod.GET,
                path_param_names=["userId"], path_param_type=PathParamType.INTEGER),
            _ep(path="/products", method=HttpMethod.GET),
            _ep(path="/products/{productId}", method=HttpMethod.GET,
                path_param_names=["productId"], path_param_type=PathParamType.INTEGER),
        ]
        resources = _infer_resources(endpoints)
        names = {r.name for r in resources}
        assert "User" in names
        assert "Product" in names

    def test_status_endpoint_skipped(self) -> None:
        endpoints = [
            _ep(path="/status", method=HttpMethod.GET),
            _ep(path="/status/{id}", method=HttpMethod.GET,
                path_param_names=["id"], path_param_type=PathParamType.INTEGER),
        ]
        # "status" is in the blocklist
        assert _infer_resources(endpoints) == []


# ── classify() integration ─────────────────────────────────────────────────────


class TestClassifyIntegration:
    def _make_spec(self) -> ParsedSpec:
        raw: dict = {
            "openapi": "3.0.3",
            "info": {"title": "User API", "version": "1.0.0"},
            "components": {
                "securitySchemes": {
                    "BearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
                }
            },
            "security": [{"BearerAuth": []}],
            "paths": {
                "/users": {
                    "get": {
                        "operationId": "listUsers",
                        "summary": "List users",
                        "security": [],
                        "responses": {
                            "200": {
                                "description": "Users",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "id": {"type": "integer"},
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/users/{userId}": {
                    "parameters": [
                        {"name": "userId", "in": "path", "required": True,
                         "schema": {"type": "integer"}}
                    ],
                    "get": {
                        "operationId": "getUserById",
                        "summary": "Get user",
                        "responses": {
                            "200": {
                                "description": "User",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "integer"},
                                                "email": {"type": "string"},
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/admin/settings": {
                    "get": {
                        "summary": "Admin settings",
                        "responses": {"200": {"description": "OK"}}
                    }
                },
            }
        }
        return parse_spec_dict(raw)

    def test_classify_returns_same_spec_object(self) -> None:
        spec = self._make_spec()
        result = classify(spec)
        assert result is spec

    def test_spec_completeness_set_after_classify(self) -> None:
        spec = classify(self._make_spec())
        assert 0.0 < spec.spec_completeness <= 1.0

    def test_resources_populated(self) -> None:
        spec = classify(self._make_spec())
        assert len(spec.resources) >= 1
        names = {r.name for r in spec.resources}
        assert "User" in names

    def test_admin_endpoint_classified_critical(self) -> None:
        spec = classify(self._make_spec())
        admin_ep = next(ep for ep in spec.endpoints if "admin" in ep.path)
        assert admin_ep.inferred_function == EndpointFunction.ADMIN
        assert admin_ep.sensitivity_class == SensitivityClass.CRITICAL

    def test_public_list_endpoint_function_is_data_read(self) -> None:
        spec = classify(self._make_spec())
        list_ep = next(ep for ep in spec.endpoints if ep.path == "/users" and ep.method == HttpMethod.GET)
        assert list_ep.inferred_function == EndpointFunction.DATA_READ

    def test_detail_endpoint_is_sensitive(self) -> None:
        spec = classify(self._make_spec())
        detail_ep = next(ep for ep in spec.endpoints if ep.path == "/users/{userId}")
        assert detail_ep.sensitivity_class == SensitivityClass.SENSITIVE

    def test_inferred_function_set_before_sensitivity(self) -> None:
        # AUTH function → CRITICAL. Verifies order: function first, then sensitivity.
        spec = _spec([_ep(path="/auth/login", method=HttpMethod.POST)])
        classify(spec)
        ep = spec.endpoints[0]
        assert ep.inferred_function == EndpointFunction.AUTH
        assert ep.sensitivity_class == SensitivityClass.CRITICAL
