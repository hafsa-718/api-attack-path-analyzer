"""Unit tests for api_analyzer.models.spec.

Covers: ParsedSchema self-referential model, ParsedEndpoint id validator,
ParsedSpec computed fields, and the full model hierarchy from OAuthFlow
down to ParsedSpec.
"""

import pytest
from pydantic import ValidationError

from api_analyzer.models import (
    AuthScheme,
    AuthType,
    EndpointFunction,
    HttpMethod,
    InferredResource,
    OAuthFlow,
    ParsedEndpoint,
    ParsedParameter,
    ParsedRequestBody,
    ParsedSchema,
    ParsedSpec,
    ParameterLocation,
    PathParamType,
    SensitivityClass,
    SpecFormat,
)


class TestParsedSchemaRecursion:
    """ParsedSchema is self-referential — model_rebuild() must have been called."""

    def test_accepts_flat_schema(self) -> None:
        schema = ParsedSchema(schema_type="string")
        assert schema.schema_type == "string"
        assert schema.properties == {}

    def test_accepts_one_level_nested_properties(self) -> None:
        child = ParsedSchema(schema_type="string")
        parent = ParsedSchema(
            schema_type="object",
            properties={"name": child},
        )
        assert parent.properties["name"].schema_type == "string"

    def test_accepts_two_level_nested_properties(self) -> None:
        grandchild = ParsedSchema(schema_type="integer")
        child = ParsedSchema(schema_type="object", properties={"age": grandchild})
        parent = ParsedSchema(schema_type="object", properties={"profile": child})
        assert parent.properties["profile"].properties["age"].schema_type == "integer"

    def test_accepts_array_schema_with_items(self) -> None:
        item_schema = ParsedSchema(schema_type="string")
        array_schema = ParsedSchema(
            schema_type="array",
            items=item_schema,
            is_collection=True,
        )
        assert array_schema.items is not None
        assert array_schema.items.schema_type == "string"
        assert array_schema.is_collection is True

    def test_items_defaults_to_none(self) -> None:
        schema = ParsedSchema(schema_type="object")
        assert schema.items is None

    def test_max_depth_reached_flag_is_preserved(self) -> None:
        schema = ParsedSchema(schema_type="object", max_depth_reached=True)
        assert schema.max_depth_reached is True
        assert schema.properties == {}

    def test_pii_fields_propagated(self) -> None:
        schema = ParsedSchema(
            schema_type="object",
            has_pii_fields=True,
            pii_field_names=["email", "ssn"],
        )
        assert schema.has_pii_fields is True
        assert "email" in schema.pii_field_names

    def test_ref_name_preserved(self) -> None:
        schema = ParsedSchema(ref_name="UserProfile", schema_type="object")
        assert schema.ref_name == "UserProfile"

    def test_schema_defaults(self) -> None:
        schema = ParsedSchema()
        assert schema.title is None
        assert schema.schema_type is None
        assert schema.properties == {}
        assert schema.required_fields == []
        assert schema.has_pii_fields is False
        assert schema.has_sensitive_fields is False
        assert schema.max_depth_reached is False


class TestParsedEndpointIdValidator:
    """ParsedEndpoint.id must be in 'METHOD:path' format."""

    def test_valid_id_accepted(self) -> None:
        ep = ParsedEndpoint(id="GET:/users/{userId}", path="/users/{userId}", method=HttpMethod.GET, is_public=False)
        assert ep.id == "GET:/users/{userId}"

    def test_valid_root_path_accepted(self) -> None:
        ep = ParsedEndpoint(id="GET:/", path="/", method=HttpMethod.GET, is_public=True)
        assert ep.id == "GET:/"

    def test_valid_post_method_accepted(self) -> None:
        ep = ParsedEndpoint(id="POST:/users", path="/users", method=HttpMethod.POST, is_public=False)
        assert ep.id == "POST:/users"

    def test_missing_colon_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ParsedEndpoint(id="GET/users", path="/users", method=HttpMethod.GET, is_public=False)
        assert "METHOD:path" in str(exc_info.value)

    def test_path_not_starting_with_slash_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ParsedEndpoint(id="GET:users", path="users", method=HttpMethod.GET, is_public=False)
        assert "must start with '/'" in str(exc_info.value)

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParsedEndpoint(id="", path="/users", method=HttpMethod.GET, is_public=False)

    def test_id_with_nested_path_accepted(self) -> None:
        ep = ParsedEndpoint(
            id="GET:/users/{userId}/orders/{orderId}",
            path="/users/{userId}/orders/{orderId}",
            method=HttpMethod.GET,
            is_public=False,
        )
        assert ep.id == "GET:/users/{userId}/orders/{orderId}"


class TestParsedEndpointDefaults:
    """Fields not provided should resolve to safe defaults."""

    def test_minimal_endpoint_construction(self) -> None:
        ep = ParsedEndpoint(
            id="GET:/health",
            path="/health",
            method=HttpMethod.GET,
            is_public=True,
        )
        assert ep.tags == []
        assert ep.parameters == []
        assert ep.request_body is None
        assert ep.auth_scheme_names == []
        assert ep.sensitivity_class == SensitivityClass.PUBLIC
        assert ep.inferred_function == EndpointFunction.UNKNOWN
        assert ep.path_param_names == []
        assert ep.path_param_type == PathParamType.NONE
        assert ep.returns_pii is False
        assert ep.accepts_pii is False
        assert ep.accepts_url_param is False
        assert ep.returns_collection is False
        assert ep.has_role_param is False
        assert ep.resource_name is None

    def test_is_public_stored_correctly(self) -> None:
        pub = ParsedEndpoint(id="GET:/open", path="/open", method=HttpMethod.GET, is_public=True)
        priv = ParsedEndpoint(id="GET:/private", path="/private", method=HttpMethod.GET, is_public=False)
        assert pub.is_public is True
        assert priv.is_public is False


class TestParsedSpecComputedFields:
    """Computed fields must derive from the data, not be settable directly."""

    def test_endpoint_count_reflects_endpoints_list(self, sample_spec: ParsedSpec) -> None:
        assert sample_spec.endpoint_count == len(sample_spec.endpoints)

    def test_endpoint_count_zero_for_empty_spec(self) -> None:
        spec = ParsedSpec(title="Empty", version="0", spec_format=SpecFormat.OPENAPI3)
        assert spec.endpoint_count == 0

    def test_public_endpoint_count_counts_is_public_true(self, sample_spec: ParsedSpec) -> None:
        public_count = sum(1 for e in sample_spec.endpoints if e.is_public)
        assert sample_spec.public_endpoint_count == public_count

    def test_public_endpoint_count_zero_when_all_private(self) -> None:
        ep = ParsedEndpoint(id="GET:/private", path="/private", method=HttpMethod.GET, is_public=False)
        spec = ParsedSpec(title="Priv", version="1", spec_format=SpecFormat.OPENAPI3, endpoints=[ep])
        assert spec.public_endpoint_count == 0

    def test_auth_declared_true_when_schemes_present(self, sample_spec: ParsedSpec) -> None:
        assert sample_spec.auth_declared is True

    def test_auth_declared_false_when_no_schemes(self) -> None:
        spec = ParsedSpec(title="NoAuth", version="1", spec_format=SpecFormat.OPENAPI3)
        assert spec.auth_declared is False

    def test_computed_fields_included_in_serialisation(self, sample_spec: ParsedSpec) -> None:
        data = sample_spec.model_dump()
        assert "endpoint_count" in data
        assert "public_endpoint_count" in data
        assert "auth_declared" in data

    def test_spec_completeness_within_bounds(self) -> None:
        spec = ParsedSpec(
            title="T",
            version="1",
            spec_format=SpecFormat.OPENAPI3,
            spec_completeness=0.85,
        )
        assert spec.spec_completeness == 0.85

    def test_spec_completeness_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParsedSpec(title="T", version="1", spec_format=SpecFormat.OPENAPI3, spec_completeness=-0.1)

    def test_spec_completeness_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParsedSpec(title="T", version="1", spec_format=SpecFormat.OPENAPI3, spec_completeness=1.1)


class TestAuthSchemeConstruction:
    def test_bearer_jwt_scheme(self) -> None:
        scheme = AuthScheme(
            name="BearerAuth",
            auth_type=AuthType.HTTP_BEARER,
            scheme="bearer",
            bearer_format="JWT",
            is_jwt=True,
        )
        assert scheme.is_jwt is True
        assert scheme.flows == []

    def test_api_key_scheme(self) -> None:
        scheme = AuthScheme(
            name="ApiKey",
            auth_type=AuthType.API_KEY,
            in_location="header",
        )
        assert scheme.in_location == "header"
        assert scheme.is_jwt is False

    def test_oauth2_scheme_with_flows(self, sample_oauth_flow: OAuthFlow) -> None:
        scheme = AuthScheme(
            name="OAuth2",
            auth_type=AuthType.OAUTH2,
            flows=[sample_oauth_flow],
        )
        assert len(scheme.flows) == 1
        assert scheme.flows[0].flow_type == "authorizationCode"


class TestOAuthFlowConstruction:
    def test_minimal_flow(self) -> None:
        flow = OAuthFlow(flow_type="clientCredentials", token_url="https://token.example.com/token")
        assert flow.flow_type == "clientCredentials"
        assert flow.scopes == {}
        assert flow.authorization_url is None
        assert flow.refresh_url is None

    def test_unknown_flow_type_accepted(self) -> None:
        """Non-standard flow types from vendor extensions must not raise."""
        flow = OAuthFlow(flow_type="x-custom-flow")
        assert flow.flow_type == "x-custom-flow"


class TestInferredResourceConstruction:
    def test_top_level_resource(self) -> None:
        r = InferredResource(
            name="User",
            path_prefix="/users",
            identifier_type=PathParamType.INTEGER,
        )
        assert r.parent_resource_name is None
        assert r.child_resource_names == []

    def test_child_resource_with_parent(self) -> None:
        r = InferredResource(
            name="Order",
            path_prefix="/users/{userId}/orders",
            parent_resource_name="User",
            identifier_type=PathParamType.UUID,
        )
        assert r.parent_resource_name == "User"


class TestParsedParameterConstruction:
    def test_sensitive_parameter_with_signals(self) -> None:
        param = ParsedParameter(
            name="password",
            location=ParameterLocation.QUERY,
            required=True,
            schema_type="string",
            is_sensitive=True,
            sensitivity_signals=["name_matches:password"],
        )
        assert param.is_sensitive is True
        assert "name_matches:password" in param.sensitivity_signals

    def test_url_accepting_parameter(self) -> None:
        param = ParsedParameter(
            name="callback",
            location=ParameterLocation.QUERY,
            schema_format="uri",
            accepts_url=True,
        )
        assert param.accepts_url is True
