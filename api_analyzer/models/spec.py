"""Data models for parsed OpenAPI/Swagger specifications.

These models are the output contract between the parser module (M2 + M3) and
the graph builder (M5).  They represent every security-relevant aspect of an
API specification with all $ref references resolved and sensitivity signals
pre-computed so downstream modules never need to re-inspect raw schema text.

Module dependency: imports only from api_analyzer.models.enums.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from api_analyzer.models.enums import (
    AuthType,
    EndpointFunction,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SensitivityClass,
    SpecFormat,
)


class OAuthFlow(BaseModel):
    """One OAuth 2.0 grant type declared in a security scheme's ``flows`` map.

    ``flow_type`` is stored as ``str`` rather than an enum because some vendors
    declare non-standard flow types in ``x-`` extensions.  The parser logs a
    warning for unknown values but does not reject them.
    """

    model_config = ConfigDict(frozen=False, str_strip_whitespace=True)

    flow_type: str = Field(
        description=(
            "OAuth 2.0 grant type: implicit | password | "
            "clientCredentials | authorizationCode"
        )
    )
    authorization_url: str | None = Field(
        default=None,
        description="Present for implicit and authorizationCode flows",
    )
    token_url: str | None = Field(
        default=None,
        description="Present for password, clientCredentials, and authorizationCode flows",
    )
    refresh_url: str | None = Field(default=None)
    scopes: dict[str, str] = Field(
        default_factory=dict,
        description="Map of scope name → human-readable description",
    )


class AuthScheme(BaseModel):
    """One entry from the spec's ``securitySchemes`` map.

    The ``name`` field is the key used in endpoint ``security`` arrays.
    ``is_jwt`` is set by the classifier (M3) using bearer_format heuristics
    and scheme name patterns — it signals JWT-specific attack patterns to the
    graph traversal engine and LLM reasoning agent.
    """

    model_config = ConfigDict(frozen=False, str_strip_whitespace=True)

    name: str = Field(description="Key from the securitySchemes map")
    auth_type: AuthType
    in_location: str | None = Field(
        default=None,
        description="For apiKey schemes: 'header' | 'query' | 'cookie'",
    )
    scheme: str | None = Field(
        default=None,
        description="For HTTP schemes: 'bearer' | 'basic' | 'digest'",
    )
    bearer_format: str | None = Field(
        default=None,
        description="Hint for bearer token format, e.g. 'JWT'",
    )
    flows: list[OAuthFlow] = Field(
        default_factory=list,
        description="OAuth 2.0 flows; empty list for non-OAuth scheme types",
    )
    is_jwt: bool = Field(
        default=False,
        description=(
            "True if bearer_format indicates JWT or scheme name heuristics suggest JWT. "
            "Set by classifier (M3), not derived from the spec field alone."
        ),
    )


class ParsedParameter(BaseModel):
    """One parameter from an endpoint's parameters array or request body schema.

    ``sensitivity_signals`` preserves the classification evidence so the LLM
    reasoning agent can explain *why* a parameter is sensitive in the chain
    narrative, not just assert that it is.  Format: 'rule:matched_value',
    e.g. ``'name_matches:password'``, ``'schema_format:email'``.
    """

    model_config = ConfigDict(frozen=False, str_strip_whitespace=True)

    name: str
    location: ParameterLocation
    required: bool = False
    schema_type: str | None = Field(
        default=None,
        description="JSON Schema primitive type: string | integer | number | boolean | array | object",
    )
    schema_format: str | None = Field(
        default=None,
        description="JSON Schema format hint: uuid | email | uri | date-time | etc.",
    )
    is_identifier: bool = Field(
        default=False,
        description="True if name matches ID patterns: *Id, *_id, *ID, or bare 'id'",
    )
    is_sensitive: bool = Field(
        default=False,
        description="True if name or format matches PII or credential patterns",
    )
    sensitivity_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence that triggered the sensitive classification. "
            "Used in LLM narrative generation for transparent reasoning."
        ),
    )
    accepts_url: bool = Field(
        default=False,
        description=(
            "True if schema_format='uri' or name matches URL/callback/webhook/redirect patterns. "
            "Primary signal for SSRF pattern detection (AP-006)."
        ),
    )


class ParsedSchema(BaseModel):
    """A JSON Schema object, recursively composed.

    Supports schemas up to ``MAX_SCHEMA_DEPTH`` levels deep (enforced by the
    parser, M2).  When the depth limit is reached, ``max_depth_reached`` is set
    True and ``properties``/``items`` are empty — downstream modules must not
    over-claim about fields they did not see.

    CRITICAL IMPLEMENTATION NOTE
    ----------------------------
    This model is self-referential.  ``ParsedSchema.model_rebuild()`` **must**
    be called immediately after this class definition.  Omitting it causes a
    ``PydanticUserError`` at import time in Python 3.11 and silent validation
    failures in some 3.12 builds.
    """

    model_config = ConfigDict(frozen=False)

    title: str | None = None
    schema_type: str | None = Field(
        default=None,
        description="JSON Schema type: object | array | string | integer | number | boolean",
    )
    schema_format: str | None = None
    properties: dict[str, "ParsedSchema"] = Field(default_factory=dict)
    required_fields: list[str] = Field(default_factory=list)
    items: Optional["ParsedSchema"] = Field(
        default=None,
        description="Schema for array item type; None for non-array schemas",
    )
    ref_name: str | None = Field(
        default=None,
        description="Original $ref component name before resolution, e.g. 'User'",
    )
    has_pii_fields: bool = False
    has_sensitive_fields: bool = False
    pii_field_names: list[str] = Field(
        default_factory=list,
        description="Property names that triggered PII classification",
    )
    sensitive_field_names: list[str] = Field(
        default_factory=list,
        description="Property names that triggered sensitivity classification",
    )
    is_collection: bool = Field(
        default=False,
        description="True if schema_type='array' with a non-None items schema",
    )
    max_depth_reached: bool = Field(
        default=False,
        description=(
            "True if this schema was truncated at the parser's recursion depth limit. "
            "Downstream modules must not assert complete field coverage when True."
        ),
    )


# Self-referential models require an explicit rebuild call in Pydantic v2.
# This resolves the forward references in `properties` and `items` fields.
ParsedSchema.model_rebuild()


class ParsedRequestBody(BaseModel):
    """The request body of an endpoint with schema extracted and pre-classified.

    ``has_pii_fields`` and ``has_role_fields`` are propagated from the schema
    to avoid requiring downstream consumers to traverse nested schema trees.
    """

    model_config = ConfigDict(frozen=False)

    required: bool = False
    content_types: list[str] = Field(
        default_factory=list,
        description="MIME types: ['application/json', 'multipart/form-data', ...]",
    )
    body_schema: ParsedSchema | None = None
    has_pii_fields: bool = Field(
        default=False,
        description="Propagated from schema.has_pii_fields for O(1) access",
    )
    has_role_fields: bool = Field(
        default=False,
        description=(
            "True if schema contains role / permission / admin field names. "
            "Primary signal for privilege escalation pattern (AP-003)."
        ),
    )


class InferredResource(BaseModel):
    """A business resource inferred from URL path structure.

    Path ``/users/{userId}/orders/{orderId}`` produces two resources:
    ``User`` (top-level) and ``Order`` (child of User).

    The combination ``identifier_type=INTEGER`` + ``collection_endpoint_id``
    set + ``detail_endpoint_id`` set is the primary BOLA detection signal at
    resource level and drives the AP-001 Cypher query in M7.
    """

    model_config = ConfigDict(frozen=False)

    name: str = Field(description="Title-cased resource name, e.g. 'User', 'Order'")
    path_prefix: str = Field(description="URL path prefix, e.g. '/users'")
    parent_resource_name: str | None = Field(
        default=None,
        description="Name of parent resource; None for top-level resources",
    )
    child_resource_names: list[str] = Field(default_factory=list)
    collection_endpoint_id: str | None = Field(
        default=None,
        description="ID of the endpoint returning a collection, e.g. 'GET:/users'",
    )
    detail_endpoint_id: str | None = Field(
        default=None,
        description="ID of the single-resource endpoint, e.g. 'GET:/users/{userId}'",
    )
    write_endpoint_ids: list[str] = Field(
        default_factory=list,
        description="IDs of POST/PUT/PATCH endpoints for this resource",
    )
    delete_endpoint_id: str | None = None
    identifier_name: str | None = Field(
        default=None,
        description="Path parameter name identifying this resource, e.g. 'userId'",
    )
    identifier_type: PathParamType = Field(
        default=PathParamType.NONE,
        description="Data type of the resource identifier — INTEGER is highest BOLA risk",
    )


class ParsedEndpoint(BaseModel):
    """One API operation extracted and classified from an OpenAPI specification.

    The central entity of the spec model.  All security analysis traces back
    to endpoints.  The parser (M2) populates structural fields; the classifier
    (M3) sets ``sensitivity_class`` and ``inferred_function``.

    ID format
    ---------
    ``'{METHOD}:{path}'``, e.g. ``'GET:/users/{userId}'``.
    This format is stable across parse runs and used as the Neo4j node ID.
    The validator enforces the format at construction time to catch bugs early.
    """

    model_config = ConfigDict(frozen=False)

    id: str = Field(
        description="Stable unique identifier: 'METHOD:path', e.g. 'GET:/users/{userId}'"
    )
    path: str
    method: HttpMethod
    operation_id: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    parameters: list[ParsedParameter] = Field(default_factory=list)
    request_body: ParsedRequestBody | None = None
    response_schemas: dict[str, ParsedSchema] = Field(
        default_factory=dict,
        description="HTTP status code → response schema, e.g. {'200': ParsedSchema(...)}",
    )
    auth_scheme_names: list[str] = Field(
        default_factory=list,
        description="Names of security schemes required, resolved from securitySchemes map",
    )
    is_public: bool = Field(
        description=(
            "True if this endpoint declares no security requirement: either "
            "'security: []' at operation level or no security at any level."
        ),
    )
    auth_declared: bool = Field(
        default=False,
        description=(
            "True when the effective security requirement is explicitly stated — "
            "either via an operation-level 'security:' key or inherited from a "
            "non-empty global 'security:' array at the spec root. "
            "False when neither the operation nor the spec root declares any security: "
            "auth status is UNKNOWN, not confirmed public. "
            "A chain step with auth_declared=False and is_public=True is a SPEC GAP, "
            "not a confirmed vulnerability."
        ),
    )
    sensitivity_class: SensitivityClass = Field(
        default=SensitivityClass.PUBLIC,
        description="Set by classifier (M3). Defaults to PUBLIC until classifier runs.",
    )
    inferred_function: EndpointFunction = Field(
        default=EndpointFunction.UNKNOWN,
        description="Set by classifier (M3).",
    )
    path_param_names: list[str] = Field(
        default_factory=list,
        description="Names of all path parameters, e.g. ['userId', 'orderId']",
    )
    path_param_type: PathParamType = Field(
        default=PathParamType.NONE,
        description=(
            "Data type of the deepest path parameter. "
            "The deepest param identifies the target resource for IDOR analysis."
        ),
    )
    returns_pii: bool = Field(
        default=False,
        description="True if any response schema contains PII-classified fields",
    )
    accepts_pii: bool = Field(
        default=False,
        description="True if the request body contains PII-classified fields",
    )
    accepts_url_param: bool = Field(
        default=False,
        description="True if any parameter accepts URL values — primary SSRF signal",
    )
    returns_collection: bool = Field(
        default=False,
        description="True if the 200-response schema is an array type",
    )
    has_role_param: bool = Field(
        default=False,
        description=(
            "True if the request body contains role/permission/admin field names. "
            "Primary signal for privilege escalation pattern AP-003."
        ),
    )
    resource_name: str | None = Field(
        default=None,
        description="Name of the InferredResource this endpoint operates on",
    )

    @field_validator("id")
    @classmethod
    def validate_id_format(cls, v: str) -> str:
        """Enforce 'METHOD:path' format to catch construction mistakes early.

        A malformed ID would silently produce broken Neo4j node IDs and
        unresolvable cross-references between the parser and graph builder.
        """
        if ":" not in v:
            raise ValueError(
                f"Endpoint id must use 'METHOD:path' format, got: {v!r}. "
                "Example: 'GET:/users/{userId}'"
            )
        _method, path_part = v.split(":", 1)
        if not path_part.startswith("/"):
            raise ValueError(
                f"Path component of endpoint id must start with '/', got: {path_part!r}"
            )
        return v


class ParsedSpec(BaseModel):
    """Top-level output of the specification parser.

    The single object passed from the parser (M2) to the graph builder (M5).
    Contains the complete security-relevant representation of an API
    specification with all $ref references resolved.

    Computed fields
    ---------------
    ``endpoint_count``, ``public_endpoint_count``, and ``auth_declared`` are
    derived automatically from ``endpoints`` and ``auth_schemes`` and included
    in serialisation output via ``@computed_field``.

    ``spec_completeness`` is a float set by the classifier (M3) based on:
    fraction of endpoints with descriptions, fraction with explicit security
    declarations, and fraction of parameters with type information.  It feeds
    directly into the confidence breakdown for every chain this spec produces.
    """

    model_config = ConfigDict(frozen=False)

    title: str = Field(description="API title from spec info.title")
    version: str = Field(description="API version from spec info.version")
    description: str | None = None
    base_url: str | None = Field(
        default=None,
        description="First server URL from spec servers list, if present",
    )
    spec_format: SpecFormat
    endpoints: list[ParsedEndpoint] = Field(default_factory=list)
    auth_schemes: dict[str, AuthScheme] = Field(
        default_factory=dict,
        description="Map of scheme name → AuthScheme, from spec securitySchemes",
    )
    resources: list[InferredResource] = Field(
        default_factory=list,
        description="Business resources inferred from URL path nesting structure",
    )
    spec_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Completeness score 0.0–1.0, set by classifier (M3). "
            "Feeds into ConfidenceBreakdown.auth_clarity_score for every chain."
        ),
    )
    parse_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered during parsing, e.g. missing descriptions",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def endpoint_count(self) -> int:
        """Total number of parsed API endpoints."""
        return len(self.endpoints)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def public_endpoint_count(self) -> int:
        """Number of endpoints that require no authentication."""
        return sum(1 for e in self.endpoints if e.is_public)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def auth_declared(self) -> bool:
        """True if at least one security scheme is declared in the spec."""
        return len(self.auth_schemes) > 0
