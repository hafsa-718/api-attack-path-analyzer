"""Shared pytest fixtures for the api-attack-path-analyzer test suite.

All fixtures produce realistic, valid model instances that reflect the
pipeline's actual data shapes.  Tests that need a model with specific
field values should use the factory fixtures (``make_*``) which accept
keyword overrides, rather than mutating the plain fixtures.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from api_analyzer.models import (
    AnalysisResult,
    AttackStep,
    AuthScheme,
    AuthType,
    CandidateChain,
    ConfidenceBreakdown,
    EndpointFunction,
    HttpMethod,
    InferredResource,
    OAuthFlow,
    ParameterLocation,
    ParsedEndpoint,
    ParsedParameter,
    ParsedRequestBody,
    ParsedSchema,
    ParsedSpec,
    PathParamType,
    Severity,
    SensitivityClass,
    SpecFormat,
    ValidatedChain,
)

# ── Shared constants ──────────────────────────────────────────────────────────

SAMPLE_NARRATIVE = (
    "An attacker begins by calling the public user listing endpoint to enumerate "
    "integer user identifiers. These sequential IDs are then used to access "
    "individual user profile endpoints without any ownership validation, exposing "
    "personally identifiable information for arbitrary accounts. This chain "
    "represents a classic Broken Object Level Authorization vulnerability that "
    "enables mass data harvesting at scale with no authentication required."
)  # 436 chars — well above the 100-char minimum

SAMPLE_MITRE_IDS = ["T1589.001", "T1530"]

# ── Spec model fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def sample_oauth_flow() -> OAuthFlow:
    return OAuthFlow(
        flow_type="authorizationCode",
        authorization_url="https://auth.example.com/oauth/authorize",
        token_url="https://auth.example.com/oauth/token",
        scopes={"read:users": "Read user profiles", "write:users": "Modify user data"},
    )


@pytest.fixture
def sample_auth_scheme(sample_oauth_flow: OAuthFlow) -> AuthScheme:
    return AuthScheme(
        name="BearerAuth",
        auth_type=AuthType.HTTP_BEARER,
        scheme="bearer",
        bearer_format="JWT",
        is_jwt=True,
    )


@pytest.fixture
def sample_parameter() -> ParsedParameter:
    return ParsedParameter(
        name="userId",
        location=ParameterLocation.PATH,
        required=True,
        schema_type="integer",
        schema_format=None,
        is_identifier=True,
        is_sensitive=False,
    )


@pytest.fixture
def sample_schema() -> ParsedSchema:
    """A ParsedSchema with one level of nesting to exercise the recursive model."""
    nested = ParsedSchema(
        title="Address",
        schema_type="object",
        properties={
            "street": ParsedSchema(schema_type="string"),
            "postcode": ParsedSchema(schema_type="string"),
        },
        required_fields=["street"],
    )
    return ParsedSchema(
        title="UserProfile",
        schema_type="object",
        ref_name="UserProfile",
        properties={
            "id": ParsedSchema(schema_type="integer"),
            "email": ParsedSchema(schema_type="string", schema_format="email"),
            "address": nested,
        },
        required_fields=["id", "email"],
        has_pii_fields=True,
        pii_field_names=["email"],
    )


@pytest.fixture
def sample_request_body(sample_schema: ParsedSchema) -> ParsedRequestBody:
    return ParsedRequestBody(
        required=True,
        content_types=["application/json"],
        body_schema=sample_schema,
        has_pii_fields=True,
        has_role_fields=False,
    )


@pytest.fixture
def sample_endpoint(sample_parameter: ParsedParameter, sample_schema: ParsedSchema) -> ParsedEndpoint:
    return ParsedEndpoint(
        id="GET:/users/{userId}",
        path="/users/{userId}",
        method=HttpMethod.GET,
        operation_id="getUserById",
        summary="Get a user profile by ID",
        tags=["users"],
        parameters=[sample_parameter],
        auth_scheme_names=["BearerAuth"],
        is_public=False,
        sensitivity_class=SensitivityClass.SENSITIVE,
        inferred_function=EndpointFunction.DATA_READ,
        path_param_names=["userId"],
        path_param_type=PathParamType.INTEGER,
        returns_pii=True,
        response_schemas={"200": sample_schema},
    )


@pytest.fixture
def sample_public_endpoint(sample_schema: ParsedSchema) -> ParsedEndpoint:
    """A public (unauthenticated) collection endpoint — entry point for BOLA chains."""
    return ParsedEndpoint(
        id="GET:/users",
        path="/users",
        method=HttpMethod.GET,
        is_public=True,
        sensitivity_class=SensitivityClass.PUBLIC,
        inferred_function=EndpointFunction.DATA_READ,
        returns_collection=True,
        response_schemas={"200": sample_schema},
    )


@pytest.fixture
def sample_resource() -> InferredResource:
    return InferredResource(
        name="User",
        path_prefix="/users",
        collection_endpoint_id="GET:/users",
        detail_endpoint_id="GET:/users/{userId}",
        write_endpoint_ids=["POST:/users", "PUT:/users/{userId}"],
        identifier_name="userId",
        identifier_type=PathParamType.INTEGER,
    )


@pytest.fixture
def sample_spec(
    sample_endpoint: ParsedEndpoint,
    sample_public_endpoint: ParsedEndpoint,
    sample_auth_scheme: AuthScheme,
    sample_resource: InferredResource,
) -> ParsedSpec:
    return ParsedSpec(
        title="Test API",
        version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        endpoints=[sample_public_endpoint, sample_endpoint],
        auth_schemes={"BearerAuth": sample_auth_scheme},
        resources=[sample_resource],
        spec_completeness=0.75,
    )


# ── Chain model fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def sample_candidate_chain() -> CandidateChain:
    return CandidateChain(
        id=str(uuid4()),
        pattern_id="AP-001",
        pattern_name="BOLA via Predictable Integer Identifiers",
        owasp_category="API1:2023",
        mitre_hints=["T1589.001", "T1530"],
        confidence_base=0.70,
        endpoint_ids=["GET:/users", "GET:/users/{userId}"],
        hop_count=1,
        entry_endpoint_id="GET:/users",
        exit_endpoint_id="GET:/users/{userId}",
        crosses_auth_boundary=True,
        sensitivity_delta=2,
        rank_score=14.0,
        entry_summary="GET /users (PUBLIC)",
        exit_summary="GET /users/{userId} (SENSITIVE)",
    )


@pytest.fixture
def sample_attack_steps() -> list[AttackStep]:
    return [
        AttackStep(
            sequence=1,
            endpoint_id="GET:/users",
            path="/users",
            method="GET",
            auth_required="None — public endpoint",
            action="Call the user listing endpoint to retrieve a page of user objects",
            attacker_gains="Integer user IDs extracted from the id field of each returned object",
            technique="Unauthenticated resource enumeration via public collection endpoint",
        ),
        AttackStep(
            sequence=2,
            endpoint_id="GET:/users/{userId}",
            path="/users/{userId}",
            method="GET",
            auth_required="None — endpoint declares no security requirement",
            action="Iterate through harvested integer IDs to fetch individual user profiles",
            attacker_gains="Full user profile including email, address, and account details",
            technique="Broken Object Level Authorization via predictable integer path parameter",
        ),
    ]


@pytest.fixture
def sample_confidence() -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        graph_match_score=0.85,
        auth_clarity_score=0.78,
        llm_self_score=0.82,
        evidence_count=3,
        rationale=(
            "Graph confirmed integer path parameter on detail endpoint and public collection "
            "endpoint. Auth declarations are clear — detail endpoint has no security "
            "requirement. LLM validated the BOLA chain with 3 supporting tool calls."
        ),
    )


@pytest.fixture
def sample_validated_chain(
    sample_attack_steps: list[AttackStep],
    sample_confidence: ConfidenceBreakdown,
) -> ValidatedChain:
    return ValidatedChain(
        id=str(uuid4()),
        candidate_id=str(uuid4()),
        pattern_id="AP-001",
        name="BOLA → Mass PII Exfiltration via Integer User IDs",
        severity=Severity.HIGH,
        confidence=sample_confidence,
        steps=sample_attack_steps,
        narrative=SAMPLE_NARRATIVE,
        mitre_techniques=SAMPLE_MITRE_IDS,
        owasp_category="API1:2023 Broken Object Level Authorization",
        remediation=[
            "Implement ownership checks on GET /users/{userId}: verify the caller's "
            "token subject matches the requested userId.",
            "Add authentication requirement to GET /users/{userId} — the endpoint "
            "currently declares no security scheme.",
            "Consider returning opaque UUIDs instead of sequential integers to raise "
            "the cost of enumeration even if ownership checks are missing.",
        ],
        tool_calls_used=[
            "get_endpoint(GET:/users) → is_public=True, returns_collection=True",
            "get_endpoint(GET:/users/{userId}) → is_public=True, path_param_type=INTEGER, returns_pii=True",
            "query_graph(auth boundary check) → crosses_auth_boundary=True",
        ],
        analyzed_at=datetime.now(tz=timezone.utc),
        llm_model="claude-sonnet-4-6",
        tokens_used=1842,
    )


@pytest.fixture
def make_validated_chain(
    sample_attack_steps: list[AttackStep],
    sample_confidence: ConfidenceBreakdown,
):
    """Factory fixture for creating ValidatedChain with custom severity or fields.

    Usage::

        def test_something(make_validated_chain):
            chain = make_validated_chain(severity=Severity.CRITICAL)
    """

    def _make(severity: Severity = Severity.HIGH, **overrides: object) -> ValidatedChain:
        defaults: dict[str, object] = {
            "id": str(uuid4()),
            "candidate_id": str(uuid4()),
            "pattern_id": "AP-001",
            "name": "Test Chain",
            "severity": severity,
            "confidence": sample_confidence,
            "steps": sample_attack_steps,
            "narrative": SAMPLE_NARRATIVE,
            "mitre_techniques": list(SAMPLE_MITRE_IDS),
            "owasp_category": "API1:2023",
            "remediation": ["Implement ownership check on the affected endpoint."],
            "tool_calls_used": [],
            "analyzed_at": datetime.now(tz=timezone.utc),
            "llm_model": "claude-sonnet-4-6",
            "tokens_used": 1000,
        }
        defaults.update(overrides)
        return ValidatedChain(**defaults)

    return _make


# ── Report model fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_analysis_result(sample_validated_chain: ValidatedChain) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=str(uuid4()),
        spec_title="Test API",
        spec_version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        analyzed_at=datetime.now(tz=timezone.utc),
        duration_seconds=42.3,
        endpoint_count=20,
        public_endpoint_count=5,
        auth_declared=True,
        spec_completeness=0.75,
        chains=[sample_validated_chain],
        graph_node_count=48,
        graph_edge_count=62,
        patterns_run=["AP-001", "AP-002", "AP-003"],
        candidates_evaluated=8,
        candidates_rejected=7,
        llm_tokens_used=1842,
        estimated_cost_usd=0.012,
    )
