"""Public interface for the api_analyzer.models package.

All models, enums, and type aliases that cross module boundaries are
re-exported here.  Consumers import from this package, never from submodules:

    from api_analyzer.models import ParsedEndpoint, ValidatedChain, Severity

Import order below mirrors the dependency chain:
  enums (no deps) → spec (enums) → chain (enums) → report (chain + enums)
"""

from api_analyzer.models.enums import (
    AuthType,
    EndpointFunction,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    Severity,
    SensitivityClass,
    SpecFormat,
)
from api_analyzer.models.spec import (
    AuthScheme,
    InferredResource,
    OAuthFlow,
    ParsedEndpoint,
    ParsedParameter,
    ParsedRequestBody,
    ParsedSchema,
    ParsedSpec,
)
from api_analyzer.models.chain import (
    AttackStep,
    CandidateChain,
    ConfidenceBreakdown,
    ValidatedChain,
)
from api_analyzer.models.report import (
    AnalysisResult,
    ChainSummary,
    ReportContext,
)

__all__ = [
    # Enums
    "AuthType",
    "EndpointFunction",
    "HttpMethod",
    "ParameterLocation",
    "PathParamType",
    "Severity",
    "SensitivityClass",
    "SpecFormat",
    # Spec models
    "AuthScheme",
    "InferredResource",
    "OAuthFlow",
    "ParsedEndpoint",
    "ParsedParameter",
    "ParsedRequestBody",
    "ParsedSchema",
    "ParsedSpec",
    # Chain models
    "AttackStep",
    "CandidateChain",
    "ConfidenceBreakdown",
    "ValidatedChain",
    # Report models
    "AnalysisResult",
    "ChainSummary",
    "ReportContext",
]
