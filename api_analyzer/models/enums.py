"""Shared enumerations used across all modules.

All enums use StrEnum (Python 3.11+) so they serialise to their string values
in JSON, Neo4j property stores, and API responses without extra coercion.
StrEnum guarantees str(member) == member.value, which (str, Enum) stopped
guaranteeing in Python 3.12.

ORDERING CONTRACT — two enums have ordering semantics that downstream
modules depend on.  Do not reorder their members:

  SensitivityClass: PUBLIC(0) < INTERNAL(1) < SENSITIVE(2) < CRITICAL(3)
    The attack path ranker (M8) calls list(SensitivityClass).index(value)
    to derive a numeric weight for scoring the sensitivity delta across hops.

  Severity: CRITICAL(0) > HIGH(1) > MEDIUM(2) > LOW(3) > INFO(4)
    AnalysisResult.highest_severity uses list(Severity).index(value) to
    find the worst severity in a chain list; lower index = higher severity.
"""

from enum import StrEnum


class HttpMethod(StrEnum):
    """HTTP request methods (RFC 7231 + RFC 5789)."""

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class ParameterLocation(StrEnum):
    """Locations where an OpenAPI parameter may appear."""

    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"


class AuthType(StrEnum):
    """OpenAPI 3.x security scheme types.

    HTTP_BEARER and HTTP_BASIC are split from the generic 'http' type so
    downstream modules can treat bearer-token auth differently from basic auth
    without parsing the scheme sub-field.
    """

    API_KEY = "apiKey"
    HTTP_BEARER = "http_bearer"
    HTTP_BASIC = "http_basic"
    OAUTH2 = "oauth2"
    OPENID_CONNECT = "openIdConnect"


class SpecFormat(StrEnum):
    """Supported API specification formats."""

    OPENAPI3 = "openapi3"
    SWAGGER2 = "swagger2"


class SensitivityClass(StrEnum):
    """Endpoint sensitivity classification assigned by the classifier (M3).

    Declaration order encodes numeric severity weight 0–3.  Do not reorder.

      PUBLIC   (0) — no auth, no PII, read-only side effects (GET /health)
      INTERNAL (1) — auth required, or touches non-sensitive business data
      SENSITIVE(2) — returns PII or touches financial / user-owned data
      CRITICAL (3) — admin ops, bulk export, delete, auth management
    """

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    CRITICAL = "CRITICAL"


class EndpointFunction(StrEnum):
    """Inferred functional role of an API endpoint."""

    AUTH = "AUTH"             # login, token, password-reset, logout
    DATA_READ = "DATA_READ"   # GET returning resource data
    DATA_WRITE = "DATA_WRITE" # POST/PUT/PATCH creating or mutating resources
    ADMIN = "ADMIN"           # /admin/, /internal/, /management/ paths
    WEBHOOK = "WEBHOOK"       # callback URL registration endpoints
    UNKNOWN = "UNKNOWN"       # classification failed or ambiguous


class PathParamType(StrEnum):
    """Data type of path parameters — primary BOLA/IDOR risk signal.

    INTEGER is the highest-risk type: sequential integers are trivially
    enumerable.  UUID has lower but non-zero risk (some DBs generate
    predictable UUIDs).  STRING slugs have the lowest IDOR risk.
    """

    INTEGER = "INTEGER"
    UUID = "UUID"
    STRING = "STRING"
    NONE = "NONE"   # endpoint has no path parameters


class Severity(StrEnum):
    """Finding severity levels.

    Declaration order encodes severity rank 0–4.  Do not reorder.

      CRITICAL (0) — highest severity
      HIGH     (1)
      MEDIUM   (2)
      LOW      (3)
      INFO     (4) — lowest severity
    """

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
