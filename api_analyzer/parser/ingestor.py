"""OpenAPI / Swagger specification parser.

Public entry points
-------------------
``ingest(path)``          — load a file, resolve $refs, return ParsedSpec
``parse_spec_dict(raw)``  — parse a pre-resolved dict; used in tests

Pipeline position:
  CLI/API (M12/M13) → ingest(path) → ParsedSpec → classifier (M3) → graph builder (M5)

Supported formats
-----------------
OpenAPI 3.0.x and 3.1.x  (detected via root "openapi" key)
Swagger 2.0               (detected via root "swagger" key)

DoS protection
--------------
JSON Schema property nesting is bounded by MAX_SCHEMA_DEPTH (default 8).
At the limit, ParsedSchema is returned with max_depth_reached=True and empty
properties/items.  Circular $ref chains are caught by prance during resolution
before this module runs.

Out of scope for M2
-------------------
- Sensitivity classification (M3)
- Resource inference (M3)
- Graph construction (M5)
ParsedSpec.sensitivity_class, inferred_function, and resources are left at
their model defaults; M3 fills them in.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from api_analyzer.models.enums import (
    AuthType,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SpecFormat,
)
from api_analyzer.models.spec import (
    AuthScheme,
    OAuthFlow,
    ParsedEndpoint,
    ParsedParameter,
    ParsedRequestBody,
    ParsedSchema,
    ParsedSpec,
)

logger = logging.getLogger(__name__)

# Schema recursion depth limit — prevents DoS from deeply nested or circular schemas.
MAX_SCHEMA_DEPTH: int = 8

_PII_FIELD_NAMES: frozenset[str] = frozenset({
    "email", "emailaddress", "email_address",
    "phone", "phonenumber", "phone_number", "mobile", "mobilenumber",
    "ssn", "socialsecuritynumber", "social_security_number",
    "dob", "dateofbirth", "date_of_birth", "birthdate", "birthday",
    "password", "passwd", "pwd",
    "address", "streetaddress", "street_address",
    "creditcard", "credit_card", "cardnumber", "card_number", "ccnumber",
    "cvv", "cvc", "ccv", "cvc2",
    "passportno", "passport", "passportnumber", "passport_number",
    "nationalid", "national_id", "taxid", "tax_id", "tin",
    "firstname", "first_name", "lastname", "last_name", "fullname", "full_name",
})
_PII_FORMATS: frozenset[str] = frozenset({"email", "password"})
_URL_FORMATS: frozenset[str] = frozenset({"uri", "url", "iri", "iri-reference", "uri-reference"})
_URL_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"url|uri|webhook|callback|redirect|hook", re.IGNORECASE
)
# Matches names like: id, userId, user_id, order_id, orderId, upload_id
_IDENTIFIER_PATTERN: re.Pattern[str] = re.compile(
    r"(?:^|[_\-])id$|Id$", re.IGNORECASE
)
_HTTP_METHODS: frozenset[str] = frozenset(m.value.lower() for m in HttpMethod)
_ROLE_FIELD_NAMES: frozenset[str] = frozenset({
    "role", "roles", "permission", "permissions", "scope", "scopes",
    "privilege", "privileges", "admin", "isadmin", "is_admin",
    "superuser", "is_superuser", "group", "groups",
})


class SpecParseError(Exception):
    """Raised when a spec file cannot be loaded, resolved, or structurally parsed."""


# ── Public entry points ────────────────────────────────────────────────────────


def ingest(path: Path | str) -> ParsedSpec:
    """Load, resolve $refs, and parse an OpenAPI/Swagger spec file.

    :param path: Path to a .yaml, .yml, or .json spec file.
    :raises SpecParseError: If the file cannot be read, is not valid YAML/JSON,
        $refs cannot be resolved (circular or missing), or the spec has neither
        an "openapi" nor a "swagger" root key.
    """
    from api_analyzer.security.injection_guard import sanitise_spec  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise SpecParseError(f"Spec file not found: {path}")
    raw = _load_raw(path)
    resolved = _resolve_refs(raw, path)
    sanitised, injection_warnings = sanitise_spec(resolved)
    spec = parse_spec_dict(sanitised)
    if injection_warnings:
        spec = spec.model_copy(update={
            "parse_warnings": list(spec.parse_warnings) + injection_warnings
        })
    return spec


def parse_spec_dict(raw: dict[str, Any]) -> ParsedSpec:
    """Parse a pre-resolved spec dict to a ParsedSpec.

    All $ref values must already be inlined (i.e. prance has run or the
    spec has no $refs).  Used directly in unit tests to avoid file I/O
    and prance dependencies.

    :param raw: Fully resolved OpenAPI 3.x or Swagger 2.0 dict.
    :raises SpecParseError: If the format cannot be detected.
    """
    spec_format = _detect_format(raw)
    warnings: list[str] = []
    if spec_format == SpecFormat.OPENAPI3:
        return _parse_openapi3(raw, warnings)
    return _parse_swagger2(raw, warnings)


# ── File loading ───────────────────────────────────────────────────────────────


def _load_raw(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise SpecParseError(f"Cannot read spec file {path}: {e}") from e

    # Detect by content first (files are often mislabelled).
    # JSON objects always start with '{' (after optional BOM/whitespace).
    _stripped = text.lstrip("﻿ \t\r\n")
    _looks_json = _stripped.startswith("{")

    try:
        if _looks_json:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = yaml.safe_load(text)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            try:
                data = yaml.safe_load(text)
            except yaml.YAMLError:
                data = json.loads(text)
        elif path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise SpecParseError(f"Cannot parse {path} as YAML/JSON: {e}") from e

    if not isinstance(data, dict):
        raise SpecParseError(
            f"Spec root must be a JSON object, got {type(data).__name__!r}"
        )
    return data


def _resolve_refs(raw: dict[str, Any], path: Path) -> dict[str, Any]:
    """Inline all $ref values using prance.

    Falls back to the raw dict when prance cannot determine the spec version
    (e.g. some community specs skip the strict openapi/swagger version header
    that prance's validator requires).  Internal $refs remain unresolved in
    that case, but the parser handles missing schema properties gracefully.
    """
    try:
        import prance  # noqa: PLC0415 — deferred to keep prance optional in tests
        parser = prance.ResolvingParser(str(path), lazy=False, strict=False)
        return parser.specification  # type: ignore[return-value]
    except ImportError:
        raise SpecParseError(
            "prance is not installed. Run: pip install 'prance[cli]'"
        ) from None
    except Exception as e:
        msg = str(e)
        # prance's spec-validator can't detect version on some real-world specs —
        # skip $ref resolution and parse the raw dict directly.
        if "specification schema version" in msg or "Could not resolve" in msg:
            return raw
        raise SpecParseError(f"Failed to resolve $refs in {path}: {e}") from e


# ── Format detection ───────────────────────────────────────────────────────────


def _detect_format(raw: dict[str, Any]) -> SpecFormat:
    if "openapi" in raw:
        return SpecFormat.OPENAPI3
    if "swagger" in raw:
        return SpecFormat.SWAGGER2
    raise SpecParseError(
        "Cannot detect spec format: root object has neither 'openapi' nor 'swagger' key"
    )


# ── OpenAPI 3.x parsing ────────────────────────────────────────────────────────


def _parse_openapi3(raw: dict[str, Any], warnings: list[str]) -> ParsedSpec:
    info = raw.get("info", {})
    components = raw.get("components", {})
    auth_schemes = _parse_auth_schemes_v3(
        components.get("securitySchemes", {}), warnings
    )
    global_security: list[dict[str, Any]] = raw.get("security", [])
    endpoints = _parse_endpoints_v3(
        raw.get("paths", {}), global_security, auth_schemes, warnings
    )
    return ParsedSpec(
        title=str(info.get("title", "Unknown API")),
        version=str(info.get("version", "unknown")),
        description=info.get("description"),
        base_url=_extract_base_url_v3(raw),
        spec_format=SpecFormat.OPENAPI3,
        endpoints=endpoints,
        auth_schemes=auth_schemes,
        parse_warnings=warnings,
    )


def _extract_base_url_v3(raw: dict[str, Any]) -> str | None:
    servers = raw.get("servers", [])
    if isinstance(servers, list) and servers:
        first = servers[0]
        if isinstance(first, dict):
            url = first.get("url")
            return str(url) if isinstance(url, str) else None
    return None


def _parse_auth_schemes_v3(
    security_schemes: dict[str, Any], warnings: list[str]
) -> dict[str, AuthScheme]:
    result: dict[str, AuthScheme] = {}
    for name, d in security_schemes.items():
        if not isinstance(d, dict):
            continue
        scheme = _build_auth_scheme_v3(name, d, warnings)
        if scheme is not None:
            result[name] = scheme
    return result


def _build_auth_scheme_v3(
    name: str, d: dict[str, Any], warnings: list[str]
) -> AuthScheme | None:
    raw_type = str(d.get("type", "")).lower()

    if raw_type == "apikey":
        return AuthScheme(name=name, auth_type=AuthType.API_KEY, in_location=d.get("in"))

    if raw_type == "http":
        raw_scheme = str(d.get("scheme", "")).lower()
        bearer_format: str | None = d.get("bearerFormat")
        if raw_scheme == "bearer":
            is_jwt = (
                isinstance(bearer_format, str) and bearer_format.upper() == "JWT"
            ) or "jwt" in name.lower()
            return AuthScheme(
                name=name,
                auth_type=AuthType.HTTP_BEARER,
                scheme="bearer",
                bearer_format=bearer_format,
                is_jwt=is_jwt,
            )
        return AuthScheme(
            name=name, auth_type=AuthType.HTTP_BASIC, scheme=raw_scheme or "basic"
        )

    if raw_type == "oauth2":
        return AuthScheme(
            name=name,
            auth_type=AuthType.OAUTH2,
            flows=_parse_oauth_flows(d.get("flows", {})),
        )

    if raw_type == "openidconnect":
        return AuthScheme(name=name, auth_type=AuthType.OPENID_CONNECT)

    warnings.append(
        f"Unknown security scheme type {raw_type!r} for {name!r}; skipped"
    )
    return None


def _parse_oauth_flows(flows_dict: dict[str, Any]) -> list[OAuthFlow]:
    result: list[OAuthFlow] = []
    for flow_type, cfg in flows_dict.items():
        if not isinstance(cfg, dict):
            continue
        result.append(OAuthFlow(
            flow_type=flow_type,
            authorization_url=cfg.get("authorizationUrl"),
            token_url=cfg.get("tokenUrl"),
            refresh_url=cfg.get("refreshUrl"),
            scopes=cfg.get("scopes", {}),
        ))
    return result


def _parse_endpoints_v3(
    paths: dict[str, Any],
    global_security: list[dict[str, Any]],
    auth_schemes: dict[str, AuthScheme],
    warnings: list[str],
) -> list[ParsedEndpoint]:
    endpoints: list[ParsedEndpoint] = []
    for path_str, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        path_params = path_item.get("parameters", [])
        for method_str, operation in path_item.items():
            if method_str not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            try:
                endpoints.append(_parse_endpoint_v3(
                    path=path_str,
                    method=method_str.upper(),
                    op=operation,
                    path_level_params=path_params,
                    global_security=global_security,
                ))
            except Exception as exc:
                warnings.append(f"Skipped {method_str.upper()} {path_str}: {exc}")
    return endpoints


def _parse_endpoint_v3(
    path: str,
    method: str,
    op: dict[str, Any],
    path_level_params: list[Any],
    global_security: list[dict[str, Any]],
) -> ParsedEndpoint:
    all_params = _merge_parameters(path_level_params, op.get("parameters", []))
    parameters = [_parse_parameter(p) for p in all_params if isinstance(p, dict)]
    request_body = _parse_request_body_v3(op.get("requestBody"))
    response_schemas = _parse_response_schemas_v3(op.get("responses", {}))

    op_security = op.get("security")
    is_public = _is_public(op_security, global_security)
    auth_scheme_names = _auth_scheme_names(op_security, global_security)

    path_params = [p for p in parameters if p.location == ParameterLocation.PATH]

    return ParsedEndpoint(
        id=f"{method}:{path}",
        path=path,
        method=HttpMethod(method),
        operation_id=op.get("operationId"),
        summary=op.get("summary"),
        tags=op.get("tags", []),
        parameters=parameters,
        request_body=request_body,
        response_schemas=response_schemas,
        auth_scheme_names=auth_scheme_names,
        is_public=is_public,
        auth_declared=_auth_declared(op_security, global_security),
        path_param_names=[p.name for p in path_params],
        path_param_type=_infer_path_param_type(path_params),
        returns_pii=_schema_has_pii(response_schemas.get("200")),
        accepts_pii=bool(request_body and request_body.has_pii_fields),
        accepts_url_param=any(p.accepts_url for p in parameters),
        returns_collection=_schema_is_collection(response_schemas.get("200")),
        has_role_param=bool(request_body and request_body.has_role_fields),
    )


def _parse_response_schemas_v3(responses: dict[str, Any]) -> dict[str, ParsedSchema]:
    result: dict[str, ParsedSchema] = {}
    for status_code, resp in responses.items():
        if not isinstance(resp, dict):
            continue
        schema_dict = _schema_from_content(resp.get("content", {}))
        if schema_dict is not None:
            result[str(status_code)] = _parse_schema(schema_dict, depth=0)
    return result


def _schema_from_content(content: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON Schema dict from a content map; prefer application/json."""
    for mt in ("application/json", "application/json; charset=utf-8"):
        if mt in content and isinstance(content[mt], dict):
            return content[mt].get("schema")  # type: ignore[return-value]
    for media_obj in content.values():
        if isinstance(media_obj, dict) and "schema" in media_obj:
            return media_obj["schema"]  # type: ignore[return-value]
    return None


def _parse_request_body_v3(body: Any) -> ParsedRequestBody | None:
    if not isinstance(body, dict):
        return None
    content = body.get("content", {})
    schema_dict = _schema_from_content(content)
    parsed = _parse_schema(schema_dict, depth=0) if isinstance(schema_dict, dict) else None
    return ParsedRequestBody(
        required=bool(body.get("required", False)),
        content_types=list(content.keys()),
        body_schema=parsed,
        has_pii_fields=bool(parsed and parsed.has_pii_fields),
        has_role_fields=_dict_has_role_fields(schema_dict),
    )


# ── Schema parsing ─────────────────────────────────────────────────────────────


def _parse_schema(schema_dict: dict[str, Any] | None, depth: int) -> ParsedSchema:
    """Recursively parse a JSON Schema dict.

    Returns a truncated ParsedSchema with max_depth_reached=True when
    MAX_SCHEMA_DEPTH is exceeded rather than recursing further.
    """
    if not isinstance(schema_dict, dict):
        return ParsedSchema()
    if depth >= MAX_SCHEMA_DEPTH:
        return ParsedSchema(max_depth_reached=True)

    schema_type: str | None = schema_dict.get("type")
    schema_format: str | None = schema_dict.get("format")

    # Parse nested object properties
    properties: dict[str, ParsedSchema] = {}
    pii_names: list[str] = []
    has_pii = _format_is_pii(schema_format)

    raw_props = schema_dict.get("properties", {})
    if isinstance(raw_props, dict):
        for prop_name, prop_schema in raw_props.items():
            child = _parse_schema(prop_schema, depth + 1)
            properties[prop_name] = child
            if _field_name_is_pii(prop_name) or _format_is_pii(
                prop_schema.get("format") if isinstance(prop_schema, dict) else None
            ):
                pii_names.append(prop_name)
                has_pii = True
            if child.has_pii_fields:
                has_pii = True
                pii_names.extend(child.pii_field_names)

    # Parse array items
    items_schema: ParsedSchema | None = None
    is_collection = False
    raw_items = schema_dict.get("items")
    if schema_type == "array" and isinstance(raw_items, dict):
        items_schema = _parse_schema(raw_items, depth + 1)
        is_collection = True
        if items_schema.has_pii_fields:
            has_pii = True
            pii_names.extend(items_schema.pii_field_names)

    return ParsedSchema(
        title=schema_dict.get("title"),
        schema_type=schema_type,
        schema_format=schema_format,
        properties=properties,
        required_fields=schema_dict.get("required", []),
        items=items_schema,
        ref_name=schema_dict.get("x-ref-name"),
        has_pii_fields=has_pii,
        pii_field_names=pii_names,
        is_collection=is_collection,
    )


def _field_name_is_pii(name: str) -> bool:
    return name.lower().replace("-", "_") in _PII_FIELD_NAMES


def _format_is_pii(fmt: str | None) -> bool:
    return isinstance(fmt, str) and fmt.lower() in _PII_FORMATS


def _schema_has_pii(schema: ParsedSchema | None) -> bool:
    return schema is not None and schema.has_pii_fields


def _schema_is_collection(schema: ParsedSchema | None) -> bool:
    return schema is not None and schema.is_collection


def _dict_has_role_fields(schema_dict: dict[str, Any] | None) -> bool:
    if not isinstance(schema_dict, dict):
        return False
    props = schema_dict.get("properties", {})
    return isinstance(props, dict) and any(k.lower() in _ROLE_FIELD_NAMES for k in props)


# ── Parameter parsing ──────────────────────────────────────────────────────────


def _parse_parameter(param_dict: dict[str, Any]) -> ParsedParameter:
    name = str(param_dict.get("name", ""))
    in_str = str(param_dict.get("in", "")).lower()
    schema = param_dict.get("schema", {})
    if not isinstance(schema, dict):
        schema = {}
    schema_type: str | None = schema.get("type")
    schema_format: str | None = schema.get("format")

    is_sensitive, signals = _param_sensitivity(name, schema_format)

    return ParsedParameter(
        name=name,
        location=_param_location(in_str),
        required=bool(param_dict.get("required", in_str == "path")),
        schema_type=schema_type,
        schema_format=schema_format,
        is_identifier=_param_is_identifier(name),
        is_sensitive=is_sensitive,
        sensitivity_signals=signals,
        accepts_url=_param_accepts_url(name, schema_format),
    )


def _param_location(in_str: str) -> ParameterLocation:
    return {
        "path": ParameterLocation.PATH,
        "query": ParameterLocation.QUERY,
        "header": ParameterLocation.HEADER,
        "cookie": ParameterLocation.COOKIE,
    }.get(in_str, ParameterLocation.QUERY)


def _param_is_identifier(name: str) -> bool:
    return name.lower() == "id" or bool(_IDENTIFIER_PATTERN.search(name))


def _param_sensitivity(name: str, fmt: str | None) -> tuple[bool, list[str]]:
    signals: list[str] = []
    if name.lower().replace("-", "_") in _PII_FIELD_NAMES:
        signals.append(f"name_matches:{name}")
    if fmt and fmt.lower() in _PII_FORMATS:
        signals.append(f"schema_format:{fmt}")
    return bool(signals), signals


def _param_accepts_url(name: str, fmt: str | None) -> bool:
    if fmt and fmt.lower() in _URL_FORMATS:
        return True
    return bool(_URL_NAME_PATTERN.search(name))


def _infer_path_param_type(path_params: list[ParsedParameter]) -> PathParamType:
    """Classify the deepest path parameter's type — highest IDOR risk signal."""
    if not path_params:
        return PathParamType.NONE
    last = path_params[-1]
    if last.schema_type in {"integer", "number"}:
        return PathParamType.INTEGER
    if last.schema_format and "uuid" in last.schema_format.lower():
        return PathParamType.UUID
    if last.schema_type == "string":
        return PathParamType.STRING
    return PathParamType.NONE


# ── Parameter merge ────────────────────────────────────────────────────────────


def _merge_parameters(
    path_params: list[Any], op_params: list[Any]
) -> list[Any]:
    """Operation-level parameters override path-level ones by name+in key."""
    merged: dict[tuple[str, str], Any] = {}
    for p in path_params:
        if isinstance(p, dict) and "name" in p and "in" in p:
            merged[(p["name"], p["in"])] = p
    for p in op_params:
        if isinstance(p, dict) and "name" in p and "in" in p:
            merged[(p["name"], p["in"])] = p
    return list(merged.values())


# ── Security helpers ───────────────────────────────────────────────────────────


def _is_public(op_security: Any, global_security: list[dict[str, Any]]) -> bool:
    """True when the effective security requirement list is empty.

    op_security=[]  (empty list)     → explicitly public, overrides global
    op_security=None                 → fall back to global_security
    op_security=[{...}]              → secured
    global_security=[]               → public by default
    """
    effective = op_security if op_security is not None else global_security
    if not isinstance(effective, list):
        return False  # malformed — treat as secured
    return len(effective) == 0


def _auth_declared(op_security: Any, global_security: list[dict[str, Any]]) -> bool:
    """True when security is explicitly stated — op-level key exists OR global is non-empty.

    False only when op_security is None AND global_security is empty: the spec
    gives zero signal about auth requirements for this endpoint (spec gap).
    """
    if op_security is not None:
        return True
    return len(global_security) > 0


def _auth_scheme_names(
    op_security: Any, global_security: list[dict[str, Any]]
) -> list[str]:
    effective = op_security if op_security is not None else global_security
    if not isinstance(effective, list):
        return []
    return [name for req in effective if isinstance(req, dict) for name in req]


# ── Swagger 2.0 parsing ────────────────────────────────────────────────────────


def _parse_swagger2(raw: dict[str, Any], warnings: list[str]) -> ParsedSpec:
    info = raw.get("info", {})
    auth_schemes = _parse_auth_schemes_v2(raw.get("securityDefinitions", {}), warnings)
    global_security: list[dict[str, Any]] = raw.get("security", [])
    global_consumes: list[str] = raw.get("consumes", ["application/json"])
    endpoints = _parse_endpoints_v2(
        raw.get("paths", {}), global_security, global_consumes, warnings
    )
    return ParsedSpec(
        title=str(info.get("title", "Unknown API")),
        version=str(info.get("version", "unknown")),
        description=info.get("description"),
        base_url=_extract_base_url_v2(raw),
        spec_format=SpecFormat.SWAGGER2,
        endpoints=endpoints,
        auth_schemes=auth_schemes,
        parse_warnings=warnings,
    )


def _extract_base_url_v2(raw: dict[str, Any]) -> str | None:
    host = raw.get("host")
    if not host:
        return None
    base_path = raw.get("basePath", "/")
    schemes = raw.get("schemes", ["https"])
    scheme = schemes[0] if isinstance(schemes, list) and schemes else "https"
    return f"{scheme}://{host}{base_path}"


def _parse_auth_schemes_v2(
    security_defs: dict[str, Any], warnings: list[str]
) -> dict[str, AuthScheme]:
    result: dict[str, AuthScheme] = {}
    for name, d in security_defs.items():
        if not isinstance(d, dict):
            continue
        scheme = _build_auth_scheme_v2(name, d, warnings)
        if scheme is not None:
            result[name] = scheme
    return result


def _build_auth_scheme_v2(
    name: str, d: dict[str, Any], warnings: list[str]
) -> AuthScheme | None:
    raw_type = str(d.get("type", "")).lower()
    if raw_type == "apikey":
        return AuthScheme(name=name, auth_type=AuthType.API_KEY, in_location=d.get("in"))
    if raw_type == "basic":
        return AuthScheme(name=name, auth_type=AuthType.HTTP_BASIC, scheme="basic")
    if raw_type == "oauth2":
        flow_type = d.get("flow", "implicit")
        return AuthScheme(
            name=name,
            auth_type=AuthType.OAUTH2,
            flows=[OAuthFlow(
                flow_type=flow_type,
                authorization_url=d.get("authorizationUrl"),
                token_url=d.get("tokenUrl"),
                scopes=d.get("scopes", {}),
            )],
        )
    warnings.append(f"Unknown Swagger 2.0 security type {raw_type!r} for {name!r}; skipped")
    return None


def _parse_endpoints_v2(
    paths: dict[str, Any],
    global_security: list[dict[str, Any]],
    global_consumes: list[str],
    warnings: list[str],
) -> list[ParsedEndpoint]:
    endpoints: list[ParsedEndpoint] = []
    for path_str, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        path_params = path_item.get("parameters", [])
        for method_str, operation in path_item.items():
            if method_str not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            try:
                endpoints.append(_parse_endpoint_v2(
                    path=path_str,
                    method=method_str.upper(),
                    op=operation,
                    path_level_params=path_params,
                    global_security=global_security,
                    op_consumes=operation.get("consumes", global_consumes),
                ))
            except Exception as exc:
                warnings.append(f"Skipped {method_str.upper()} {path_str}: {exc}")
    return endpoints


def _parse_endpoint_v2(
    path: str,
    method: str,
    op: dict[str, Any],
    path_level_params: list[Any],
    global_security: list[dict[str, Any]],
    op_consumes: list[str],
) -> ParsedEndpoint:
    all_params = _merge_parameters(path_level_params, op.get("parameters", []))
    body_params = [p for p in all_params if isinstance(p, dict) and p.get("in") == "body"]
    non_body = [p for p in all_params if not (isinstance(p, dict) and p.get("in") == "body")]

    parameters = [_parse_parameter(p) for p in non_body if isinstance(p, dict)]
    request_body = _parse_body_param_v2(body_params, op_consumes)
    response_schemas = _parse_response_schemas_v2(op.get("responses", {}))

    op_security = op.get("security")
    path_params = [p for p in parameters if p.location == ParameterLocation.PATH]

    return ParsedEndpoint(
        id=f"{method}:{path}",
        path=path,
        method=HttpMethod(method),
        operation_id=op.get("operationId"),
        summary=op.get("summary"),
        tags=op.get("tags", []),
        parameters=parameters,
        request_body=request_body,
        response_schemas=response_schemas,
        auth_scheme_names=_auth_scheme_names(op_security, global_security),
        is_public=_is_public(op_security, global_security),
        auth_declared=_auth_declared(op_security, global_security),
        path_param_names=[p.name for p in path_params],
        path_param_type=_infer_path_param_type(path_params),
        returns_pii=_schema_has_pii(response_schemas.get("200")),
        accepts_pii=bool(request_body and request_body.has_pii_fields),
        accepts_url_param=any(p.accepts_url for p in parameters),
        returns_collection=_schema_is_collection(response_schemas.get("200")),
        has_role_param=bool(request_body and request_body.has_role_fields),
    )


def _parse_body_param_v2(
    body_params: list[Any], consumes: list[str]
) -> ParsedRequestBody | None:
    # Swagger 2.0 allows at most one body parameter per operation
    if not body_params or not isinstance(body_params[0], dict):
        return None
    param = body_params[0]
    schema_dict = param.get("schema")
    parsed = _parse_schema(schema_dict, depth=0) if isinstance(schema_dict, dict) else None
    return ParsedRequestBody(
        required=bool(param.get("required", False)),
        content_types=list(consumes),
        body_schema=parsed,
        has_pii_fields=bool(parsed and parsed.has_pii_fields),
        has_role_fields=_dict_has_role_fields(schema_dict),
    )


def _parse_response_schemas_v2(responses: dict[str, Any]) -> dict[str, ParsedSchema]:
    result: dict[str, ParsedSchema] = {}
    for status_code, resp in responses.items():
        if not isinstance(resp, dict):
            continue
        schema_dict = resp.get("schema")
        if isinstance(schema_dict, dict):
            result[str(status_code)] = _parse_schema(schema_dict, depth=0)
    return result
