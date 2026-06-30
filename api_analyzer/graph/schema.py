"""Neo4j graph schema for the API Attack Path Analyzer.

Centralises every string constant used across the graph layer so a rename
is a one-file change.  Also owns the DDL statements that create constraints
and indexes, and the two lifecycle helpers (apply_schema / wipe_spec).

Node model
----------
  :ApiSpec    — root node, one per analysed specification
  :Endpoint   — one per API operation (ParsedEndpoint)
  :Resource   — one per inferred REST resource (InferredResource)
  :AuthScheme — one per security scheme declared in the spec

Every node except :ApiSpec carries a ``spec_id`` property that ties it to
its parent :ApiSpec.  :ApiSpec stores spec_id in its own ``id`` field,
making the wipe query trivially ``MATCH (n) WHERE n.spec_id = $id DETACH DELETE n``.

Relationship model
------------------
  (:Endpoint)-[:PART_OF]->(:ApiSpec)
  (:Endpoint)-[:REQUIRES_AUTH]->(:AuthScheme)
  (:Endpoint)-[:LISTS]->(:Resource)      collection GET  → resource
  (:Endpoint)-[:READS]->(:Resource)      detail GET      → resource
  (:Endpoint)-[:WRITES]->(:Resource)     POST/PUT/PATCH  → resource
  (:Endpoint)-[:DELETES]->(:Resource)    DELETE          → resource
  (:Resource)-[:CHILD_OF]->(:Resource)   sub-resource hierarchy

Attack chain edges (added by M9 traversal, not the schema itself):
  (:Endpoint)-[:CAN_REACH {pattern, reason, risk_score}]->(:Endpoint)

Schema application
------------------
``apply_schema(driver)`` is idempotent — it uses ``IF NOT EXISTS`` on every
statement.  Call it once at CLI startup before any graph writes.
"""

from __future__ import annotations

import re

from neo4j import Driver

# ── Node labels ────────────────────────────────────────────────────────────────

LABEL_API_SPEC: str = "ApiSpec"
LABEL_ENDPOINT: str = "Endpoint"
LABEL_RESOURCE: str = "Resource"
LABEL_AUTH_SCHEME: str = "AuthScheme"

# ── Relationship types ─────────────────────────────────────────────────────────

REL_PART_OF: str = "PART_OF"           # Endpoint → ApiSpec
REL_REQUIRES_AUTH: str = "REQUIRES_AUTH"  # Endpoint → AuthScheme
REL_LISTS: str = "LISTS"               # Endpoint → Resource  (collection GET)
REL_READS: str = "READS"               # Endpoint → Resource  (detail GET)
REL_WRITES: str = "WRITES"             # Endpoint → Resource  (POST/PUT/PATCH)
REL_DELETES: str = "DELETES"           # Endpoint → Resource  (DELETE)
REL_CHILD_OF: str = "CHILD_OF"         # Resource → Resource  (sub-resource)

# Attack chain relationship added by M9 (not constrained here):
REL_CAN_REACH: str = "CAN_REACH"       # Endpoint → Endpoint  (attack hop)

# ── Property keys ──────────────────────────────────────────────────────────────

# Shared
PROP_ID: str = "id"
PROP_SPEC_ID: str = "spec_id"
PROP_NAME: str = "name"

# :ApiSpec
PROP_TITLE: str = "title"
PROP_VERSION: str = "version"
PROP_SPEC_FORMAT: str = "spec_format"
PROP_SPEC_COMPLETENESS: str = "spec_completeness"

# :Endpoint
PROP_PATH: str = "path"
PROP_METHOD: str = "method"
PROP_SUMMARY: str = "summary"
PROP_IS_PUBLIC: str = "is_public"
PROP_AUTH_DECLARED: str = "auth_declared"
PROP_SENSITIVITY_CLASS: str = "sensitivity_class"
PROP_INFERRED_FUNCTION: str = "inferred_function"
PROP_PATH_PARAM_TYPE: str = "path_param_type"
PROP_RETURNS_PII: str = "returns_pii"
PROP_ACCEPTS_PII: str = "accepts_pii"
PROP_HAS_ROLE_PARAM: str = "has_role_param"
PROP_ACCEPTS_URL_PARAM: str = "accepts_url_param"
PROP_AUTH_SCHEME_NAMES: str = "auth_scheme_names"  # stored as string array

# :Resource
PROP_PATH_PREFIX: str = "path_prefix"
PROP_IDENTIFIER_TYPE: str = "identifier_type"
PROP_PARENT_RESOURCE_NAME: str = "parent_resource_name"

# :AuthScheme
PROP_AUTH_TYPE: str = "auth_type"
PROP_IS_JWT: str = "is_jwt"

# :CAN_REACH relationship properties
PROP_ATTACK_PATTERN: str = "attack_pattern"
PROP_ATTACK_REASON: str = "attack_reason"
PROP_RISK_SCORE: str = "risk_score"

# ── DDL: constraints ───────────────────────────────────────────────────────────
#
# Neo4j 5.x syntax: CREATE CONSTRAINT name IF NOT EXISTS FOR (n:Label) REQUIRE ...
# All constraints use IF NOT EXISTS for idempotent startup behaviour.

CONSTRAINTS: list[str] = [
    # :ApiSpec.id is globally unique (title+version slug).
    f"""CREATE CONSTRAINT api_spec_id_unique IF NOT EXISTS
    FOR (s:{LABEL_API_SPEC}) REQUIRE s.{PROP_ID} IS UNIQUE""",

    # :Endpoint.id is METHOD:path, unique per spec.
    # A composite unique constraint prevents duplicate endpoint nodes when the
    # same path/method pair appears in two different parsed specs.
    f"""CREATE CONSTRAINT endpoint_key_unique IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) REQUIRE (e.{PROP_ID}, e.{PROP_SPEC_ID}) IS UNIQUE""",

    # :Resource uniqueness: path_prefix + spec_id (two specs may share /users).
    f"""CREATE CONSTRAINT resource_key_unique IF NOT EXISTS
    FOR (r:{LABEL_RESOURCE}) REQUIRE (r.{PROP_PATH_PREFIX}, r.{PROP_SPEC_ID}) IS UNIQUE""",

    # :AuthScheme uniqueness: scheme name + spec_id.
    f"""CREATE CONSTRAINT auth_scheme_key_unique IF NOT EXISTS
    FOR (a:{LABEL_AUTH_SCHEME}) REQUIRE (a.{PROP_NAME}, a.{PROP_SPEC_ID}) IS UNIQUE""",
]

# ── DDL: indexes ───────────────────────────────────────────────────────────────
#
# Indexes on properties used in WHERE clauses by M9 traversal queries.
# None are needed on id/spec_id since those are covered by constraints above.

INDEXES: list[str] = [
    # Traversal filters: find all public / high-sensitivity / auth endpoints.
    f"""CREATE INDEX endpoint_is_public IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_IS_PUBLIC})""",

    f"""CREATE INDEX endpoint_auth_declared IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_AUTH_DECLARED})""",

    f"""CREATE INDEX endpoint_sensitivity IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_SENSITIVITY_CLASS})""",

    f"""CREATE INDEX endpoint_function IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_INFERRED_FUNCTION})""",

    # BOLA/IDOR detection: find resources with integer path param identifiers.
    f"""CREATE INDEX endpoint_path_param_type IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_PATH_PARAM_TYPE})""",

    f"""CREATE INDEX resource_identifier_type IF NOT EXISTS
    FOR (r:{LABEL_RESOURCE}) ON (r.{PROP_IDENTIFIER_TYPE})""",

    # Spec-scoped queries: fetch all endpoints / resources for one spec.
    f"""CREATE INDEX endpoint_spec_id IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_SPEC_ID})""",

    f"""CREATE INDEX resource_spec_id IF NOT EXISTS
    FOR (r:{LABEL_RESOURCE}) ON (r.{PROP_SPEC_ID})""",

    # PII signal: find endpoints that return or accept sensitive data.
    f"""CREATE INDEX endpoint_returns_pii IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_RETURNS_PII})""",

    f"""CREATE INDEX endpoint_accepts_pii IF NOT EXISTS
    FOR (e:{LABEL_ENDPOINT}) ON (e.{PROP_ACCEPTS_PII})""",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_spec_id(title: str, version: str) -> str:
    """Return a stable, URL-safe identifier for a spec.

    ``title`` is slugified (lowercase, non-alphanumeric runs → hyphens) then
    combined with ``version``: ``"my-api:1.0.0"``.  Stable across parse runs
    for the same spec, which is the key requirement for idempotent re-analysis.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "unknown"
    return f"{slug}:{version}"


def apply_schema(driver: Driver) -> None:
    """Create all constraints and indexes. Safe to call on every startup.

    Uses ``IF NOT EXISTS`` on every statement — Neo4j 5.x silently skips
    creation if the constraint/index is already present.
    """
    with driver.session() as session:
        for stmt in CONSTRAINTS + INDEXES:
            session.run(stmt)


def wipe_spec(driver: Driver, spec_id: str) -> None:
    """Remove all graph nodes (and their relationships) for a given spec.

    Useful before re-analysing a spec to avoid stale data.  Deletes in two
    passes: first all nodes with ``spec_id`` property, then the root
    :ApiSpec node (whose ``id`` equals ``spec_id``).
    """
    with driver.session() as session:
        # Delete all child nodes (Endpoint, Resource, AuthScheme).
        session.run(
            "MATCH (n) WHERE n.spec_id = $spec_id DETACH DELETE n",
            spec_id=spec_id,
        )
        # Delete the root ApiSpec node.
        session.run(
            f"MATCH (s:{LABEL_API_SPEC} {{id: $spec_id}}) DETACH DELETE s",
            spec_id=spec_id,
        )
