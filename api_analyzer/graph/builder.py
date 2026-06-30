"""Graph builder: translates a classified ParsedSpec into a Neo4j graph.

Entry point
-----------
``build_graph(spec, driver)`` — the only public callable.  Returns a
``BuildResult`` with counts of nodes and relationships merged.

Idempotency
-----------
Every write uses ``MERGE`` (not ``CREATE``), so calling ``build_graph`` a second
time with the same spec updates existing nodes rather than duplicating them.
To start fresh, call ``wipe_spec(driver, spec_id)`` before ``build_graph``.

Atomicity
---------
All node and relationship writes run inside a single ``session.execute_write``
callback.  If any step fails the entire graph write rolls back.

Node creation order
-------------------
  1. :ApiSpec       — root node, created first so PART_OF edges have a target
  2. :AuthScheme    — must exist before REQUIRES_AUTH edges
  3. :Endpoint      — must exist before all outbound edges
  4. :Resource      — must exist before inbound Endpoint edges

Relationship creation order (all nodes must already exist)
-----------------------------------------------------------
  5. Endpoint -[:PART_OF]->    ApiSpec
  6. Endpoint -[:REQUIRES_AUTH]-> AuthScheme
  7. Endpoint -[:LISTS/READS/WRITES/DELETES]-> Resource
  8. Resource -[:CHILD_OF]->   Resource

Property serialisation
----------------------
StrEnum values (SensitivityClass, EndpointFunction, etc.) are converted to
``str`` so the Neo4j driver receives plain Python strings.  None-valued
optional properties are omitted to avoid storing null graph properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import Driver, ManagedTransaction

from api_analyzer.graph.schema import (
    LABEL_API_SPEC,
    LABEL_AUTH_SCHEME,
    LABEL_ENDPOINT,
    LABEL_RESOURCE,
    PROP_ACCEPTS_PII,
    PROP_ACCEPTS_URL_PARAM,
    PROP_AUTH_SCHEME_NAMES,
    PROP_AUTH_DECLARED,
    PROP_AUTH_TYPE,
    PROP_HAS_ROLE_PARAM,
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
    PROP_RETURNS_PII,
    PROP_SENSITIVITY_CLASS,
    PROP_SPEC_COMPLETENESS,
    PROP_SPEC_FORMAT,
    PROP_SPEC_ID,
    PROP_SUMMARY,
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
from api_analyzer.models.spec import AuthScheme, InferredResource, ParsedEndpoint, ParsedSpec

# ── Cypher templates (computed once at import time) ────────────────────────────
# Using module-level constants prevents string re-construction on every call.
# f-string interpolation only touches LABEL_* and PROP_* module constants —
# never user-supplied values — so there is no injection risk here.

_CQL_MERGE_SPEC: str = (
    f"MERGE (s:{LABEL_API_SPEC} {{{PROP_ID}: $id}}) "
    f"ON CREATE SET s = $props "
    f"ON MATCH SET s += $props"
)

_CQL_MERGE_AUTH_SCHEMES: str = (
    f"UNWIND $schemes AS s "
    f"MERGE (a:{LABEL_AUTH_SCHEME} "
    f"{{{PROP_NAME}: s.{PROP_NAME}, {PROP_SPEC_ID}: s.{PROP_SPEC_ID}}}) "
    f"ON CREATE SET a = s "
    f"ON MATCH SET a += s"
)

_CQL_MERGE_ENDPOINTS: str = (
    f"UNWIND $endpoints AS e_data "
    f"MERGE (e:{LABEL_ENDPOINT} "
    f"{{{PROP_ID}: e_data.{PROP_ID}, {PROP_SPEC_ID}: e_data.{PROP_SPEC_ID}}}) "
    f"ON CREATE SET e = e_data "
    f"ON MATCH SET e += e_data"
)

_CQL_MERGE_RESOURCES: str = (
    f"UNWIND $resources AS r_data "
    f"MERGE (r:{LABEL_RESOURCE} "
    f"{{{PROP_PATH_PREFIX}: r_data.{PROP_PATH_PREFIX}, {PROP_SPEC_ID}: r_data.{PROP_SPEC_ID}}}) "
    f"ON CREATE SET r = r_data "
    f"ON MATCH SET r += r_data"
)

_CQL_PART_OF: str = (
    f"UNWIND $endpoint_ids AS eid "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: eid, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (s:{LABEL_API_SPEC} {{{PROP_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_PART_OF}]->(s)"
)

_CQL_REQUIRES_AUTH: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: rel.endpoint_id, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (a:{LABEL_AUTH_SCHEME} {{{PROP_NAME}: rel.scheme_name, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_REQUIRES_AUTH}]->(a)"
)

_CQL_LISTS: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: rel.endpoint_id, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (r:{LABEL_RESOURCE} {{{PROP_PATH_PREFIX}: rel.path_prefix, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_LISTS}]->(r)"
)

_CQL_READS: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: rel.endpoint_id, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (r:{LABEL_RESOURCE} {{{PROP_PATH_PREFIX}: rel.path_prefix, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_READS}]->(r)"
)

_CQL_WRITES: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: rel.endpoint_id, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (r:{LABEL_RESOURCE} {{{PROP_PATH_PREFIX}: rel.path_prefix, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_WRITES}]->(r)"
)

_CQL_DELETES: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (e:{LABEL_ENDPOINT} {{{PROP_ID}: rel.endpoint_id, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (r:{LABEL_RESOURCE} {{{PROP_PATH_PREFIX}: rel.path_prefix, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (e)-[:{REL_DELETES}]->(r)"
)

_CQL_CHILD_OF: str = (
    f"UNWIND $rels AS rel "
    f"MATCH (child:{LABEL_RESOURCE} {{{PROP_PATH_PREFIX}: rel.child_prefix, {PROP_SPEC_ID}: $spec_id}}) "
    f"MATCH (parent:{LABEL_RESOURCE} {{{PROP_NAME}: rel.parent_name, {PROP_SPEC_ID}: $spec_id}}) "
    f"MERGE (child)-[:{REL_CHILD_OF}]->(parent)"
)


# ── Public API ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuildResult:
    """Summary of a ``build_graph`` call.

    Counts reflect how many nodes/relationships were submitted for MERGE,
    not the Neo4j delta (which depends on whether nodes already existed).
    Use Neo4j ``MERGE`` counters for precise created-vs-matched breakdown.
    """

    spec_id: str
    endpoint_count: int
    resource_count: int
    auth_scheme_count: int
    rel_count: int


def build_graph(spec: ParsedSpec, driver: Driver) -> BuildResult:
    """Write a classified ParsedSpec into Neo4j.

    All writes execute inside a single managed transaction; any failure
    causes a full rollback.  The function is safe to call multiple times for
    the same spec — MERGE ensures nodes/relationships are updated, not duplicated.

    Args:
        spec:   Output of ``classify()`` (M3) — must have sensitivity_class and
                inferred_function already set on each endpoint.
        driver: An open ``neo4j.Driver`` instance.

    Returns:
        ``BuildResult`` with node and relationship counts.
    """
    spec_id = make_spec_id(spec.title, spec.version)

    def _run(tx: ManagedTransaction) -> BuildResult:
        _build_spec_node(tx, spec, spec_id)
        auth_count = _build_auth_scheme_nodes(tx, spec, spec_id)
        ep_count = _build_endpoint_nodes(tx, spec, spec_id)
        res_count = _build_resource_nodes(tx, spec, spec_id)

        rel_count = 0
        rel_count += _build_part_of_rels(tx, spec, spec_id)
        rel_count += _build_requires_auth_rels(tx, spec, spec_id)
        rel_count += _build_resource_endpoint_rels(tx, spec, spec_id)
        rel_count += _build_child_of_rels(tx, spec, spec_id)

        return BuildResult(
            spec_id=spec_id,
            endpoint_count=ep_count,
            resource_count=res_count,
            auth_scheme_count=auth_count,
            rel_count=rel_count,
        )

    with driver.session() as session:
        return session.execute_write(_run)


# ── Node builders ──────────────────────────────────────────────────────────────


def _build_spec_node(tx: ManagedTransaction, spec: ParsedSpec, spec_id: str) -> None:
    tx.run(_CQL_MERGE_SPEC, id=spec_id, props=_spec_props(spec, spec_id))


def _build_auth_scheme_nodes(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    if not spec.auth_schemes:
        return 0
    schemes = [_auth_scheme_props(s, spec_id) for s in spec.auth_schemes.values()]
    tx.run(_CQL_MERGE_AUTH_SCHEMES, schemes=schemes)
    return len(schemes)


def _build_endpoint_nodes(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    if not spec.endpoints:
        return 0
    endpoints = [_endpoint_props(ep, spec_id) for ep in spec.endpoints]
    tx.run(_CQL_MERGE_ENDPOINTS, endpoints=endpoints)
    return len(endpoints)


def _build_resource_nodes(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    if not spec.resources:
        return 0
    resources = [_resource_props(r, spec_id) for r in spec.resources]
    tx.run(_CQL_MERGE_RESOURCES, resources=resources)
    return len(resources)


# ── Relationship builders ──────────────────────────────────────────────────────


def _build_part_of_rels(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    if not spec.endpoints:
        return 0
    endpoint_ids = [ep.id for ep in spec.endpoints]
    tx.run(_CQL_PART_OF, endpoint_ids=endpoint_ids, spec_id=spec_id)
    return len(endpoint_ids)


def _build_requires_auth_rels(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    rels = [
        {"endpoint_id": ep.id, "scheme_name": scheme}
        for ep in spec.endpoints
        for scheme in ep.auth_scheme_names
    ]
    if not rels:
        return 0
    tx.run(_CQL_REQUIRES_AUTH, rels=rels, spec_id=spec_id)
    return len(rels)


def _build_resource_endpoint_rels(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    """Build LISTS, READS, WRITES, DELETES edges from endpoints to resources."""
    total = 0

    lists_rels = [
        {"endpoint_id": r.collection_endpoint_id, "path_prefix": r.path_prefix}
        for r in spec.resources
        if r.collection_endpoint_id
    ]
    if lists_rels:
        tx.run(_CQL_LISTS, rels=lists_rels, spec_id=spec_id)
        total += len(lists_rels)

    reads_rels = [
        {"endpoint_id": r.detail_endpoint_id, "path_prefix": r.path_prefix}
        for r in spec.resources
        if r.detail_endpoint_id
    ]
    if reads_rels:
        tx.run(_CQL_READS, rels=reads_rels, spec_id=spec_id)
        total += len(reads_rels)

    writes_rels = [
        {"endpoint_id": ep_id, "path_prefix": r.path_prefix}
        for r in spec.resources
        for ep_id in r.write_endpoint_ids
    ]
    if writes_rels:
        tx.run(_CQL_WRITES, rels=writes_rels, spec_id=spec_id)
        total += len(writes_rels)

    deletes_rels = [
        {"endpoint_id": r.delete_endpoint_id, "path_prefix": r.path_prefix}
        for r in spec.resources
        if r.delete_endpoint_id
    ]
    if deletes_rels:
        tx.run(_CQL_DELETES, rels=deletes_rels, spec_id=spec_id)
        total += len(deletes_rels)

    return total


def _build_child_of_rels(
    tx: ManagedTransaction, spec: ParsedSpec, spec_id: str
) -> int:
    rels = [
        {"child_prefix": r.path_prefix, "parent_name": r.parent_resource_name}
        for r in spec.resources
        if r.parent_resource_name
    ]
    if not rels:
        return 0
    tx.run(_CQL_CHILD_OF, rels=rels, spec_id=spec_id)
    return len(rels)


# ── Property extraction ────────────────────────────────────────────────────────


def _spec_props(spec: ParsedSpec, spec_id: str) -> dict[str, Any]:
    """Property dict for a :ApiSpec node."""
    props: dict[str, Any] = {
        PROP_ID: spec_id,
        PROP_SPEC_ID: spec_id,
        PROP_TITLE: spec.title,
        PROP_VERSION: spec.version,
        PROP_SPEC_FORMAT: str(spec.spec_format),
        PROP_SPEC_COMPLETENESS: spec.spec_completeness,
    }
    if spec.description:
        props["description"] = spec.description
    return props


def _endpoint_props(ep: ParsedEndpoint, spec_id: str) -> dict[str, Any]:
    """Property dict for an :Endpoint node.

    Heavy structured fields (parameters, request_body, response_schemas) are
    excluded — the graph stores only the pre-computed security signals that the
    traversal engine and LLM agent need, not the raw schema trees.
    """
    props: dict[str, Any] = {
        PROP_ID: ep.id,
        PROP_SPEC_ID: spec_id,
        PROP_PATH: ep.path,
        PROP_METHOD: str(ep.method),
        PROP_IS_PUBLIC: ep.is_public,
        PROP_AUTH_DECLARED: ep.auth_declared,
        PROP_SENSITIVITY_CLASS: str(ep.sensitivity_class),
        PROP_INFERRED_FUNCTION: str(ep.inferred_function),
        PROP_PATH_PARAM_TYPE: str(ep.path_param_type),
        PROP_RETURNS_PII: ep.returns_pii,
        PROP_ACCEPTS_PII: ep.accepts_pii,
        PROP_HAS_ROLE_PARAM: ep.has_role_param,
        PROP_ACCEPTS_URL_PARAM: ep.accepts_url_param,
        PROP_AUTH_SCHEME_NAMES: list(ep.auth_scheme_names),
    }
    if ep.summary:
        props[PROP_SUMMARY] = ep.summary
    return props


def _resource_props(r: InferredResource, spec_id: str) -> dict[str, Any]:
    """Property dict for a :Resource node."""
    props: dict[str, Any] = {
        PROP_NAME: r.name,
        PROP_PATH_PREFIX: r.path_prefix,
        PROP_SPEC_ID: spec_id,
        PROP_IDENTIFIER_TYPE: str(r.identifier_type),
    }
    if r.parent_resource_name:
        props[PROP_PARENT_RESOURCE_NAME] = r.parent_resource_name
    if r.identifier_name:
        props["identifier_name"] = r.identifier_name
    return props


def _auth_scheme_props(scheme: AuthScheme, spec_id: str) -> dict[str, Any]:
    """Property dict for an :AuthScheme node."""
    return {
        PROP_NAME: scheme.name,
        PROP_SPEC_ID: spec_id,
        PROP_AUTH_TYPE: str(scheme.auth_type),
        PROP_IS_JWT: scheme.is_jwt,
    }
