"""Unit tests for api_analyzer.graph.schema.

Tests cover: constant integrity, DDL statement syntax, make_spec_id(),
apply_schema() call pattern, and wipe_spec() call pattern.
No running Neo4j instance is needed — apply_schema and wipe_spec are
exercised against a MagicMock driver.
"""

import re
from unittest.mock import MagicMock, call, patch

import pytest

from api_analyzer.graph.schema import (
    CONSTRAINTS,
    INDEXES,
    LABEL_API_SPEC,
    LABEL_AUTH_SCHEME,
    LABEL_ENDPOINT,
    LABEL_RESOURCE,
    PROP_ACCEPTS_PII,
    PROP_ACCEPTS_URL_PARAM,
    PROP_ATTACK_PATTERN,
    PROP_ATTACK_REASON,
    PROP_AUTH_SCHEME_NAMES,
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
    PROP_RISK_SCORE,
    PROP_SENSITIVITY_CLASS,
    PROP_SPEC_COMPLETENESS,
    PROP_SPEC_FORMAT,
    PROP_SPEC_ID,
    PROP_SUMMARY,
    PROP_TITLE,
    PROP_VERSION,
    REL_CAN_REACH,
    REL_CHILD_OF,
    REL_DELETES,
    REL_LISTS,
    REL_PART_OF,
    REL_READS,
    REL_REQUIRES_AUTH,
    REL_WRITES,
    apply_schema,
    make_spec_id,
    wipe_spec,
)


# ── Node label constants ───────────────────────────────────────────────────────


class TestNodeLabels:
    def test_all_labels_are_non_empty_strings(self) -> None:
        for label in (LABEL_API_SPEC, LABEL_ENDPOINT, LABEL_RESOURCE, LABEL_AUTH_SCHEME):
            assert isinstance(label, str) and label

    def test_labels_are_title_case(self) -> None:
        """Node labels must be TitleCase — Neo4j convention."""
        for label in (LABEL_API_SPEC, LABEL_ENDPOINT, LABEL_RESOURCE, LABEL_AUTH_SCHEME):
            assert label[0].isupper(), f"{label!r} should start with uppercase"

    def test_label_values(self) -> None:
        assert LABEL_API_SPEC == "ApiSpec"
        assert LABEL_ENDPOINT == "Endpoint"
        assert LABEL_RESOURCE == "Resource"
        assert LABEL_AUTH_SCHEME == "AuthScheme"


# ── Relationship type constants ────────────────────────────────────────────────


class TestRelationshipTypes:
    _ALL_RELS = (
        REL_PART_OF, REL_REQUIRES_AUTH, REL_LISTS, REL_READS,
        REL_WRITES, REL_DELETES, REL_CHILD_OF, REL_CAN_REACH,
    )

    def test_all_rels_are_non_empty_strings(self) -> None:
        for rel in self._ALL_RELS:
            assert isinstance(rel, str) and rel

    def test_all_rels_are_screaming_snake_case(self) -> None:
        """Relationship types must be SCREAMING_SNAKE_CASE — Neo4j convention."""
        valid = re.compile(r"^[A-Z][A-Z0-9_]*$")
        for rel in self._ALL_RELS:
            assert valid.match(rel), f"{rel!r} is not SCREAMING_SNAKE_CASE"

    def test_rel_values(self) -> None:
        assert REL_PART_OF == "PART_OF"
        assert REL_REQUIRES_AUTH == "REQUIRES_AUTH"
        assert REL_LISTS == "LISTS"
        assert REL_READS == "READS"
        assert REL_WRITES == "WRITES"
        assert REL_DELETES == "DELETES"
        assert REL_CHILD_OF == "CHILD_OF"
        assert REL_CAN_REACH == "CAN_REACH"


# ── Property key constants ─────────────────────────────────────────────────────


class TestPropertyKeys:
    _ALL_PROPS = (
        PROP_ID, PROP_SPEC_ID, PROP_NAME,
        PROP_TITLE, PROP_VERSION, PROP_SPEC_FORMAT, PROP_SPEC_COMPLETENESS,
        PROP_PATH, PROP_METHOD, PROP_SUMMARY, PROP_IS_PUBLIC,
        PROP_SENSITIVITY_CLASS, PROP_INFERRED_FUNCTION, PROP_PATH_PARAM_TYPE,
        PROP_RETURNS_PII, PROP_ACCEPTS_PII, PROP_HAS_ROLE_PARAM,
        PROP_ACCEPTS_URL_PARAM, PROP_AUTH_SCHEME_NAMES,
        PROP_PATH_PREFIX, PROP_IDENTIFIER_TYPE, PROP_PARENT_RESOURCE_NAME,
        PROP_AUTH_TYPE, PROP_IS_JWT,
        PROP_ATTACK_PATTERN, PROP_ATTACK_REASON, PROP_RISK_SCORE,
    )

    def test_all_props_are_non_empty_strings(self) -> None:
        for prop in self._ALL_PROPS:
            assert isinstance(prop, str) and prop

    def test_all_props_are_snake_case(self) -> None:
        """Property keys must be snake_case — Neo4j property naming convention."""
        valid = re.compile(r"^[a-z][a-z0-9_]*$")
        for prop in self._ALL_PROPS:
            assert valid.match(prop), f"{prop!r} is not snake_case"

    def test_no_duplicate_prop_values(self) -> None:
        values = list(self._ALL_PROPS)
        assert len(values) == len(set(values)), "Duplicate property key values detected"

    def test_key_prop_values(self) -> None:
        assert PROP_ID == "id"
        assert PROP_SPEC_ID == "spec_id"
        assert PROP_IS_PUBLIC == "is_public"
        assert PROP_SENSITIVITY_CLASS == "sensitivity_class"
        assert PROP_INFERRED_FUNCTION == "inferred_function"
        assert PROP_PATH_PARAM_TYPE == "path_param_type"
        assert PROP_IDENTIFIER_TYPE == "identifier_type"
        assert PROP_AUTH_SCHEME_NAMES == "auth_scheme_names"


# ── CONSTRAINTS list ───────────────────────────────────────────────────────────


class TestConstraints:
    def test_constraints_is_non_empty_list(self) -> None:
        assert isinstance(CONSTRAINTS, list)
        assert len(CONSTRAINTS) >= 1

    def test_every_constraint_is_a_string(self) -> None:
        for stmt in CONSTRAINTS:
            assert isinstance(stmt, str) and stmt.strip()

    def test_every_constraint_contains_keyword(self) -> None:
        for stmt in CONSTRAINTS:
            assert "CONSTRAINT" in stmt.upper(), f"Missing CONSTRAINT keyword: {stmt!r}"

    def test_every_constraint_has_if_not_exists(self) -> None:
        for stmt in CONSTRAINTS:
            assert "IF NOT EXISTS" in stmt.upper(), (
                f"Missing IF NOT EXISTS (needed for idempotency): {stmt!r}"
            )

    def test_every_constraint_has_require_clause(self) -> None:
        for stmt in CONSTRAINTS:
            assert "REQUIRE" in stmt.upper(), f"Missing REQUIRE clause: {stmt!r}"

    def test_unique_constraints_reference_known_labels(self) -> None:
        known_labels = {LABEL_API_SPEC, LABEL_ENDPOINT, LABEL_RESOURCE, LABEL_AUTH_SCHEME}
        for stmt in CONSTRAINTS:
            found = any(lbl in stmt for lbl in known_labels)
            assert found, f"Constraint doesn't reference a known node label: {stmt!r}"

    def test_api_spec_constraint_present(self) -> None:
        assert any(LABEL_API_SPEC in c for c in CONSTRAINTS)

    def test_endpoint_constraint_present(self) -> None:
        assert any(LABEL_ENDPOINT in c for c in CONSTRAINTS)

    def test_resource_constraint_present(self) -> None:
        assert any(LABEL_RESOURCE in c for c in CONSTRAINTS)

    def test_auth_scheme_constraint_present(self) -> None:
        assert any(LABEL_AUTH_SCHEME in c for c in CONSTRAINTS)


# ── INDEXES list ───────────────────────────────────────────────────────────────


class TestIndexes:
    def test_indexes_is_non_empty_list(self) -> None:
        assert isinstance(INDEXES, list)
        assert len(INDEXES) >= 1

    def test_every_index_is_a_string(self) -> None:
        for stmt in INDEXES:
            assert isinstance(stmt, str) and stmt.strip()

    def test_every_index_contains_keyword(self) -> None:
        for stmt in INDEXES:
            assert "INDEX" in stmt.upper(), f"Missing INDEX keyword: {stmt!r}"

    def test_every_index_has_if_not_exists(self) -> None:
        for stmt in INDEXES:
            assert "IF NOT EXISTS" in stmt.upper(), (
                f"Missing IF NOT EXISTS (needed for idempotency): {stmt!r}"
            )

    def test_every_index_has_on_clause(self) -> None:
        for stmt in INDEXES:
            assert " ON " in stmt.upper(), f"Missing ON clause: {stmt!r}"

    def test_is_public_index_present(self) -> None:
        assert any(PROP_IS_PUBLIC in idx for idx in INDEXES)

    def test_sensitivity_class_index_present(self) -> None:
        assert any(PROP_SENSITIVITY_CLASS in idx for idx in INDEXES)

    def test_inferred_function_index_present(self) -> None:
        assert any(PROP_INFERRED_FUNCTION in idx for idx in INDEXES)

    def test_identifier_type_index_present(self) -> None:
        assert any(PROP_IDENTIFIER_TYPE in idx for idx in INDEXES)

    def test_spec_id_index_on_endpoint(self) -> None:
        assert any(PROP_SPEC_ID in idx and LABEL_ENDPOINT in idx for idx in INDEXES)


# ── make_spec_id ───────────────────────────────────────────────────────────────


class TestMakeSpecId:
    def test_simple_title_and_version(self) -> None:
        result = make_spec_id("My API", "1.0.0")
        assert result == "my-api:1.0.0"

    def test_special_chars_in_title_slugified(self) -> None:
        result = make_spec_id("My API (v2)!", "2.0")
        # Non-alphanumeric runs become single hyphens, strip leading/trailing.
        assert ":" in result
        slug, version = result.split(":", 1)
        assert version == "2.0"
        assert re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", slug), (
            f"Slug {slug!r} contains non-URL-safe chars"
        )

    def test_format_is_slug_colon_version(self) -> None:
        result = make_spec_id("Petstore", "3.0.0")
        assert result.count(":") == 1
        slug, version = result.split(":", 1)
        assert slug == "petstore"
        assert version == "3.0.0"

    def test_stable_across_calls(self) -> None:
        a = make_spec_id("Users API", "1.0")
        b = make_spec_id("Users API", "1.0")
        assert a == b

    def test_different_versions_differ(self) -> None:
        assert make_spec_id("API", "1.0") != make_spec_id("API", "2.0")

    def test_different_titles_differ(self) -> None:
        assert make_spec_id("Alpha", "1.0") != make_spec_id("Beta", "1.0")

    def test_empty_title_produces_unknown(self) -> None:
        result = make_spec_id("", "1.0")
        assert result.startswith("unknown:")

    def test_all_special_chars_title_produces_unknown(self) -> None:
        result = make_spec_id("!@#$%", "1.0")
        assert result.startswith("unknown:")

    def test_result_is_url_safe(self) -> None:
        result = make_spec_id("crAPI — Complete REST API", "0.1.0")
        # No spaces or chars that would break a URL path component.
        # Dots are allowed in the version segment (e.g. "0.1.0").
        assert " " not in result
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-:." for c in result)


# ── apply_schema ───────────────────────────────────────────────────────────────


class TestApplySchema:
    def _make_driver(self) -> MagicMock:
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver

    def test_opens_exactly_one_session(self) -> None:
        driver = self._make_driver()
        apply_schema(driver)
        driver.session.assert_called_once()

    def test_run_called_for_every_constraint_and_index(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        apply_schema(driver)
        expected_calls = len(CONSTRAINTS) + len(INDEXES)
        assert session.run.call_count == expected_calls

    def test_constraints_run_before_indexes(self) -> None:
        """Constraints must be applied before indexes (Neo4j startup order)."""
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        apply_schema(driver)
        all_stmts = [c.args[0] for c in session.run.call_args_list]
        constraint_positions = [i for i, s in enumerate(all_stmts) if "CONSTRAINT" in s.upper()]
        index_positions = [i for i, s in enumerate(all_stmts) if "INDEX" in s.upper()]
        if constraint_positions and index_positions:
            assert max(constraint_positions) < min(index_positions)

    def test_all_constraint_stmts_executed(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        apply_schema(driver)
        executed = [c.args[0] for c in session.run.call_args_list]
        for constraint in CONSTRAINTS:
            assert constraint in executed

    def test_all_index_stmts_executed(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        apply_schema(driver)
        executed = [c.args[0] for c in session.run.call_args_list]
        for index in INDEXES:
            assert index in executed


# ── wipe_spec ──────────────────────────────────────────────────────────────────


class TestWipeSpec:
    def _make_driver(self) -> MagicMock:
        driver = MagicMock()
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        return driver

    def test_opens_exactly_one_session(self) -> None:
        driver = self._make_driver()
        wipe_spec(driver, "my-api:1.0")
        driver.session.assert_called_once()

    def test_run_called_twice(self) -> None:
        """wipe_spec issues exactly two queries: child nodes, then root."""
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "my-api:1.0")
        assert session.run.call_count == 2

    def test_spec_id_passed_as_parameter_in_first_query(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "my-api:1.0")
        first_call = session.run.call_args_list[0]
        # Spec ID must be parameterised (not string-interpolated) to prevent injection.
        assert "spec_id" in first_call.kwargs or (
            len(first_call.args) > 1 and "spec_id" in str(first_call.args[1])
        )

    def test_spec_id_passed_as_parameter_in_second_query(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "my-api:1.0")
        second_call = session.run.call_args_list[1]
        assert "spec_id" in second_call.kwargs or (
            len(second_call.args) > 1 and "spec_id" in str(second_call.args[1])
        )

    def test_first_query_detach_deletes_child_nodes(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "my-api:1.0")
        first_stmt = session.run.call_args_list[0].args[0].upper()
        assert "DETACH DELETE" in first_stmt
        assert "SPEC_ID" in first_stmt  # filters on spec_id property

    def test_second_query_targets_api_spec_label(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "my-api:1.0")
        second_stmt = session.run.call_args_list[1].args[0]
        assert LABEL_API_SPEC in second_stmt

    def test_wipe_uses_provided_spec_id_value(self) -> None:
        driver = self._make_driver()
        session = driver.session.return_value.__enter__.return_value
        wipe_spec(driver, "target-api:2.5")
        # Both calls must pass the correct spec_id value.
        for c in session.run.call_args_list:
            assert "target-api:2.5" in str(c)
