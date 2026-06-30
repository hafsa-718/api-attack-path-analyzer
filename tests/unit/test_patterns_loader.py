"""Unit tests for api_analyzer.patterns.loader.

Tests cover:
  - AttackPattern dataclass structure and immutability
  - load_patterns(): correct count, types, field values
  - get_pattern(): happy path, miss, case sensitivity
  - YAML data integrity: OWASP format, severity values, confidence ranges
  - LRU cache: load_patterns() returns the same list object on repeated calls
"""

from __future__ import annotations

import re

import pytest

from api_analyzer.models.enums import Severity
from api_analyzer.patterns.loader import (
    AttackPattern,
    _DATA_FILE,
    get_pattern,
    load_patterns,
)

# All expected pattern IDs in declaration order
_EXPECTED_IDS: list[str] = [
    "AP-001",
    "AP-002",
    "AP-003",
    "AP-004",
    "AP-005",
    "AP-006",
    "AP-007",
]

_OWASP_CATEGORY_RE = re.compile(r"^API\d+:\d{4}$")
_MITRE_ID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")


# ── Data file sanity ───────────────────────────────────────────────────────────


class TestDataFile:
    def test_yaml_file_exists(self) -> None:
        assert _DATA_FILE.exists(), f"Bundled YAML not found at {_DATA_FILE}"

    def test_yaml_file_is_not_empty(self) -> None:
        assert _DATA_FILE.stat().st_size > 0


# ── AttackPattern dataclass ────────────────────────────────────────────────────


class TestAttackPatternDataclass:
    def _make(self) -> AttackPattern:
        return AttackPattern(
            id="AP-TEST",
            name="Test Pattern",
            owasp_category="API1:2023",
            owasp_description="Test",
            severity=Severity.HIGH,
            confidence_base=0.5,
            mitre_hints=("T1078",),
            description="A test pattern.",
            indicators=("signal one",),
            remediation=("fix one",),
        )

    def test_frozen_raises_on_assignment(self) -> None:
        p = self._make()
        with pytest.raises((AttributeError, TypeError)):
            p.name = "mutated"  # type: ignore[misc]

    def test_sequence_fields_are_tuples(self) -> None:
        p = self._make()
        assert isinstance(p.mitre_hints, tuple)
        assert isinstance(p.indicators, tuple)
        assert isinstance(p.remediation, tuple)

    def test_severity_is_enum(self) -> None:
        p = self._make()
        assert isinstance(p.severity, Severity)

    def test_confidence_base_is_float(self) -> None:
        p = self._make()
        assert isinstance(p.confidence_base, float)

    def test_pattern_is_hashable(self) -> None:
        p = self._make()
        assert hash(p) is not None
        d = {p: "value"}
        assert d[p] == "value"


# ── load_patterns() ────────────────────────────────────────────────────────────


class TestLoadPatterns:
    def test_returns_list(self) -> None:
        assert isinstance(load_patterns(), list)

    def test_returns_seven_patterns(self) -> None:
        assert len(load_patterns()) == 7

    def test_all_items_are_attack_patterns(self) -> None:
        for p in load_patterns():
            assert isinstance(p, AttackPattern)

    def test_all_expected_ids_present(self) -> None:
        loaded_ids = [p.id for p in load_patterns()]
        for expected_id in _EXPECTED_IDS:
            assert expected_id in loaded_ids, f"Missing pattern: {expected_id}"

    def test_ids_are_in_declaration_order(self) -> None:
        loaded_ids = [p.id for p in load_patterns()]
        assert loaded_ids == _EXPECTED_IDS

    def test_lru_cache_returns_same_object(self) -> None:
        assert load_patterns() is load_patterns()

    def test_all_names_non_empty(self) -> None:
        for p in load_patterns():
            assert p.name.strip(), f"Empty name for {p.id}"

    def test_all_descriptions_non_empty(self) -> None:
        for p in load_patterns():
            assert p.description.strip(), f"Empty description for {p.id}"

    def test_descriptions_are_stripped(self) -> None:
        for p in load_patterns():
            assert p.description == p.description.strip(), \
                f"Description not stripped for {p.id}"

    def test_all_severity_values_are_enum_members(self) -> None:
        valid = set(Severity)
        for p in load_patterns():
            assert p.severity in valid, f"Invalid severity {p.severity!r} for {p.id}"

    def test_confidence_base_in_unit_interval(self) -> None:
        for p in load_patterns():
            assert 0.0 <= p.confidence_base <= 1.0, \
                f"confidence_base {p.confidence_base} out of range for {p.id}"

    def test_confidence_base_values_are_floats(self) -> None:
        for p in load_patterns():
            assert isinstance(p.confidence_base, float), \
                f"confidence_base not float for {p.id}"

    def test_owasp_category_format(self) -> None:
        for p in load_patterns():
            assert _OWASP_CATEGORY_RE.match(p.owasp_category), \
                f"Bad OWASP category {p.owasp_category!r} for {p.id}"

    def test_owasp_description_non_empty(self) -> None:
        for p in load_patterns():
            assert p.owasp_description.strip(), f"Empty owasp_description for {p.id}"

    def test_mitre_hints_are_tuples(self) -> None:
        for p in load_patterns():
            assert isinstance(p.mitre_hints, tuple), f"mitre_hints not tuple for {p.id}"

    def test_mitre_hint_id_format(self) -> None:
        for p in load_patterns():
            for hint in p.mitre_hints:
                assert _MITRE_ID_RE.match(hint), \
                    f"Bad MITRE ID {hint!r} in {p.id}"

    def test_indicators_are_non_empty_tuples(self) -> None:
        for p in load_patterns():
            assert isinstance(p.indicators, tuple), f"indicators not tuple for {p.id}"
            assert len(p.indicators) >= 1, f"No indicators for {p.id}"

    def test_remediation_is_non_empty_tuple(self) -> None:
        for p in load_patterns():
            assert isinstance(p.remediation, tuple), f"remediation not tuple for {p.id}"
            assert len(p.remediation) >= 1, f"No remediation steps for {p.id}"


# ── Per-pattern field spot-checks ──────────────────────────────────────────────


class TestPatternFieldValues:
    def _get(self, pattern_id: str) -> AttackPattern:
        p = get_pattern(pattern_id)
        assert p is not None, f"Pattern {pattern_id} not found"
        return p

    def test_ap001_severity_is_high(self) -> None:
        assert self._get("AP-001").severity == Severity.HIGH

    def test_ap001_owasp_is_api1(self) -> None:
        assert self._get("AP-001").owasp_category == "API1:2023"

    def test_ap001_mitre_includes_t1078(self) -> None:
        assert "T1078" in self._get("AP-001").mitre_hints

    def test_ap002_severity_is_critical(self) -> None:
        assert self._get("AP-002").severity == Severity.CRITICAL

    def test_ap002_confidence_base_above_half(self) -> None:
        assert self._get("AP-002").confidence_base > 0.5

    def test_ap003_owasp_is_api3(self) -> None:
        assert self._get("AP-003").owasp_category == "API3:2023"

    def test_ap004_confidence_base_below_ap002(self) -> None:
        assert self._get("AP-004").confidence_base < self._get("AP-002").confidence_base

    def test_ap005_severity_is_medium(self) -> None:
        assert self._get("AP-005").severity == Severity.MEDIUM

    def test_ap006_owasp_is_api7(self) -> None:
        assert self._get("AP-006").owasp_category == "API7:2023"

    def test_ap007_severity_is_critical(self) -> None:
        assert self._get("AP-007").severity == Severity.CRITICAL

    def test_ap007_has_three_mitre_hints(self) -> None:
        assert len(self._get("AP-007").mitre_hints) == 3

    def test_ap007_mitre_includes_t1110(self) -> None:
        assert "T1110" in self._get("AP-007").mitre_hints


# ── get_pattern() ──────────────────────────────────────────────────────────────


class TestGetPattern:
    def test_returns_correct_pattern_by_id(self) -> None:
        p = get_pattern("AP-001")
        assert p is not None
        assert p.id == "AP-001"

    def test_returns_none_for_unknown_id(self) -> None:
        assert get_pattern("AP-999") is None

    def test_returns_none_for_empty_string(self) -> None:
        assert get_pattern("") is None

    def test_is_case_sensitive(self) -> None:
        assert get_pattern("ap-001") is None
        assert get_pattern("AP-001") is not None

    def test_all_ids_resolve(self) -> None:
        for pattern_id in _EXPECTED_IDS:
            p = get_pattern(pattern_id)
            assert p is not None, f"get_pattern({pattern_id!r}) returned None"
            assert p.id == pattern_id

    def test_returned_pattern_matches_load_patterns(self) -> None:
        by_get = get_pattern("AP-003")
        by_list = next(p for p in load_patterns() if p.id == "AP-003")
        assert by_get is by_list
