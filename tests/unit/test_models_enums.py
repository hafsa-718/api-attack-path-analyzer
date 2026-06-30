"""Unit tests for api_analyzer.models.enums.

Critical invariant: SensitivityClass and Severity have ordering semantics
that downstream modules (ranker M8, AnalysisResult) depend on via
list(Enum).index(value).  Any accidental reordering of members would
silently produce wrong scores or wrong highest_severity results.
These tests fail loudly if that ordering changes.
"""

import pytest

from api_analyzer.models.enums import (
    AuthType,
    EndpointFunction,
    HttpMethod,
    ParameterLocation,
    PathParamType,
    SensitivityClass,
    Severity,
    SpecFormat,
)


class TestSensitivityClassOrdering:
    """SensitivityClass ordering is a contract: PUBLIC=0, CRITICAL=3."""

    def test_public_is_index_zero(self) -> None:
        assert list(SensitivityClass).index(SensitivityClass.PUBLIC) == 0

    def test_internal_is_index_one(self) -> None:
        assert list(SensitivityClass).index(SensitivityClass.INTERNAL) == 1

    def test_sensitive_is_index_two(self) -> None:
        assert list(SensitivityClass).index(SensitivityClass.SENSITIVE) == 2

    def test_critical_is_index_three(self) -> None:
        assert list(SensitivityClass).index(SensitivityClass.CRITICAL) == 3

    def test_total_member_count_is_four(self) -> None:
        """Adding a new member without updating this test is a contract violation."""
        assert len(list(SensitivityClass)) == 4

    def test_index_ordering_supports_delta_calculation(self) -> None:
        """Simulate the ranker's sensitivity_delta calculation."""
        order = list(SensitivityClass)
        entry = SensitivityClass.PUBLIC
        exit_ = SensitivityClass.CRITICAL
        delta = order.index(exit_) - order.index(entry)
        assert delta == 3  # Maximum possible delta


class TestSeverityOrdering:
    """Severity ordering is a contract: CRITICAL=0 (highest), INFO=4 (lowest)."""

    def test_critical_is_index_zero(self) -> None:
        assert list(Severity).index(Severity.CRITICAL) == 0

    def test_high_is_index_one(self) -> None:
        assert list(Severity).index(Severity.HIGH) == 1

    def test_medium_is_index_two(self) -> None:
        assert list(Severity).index(Severity.MEDIUM) == 2

    def test_low_is_index_three(self) -> None:
        assert list(Severity).index(Severity.LOW) == 3

    def test_info_is_index_four(self) -> None:
        assert list(Severity).index(Severity.INFO) == 4

    def test_total_member_count_is_five(self) -> None:
        assert len(list(Severity)) == 5

    def test_min_by_index_returns_highest_severity(self) -> None:
        """Simulate AnalysisResult.highest_severity selection logic."""
        order = list(Severity)
        severities = [Severity.MEDIUM, Severity.CRITICAL, Severity.HIGH]
        worst = min(severities, key=lambda s: order.index(s))
        assert worst == Severity.CRITICAL

    def test_min_by_index_with_single_item(self) -> None:
        order = list(Severity)
        result = min([Severity.LOW], key=lambda s: order.index(s))
        assert result == Severity.LOW


class TestEnumStringSerialisation:
    """All enums inherit from str so they serialise to their value, not their name."""

    def test_http_method_serialises_to_uppercase_string(self) -> None:
        assert str(HttpMethod.GET) == "GET"
        assert HttpMethod.POST == "POST"

    def test_sensitivity_class_serialises_to_uppercase_string(self) -> None:
        assert str(SensitivityClass.CRITICAL) == "CRITICAL"

    def test_severity_serialises_to_uppercase_string(self) -> None:
        assert str(Severity.HIGH) == "HIGH"

    def test_auth_type_serialises_to_spec_value(self) -> None:
        # Values must match the OpenAPI spec type field values exactly
        assert AuthType.API_KEY == "apiKey"
        assert AuthType.OAUTH2 == "oauth2"
        assert AuthType.OPENID_CONNECT == "openIdConnect"

    def test_parameter_location_serialises_to_lowercase(self) -> None:
        assert ParameterLocation.PATH == "path"
        assert ParameterLocation.QUERY == "query"

    def test_spec_format_values(self) -> None:
        assert SpecFormat.OPENAPI3 == "openapi3"
        assert SpecFormat.SWAGGER2 == "swagger2"


class TestEnumMembership:
    """Basic membership and iteration sanity checks."""

    def test_http_method_contains_all_rest_verbs(self) -> None:
        methods = {m.value for m in HttpMethod}
        assert methods == {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}

    def test_path_param_type_includes_none_sentinel(self) -> None:
        assert PathParamType.NONE in list(PathParamType)

    def test_endpoint_function_includes_unknown_sentinel(self) -> None:
        assert EndpointFunction.UNKNOWN in list(EndpointFunction)

    def test_enum_value_lookup_by_string(self) -> None:
        """Enums must be constructable from their string values for deserialisation."""
        assert HttpMethod("GET") == HttpMethod.GET
        assert Severity("CRITICAL") == Severity.CRITICAL
        assert SensitivityClass("SENSITIVE") == SensitivityClass.SENSITIVE

    def test_invalid_enum_value_raises(self) -> None:
        with pytest.raises(ValueError):
            HttpMethod("TRACE")

        with pytest.raises(ValueError):
            Severity("EXTREME")
