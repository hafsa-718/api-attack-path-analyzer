"""Unit tests for api_analyzer.models.report.

Covers: AnalysisResult computed severity counts and highest_severity,
ChainSummary.from_chain derivation, and ReportContext.from_result.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from api_analyzer.models import (
    AnalysisResult,
    ChainSummary,
    ReportContext,
    Severity,
    SpecFormat,
    ValidatedChain,
)
from tests.conftest import SAMPLE_MITRE_IDS


class TestAnalysisResultComputedCounts:
    """Severity counts must derive from chains, not be settable separately."""

    def test_chain_count_matches_chains_length(
        self, sample_analysis_result: AnalysisResult
    ) -> None:
        assert sample_analysis_result.chain_count == len(sample_analysis_result.chains)

    def test_chain_count_zero_for_empty_chains(self) -> None:
        result = _make_result(chains=[])
        assert result.chain_count == 0

    def test_critical_count_correct(self, make_validated_chain: object) -> None:
        chains = [
            make_validated_chain(severity=Severity.CRITICAL),
            make_validated_chain(severity=Severity.HIGH),
            make_validated_chain(severity=Severity.CRITICAL),
        ]
        result = _make_result(chains=chains)
        assert result.critical_count == 2
        assert result.high_count == 1
        assert result.medium_count == 0
        assert result.low_count == 0

    def test_high_count_correct(self, make_validated_chain: object) -> None:
        chains = [make_validated_chain(severity=Severity.HIGH) for _ in range(3)]
        result = _make_result(chains=chains)
        assert result.high_count == 3
        assert result.critical_count == 0

    def test_medium_count_correct(self, make_validated_chain: object) -> None:
        chains = [make_validated_chain(severity=Severity.MEDIUM)]
        result = _make_result(chains=chains)
        assert result.medium_count == 1

    def test_low_count_correct(self, make_validated_chain: object) -> None:
        chains = [make_validated_chain(severity=Severity.LOW)]
        result = _make_result(chains=chains)
        assert result.low_count == 1

    def test_all_counts_zero_with_empty_chains(self) -> None:
        result = _make_result(chains=[])
        assert result.critical_count == 0
        assert result.high_count == 0
        assert result.medium_count == 0
        assert result.low_count == 0

    def test_computed_fields_in_serialisation(
        self, sample_analysis_result: AnalysisResult
    ) -> None:
        data = sample_analysis_result.model_dump()
        assert "chain_count" in data
        assert "critical_count" in data
        assert "high_count" in data
        assert "medium_count" in data
        assert "low_count" in data
        assert "highest_severity" in data


class TestAnalysisResultHighestSeverity:
    def test_returns_none_for_empty_chains(self) -> None:
        result = _make_result(chains=[])
        assert result.highest_severity is None

    def test_returns_critical_when_present(self, make_validated_chain: object) -> None:
        chains = [
            make_validated_chain(severity=Severity.HIGH),
            make_validated_chain(severity=Severity.CRITICAL),
            make_validated_chain(severity=Severity.MEDIUM),
        ]
        result = _make_result(chains=chains)
        assert result.highest_severity == Severity.CRITICAL

    def test_returns_high_when_no_critical(self, make_validated_chain: object) -> None:
        chains = [
            make_validated_chain(severity=Severity.HIGH),
            make_validated_chain(severity=Severity.LOW),
        ]
        result = _make_result(chains=chains)
        assert result.highest_severity == Severity.HIGH

    def test_single_chain_returns_its_severity(self, make_validated_chain: object) -> None:
        chain = make_validated_chain(severity=Severity.MEDIUM)
        result = _make_result(chains=[chain])
        assert result.highest_severity == Severity.MEDIUM

    def test_all_same_severity_returns_that_severity(self, make_validated_chain: object) -> None:
        chains = [make_validated_chain(severity=Severity.LOW) for _ in range(4)]
        result = _make_result(chains=chains)
        assert result.highest_severity == Severity.LOW

    def test_analysis_result_is_frozen(self, sample_analysis_result: AnalysisResult) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            sample_analysis_result.endpoint_count = 999  # type: ignore[misc]


class TestChainSummaryFromChain:
    def test_summary_id_matches_chain(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.id == sample_validated_chain.id

    def test_summary_name_matches_chain(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.name == sample_validated_chain.name

    def test_summary_severity_matches_chain(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.severity == sample_validated_chain.severity

    def test_summary_confidence_is_final_score(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.confidence == pytest.approx(sample_validated_chain.confidence.final_score)

    def test_summary_entry_from_first_step(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        first_step = sample_validated_chain.steps[0]
        assert first_step.method in summary.entry
        assert first_step.path in summary.entry

    def test_summary_exit_from_last_step(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        last_step = sample_validated_chain.steps[-1]
        assert last_step.method in summary.exit_point
        assert last_step.path in summary.exit_point

    def test_summary_hop_count_is_steps_minus_one(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.hop_count == len(sample_validated_chain.steps) - 1

    def test_summary_mitre_ids_match_chain(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.mitre_ids == list(sample_validated_chain.mitre_techniques)

    def test_summary_owasp_matches_chain(self, sample_validated_chain: ValidatedChain) -> None:
        summary = ChainSummary.from_chain(sample_validated_chain)
        assert summary.owasp == sample_validated_chain.owasp_category

    def test_summary_is_frozen(self, sample_validated_chain: ValidatedChain) -> None:
        from pydantic import ValidationError
        summary = ChainSummary.from_chain(sample_validated_chain)
        with pytest.raises(ValidationError):
            summary.severity = Severity.INFO  # type: ignore[misc]


class TestReportContextFromResult:
    def test_chain_summaries_count_matches_chains(
        self, sample_analysis_result: AnalysisResult
    ) -> None:
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        assert len(ctx.chain_summaries) == len(sample_analysis_result.chains)

    def test_chain_summary_ids_match_chains(
        self, sample_analysis_result: AnalysisResult
    ) -> None:
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        chain_ids = {c.id for c in sample_analysis_result.chains}
        summary_ids = {s.id for s in ctx.chain_summaries}
        assert chain_ids == summary_ids

    def test_tool_version_stored(self, sample_analysis_result: AnalysisResult) -> None:
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        assert ctx.tool_version == "0.1.0"

    def test_generated_at_is_utc(self, sample_analysis_result: AnalysisResult) -> None:
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        assert ctx.generated_at.tzinfo is not None

    def test_include_graph_defaults_true(self, sample_analysis_result: AnalysisResult) -> None:
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        assert ctx.include_graph is True

    def test_empty_chains_produces_empty_summaries(self) -> None:
        result = _make_result(chains=[])
        ctx = ReportContext.from_result(result, tool_version="0.1.0")
        assert ctx.chain_summaries == []

    def test_report_context_is_frozen(self, sample_analysis_result: AnalysisResult) -> None:
        from pydantic import ValidationError
        ctx = ReportContext.from_result(sample_analysis_result, tool_version="0.1.0")
        with pytest.raises(ValidationError):
            ctx.tool_version = "9.9.9"  # type: ignore[misc]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(chains: list[ValidatedChain]) -> AnalysisResult:
    """Build a minimal AnalysisResult with the given chains list."""
    return AnalysisResult(
        analysis_id=str(uuid4()),
        spec_title="Test API",
        spec_version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        analyzed_at=datetime.now(tz=timezone.utc),
        duration_seconds=10.0,
        endpoint_count=10,
        public_endpoint_count=2,
        auth_declared=True,
        spec_completeness=0.8,
        chains=chains,
        graph_node_count=20,
        graph_edge_count=25,
        patterns_run=["AP-001"],
        candidates_evaluated=len(chains) + 2,
        candidates_rejected=2,
        llm_tokens_used=500,
        estimated_cost_usd=0.005,
    )
