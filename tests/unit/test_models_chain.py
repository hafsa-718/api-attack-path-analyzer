"""Unit tests for api_analyzer.models.chain.

Critical paths:
  - ConfidenceBreakdown.final_score: geometric mean formula, zero-component guard,
    evidence bonus, and 1.0 cap.
  - ValidatedChain.mitre_techniques: format validation is the hallucination firewall.
  - ValidatedChain.narrative: minimum length prevents empty LLM outputs.
  - Frozen models: security findings must not be mutable after construction.
"""

import statistics
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from pydantic import ValidationError

from api_analyzer.models import (
    AttackStep,
    CandidateChain,
    ConfidenceBreakdown,
    Severity,
    ValidatedChain,
)
from tests.conftest import SAMPLE_MITRE_IDS, SAMPLE_NARRATIVE


class TestConfidenceBreakdownFinalScore:
    """final_score is a computed field using geometric mean of three components."""

    def test_geometric_mean_of_equal_components(self) -> None:
        bd = ConfidenceBreakdown(
            graph_match_score=0.8,
            auth_clarity_score=0.8,
            llm_self_score=0.8,
            evidence_count=0,
            rationale="equal components",
        )
        # geometric_mean([0.8, 0.8, 0.8]) = 0.8; evidence_bonus = 0
        assert bd.final_score == pytest.approx(0.8, abs=1e-4)

    def test_geometric_mean_of_mixed_components(self) -> None:
        bd = ConfidenceBreakdown(
            graph_match_score=0.9,
            auth_clarity_score=0.7,
            llm_self_score=0.8,
            evidence_count=0,
            rationale="mixed",
        )
        expected_base = statistics.geometric_mean([0.9, 0.7, 0.8])
        assert bd.final_score == pytest.approx(expected_base, abs=1e-4)

    def test_zero_llm_score_returns_zero(self) -> None:
        """A zero LLM self-score must collapse final_score to 0.0."""
        bd = ConfidenceBreakdown(
            graph_match_score=0.9,
            auth_clarity_score=0.9,
            llm_self_score=0.0,
            evidence_count=10,
            rationale="LLM not confident",
        )
        assert bd.final_score == 0.0

    def test_zero_graph_match_returns_zero(self) -> None:
        bd = ConfidenceBreakdown(
            graph_match_score=0.0,
            auth_clarity_score=0.9,
            llm_self_score=0.9,
            evidence_count=5,
            rationale="weak pattern match",
        )
        assert bd.final_score == 0.0

    def test_zero_auth_clarity_returns_zero(self) -> None:
        bd = ConfidenceBreakdown(
            graph_match_score=0.9,
            auth_clarity_score=0.0,
            llm_self_score=0.9,
            evidence_count=5,
            rationale="auth undeclared",
        )
        assert bd.final_score == 0.0

    def test_evidence_bonus_at_five_calls(self) -> None:
        """5 evidence calls provides full +5% bonus."""
        bd = ConfidenceBreakdown(
            graph_match_score=0.8,
            auth_clarity_score=0.8,
            llm_self_score=0.8,
            evidence_count=5,
            rationale="full evidence",
        )
        base = statistics.geometric_mean([0.8, 0.8, 0.8])  # = 0.8
        expected = round(min(base + 0.05, 1.0), 4)
        assert bd.final_score == pytest.approx(expected, abs=1e-4)

    def test_evidence_bonus_caps_at_five(self) -> None:
        """10 evidence calls gives the same bonus as 5 — bonus is capped."""
        bd_5 = ConfidenceBreakdown(
            graph_match_score=0.8, auth_clarity_score=0.8, llm_self_score=0.8,
            evidence_count=5, rationale="r",
        )
        bd_10 = ConfidenceBreakdown(
            graph_match_score=0.8, auth_clarity_score=0.8, llm_self_score=0.8,
            evidence_count=10, rationale="r",
        )
        assert bd_5.final_score == bd_10.final_score

    def test_evidence_bonus_zero_at_zero_calls(self) -> None:
        bd_0 = ConfidenceBreakdown(
            graph_match_score=0.8, auth_clarity_score=0.8, llm_self_score=0.8,
            evidence_count=0, rationale="r",
        )
        bd_3 = ConfidenceBreakdown(
            graph_match_score=0.8, auth_clarity_score=0.8, llm_self_score=0.8,
            evidence_count=3, rationale="r",
        )
        assert bd_0.final_score < bd_3.final_score

    def test_final_score_never_exceeds_one(self) -> None:
        """High component scores plus evidence bonus must not exceed 1.0."""
        bd = ConfidenceBreakdown(
            graph_match_score=0.99,
            auth_clarity_score=0.99,
            llm_self_score=0.99,
            evidence_count=5,
            rationale="near-perfect",
        )
        assert bd.final_score <= 1.0

    def test_all_max_scores_with_evidence_caps_at_one(self) -> None:
        bd = ConfidenceBreakdown(
            graph_match_score=1.0,
            auth_clarity_score=1.0,
            llm_self_score=1.0,
            evidence_count=5,
            rationale="perfect",
        )
        assert bd.final_score == 1.0

    def test_component_scores_out_of_range_raise(self) -> None:
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(
                graph_match_score=1.1,
                auth_clarity_score=0.8,
                llm_self_score=0.8,
                evidence_count=0,
                rationale="r",
            )
        with pytest.raises(ValidationError):
            ConfidenceBreakdown(
                graph_match_score=0.8,
                auth_clarity_score=-0.1,
                llm_self_score=0.8,
                evidence_count=0,
                rationale="r",
            )

    def test_final_score_included_in_serialisation(self, sample_confidence: ConfidenceBreakdown) -> None:
        data = sample_confidence.model_dump()
        assert "final_score" in data
        assert isinstance(data["final_score"], float)

    def test_confidence_breakdown_is_frozen(self, sample_confidence: ConfidenceBreakdown) -> None:
        with pytest.raises(ValidationError):
            sample_confidence.llm_self_score = 0.5  # type: ignore[misc]


class TestAttackStepValidation:
    """AttackStep rejects empty or whitespace-only LLM output strings."""

    def test_valid_step_accepted(self, sample_attack_steps: list[AttackStep]) -> None:
        assert sample_attack_steps[0].sequence == 1
        assert sample_attack_steps[1].sequence == 2

    def test_empty_action_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AttackStep(
                sequence=1,
                endpoint_id="GET:/users",
                path="/users",
                method="GET",
                auth_required="None",
                action="",
                attacker_gains="Something",
                technique="Something",
            )
        assert "empty or whitespace-only" in str(exc_info.value)

    def test_whitespace_only_attacker_gains_raises(self) -> None:
        with pytest.raises(ValidationError):
            AttackStep(
                sequence=1,
                endpoint_id="GET:/x",
                path="/x",
                method="GET",
                auth_required="None",
                action="Do something",
                attacker_gains="   ",
                technique="Something",
            )

    def test_whitespace_only_technique_raises(self) -> None:
        with pytest.raises(ValidationError):
            AttackStep(
                sequence=1,
                endpoint_id="GET:/x",
                path="/x",
                method="GET",
                auth_required="None",
                action="Do something",
                attacker_gains="Gains something",
                technique="\t\n",
            )

    def test_step_sequence_below_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            AttackStep(
                sequence=0,
                endpoint_id="GET:/x",
                path="/x",
                method="GET",
                auth_required="None",
                action="action",
                attacker_gains="gains",
                technique="technique",
            )

    def test_attack_step_is_frozen(self, sample_attack_steps: list[AttackStep]) -> None:
        with pytest.raises(ValidationError):
            sample_attack_steps[0].action = "mutated"  # type: ignore[misc]


class TestValidatedChainMitreValidator:
    """MITRE ID format validator is the hallucination firewall."""

    @pytest.mark.parametrize("valid_id", [
        "T1589",
        "T1589.001",
        "T1078",
        "T1078.001",
        "T0000",
        "T9999.999",
    ])
    def test_valid_mitre_ids_accepted(
        self,
        valid_id: str,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        chain = ValidatedChain(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            pattern_id="AP-001",
            name="Test",
            severity=Severity.HIGH,
            confidence=sample_confidence,
            steps=sample_attack_steps,
            narrative=SAMPLE_NARRATIVE,
            mitre_techniques=[valid_id],
            owasp_category="API1:2023",
            remediation=["Fix it"],
            tool_calls_used=[],
            analyzed_at=datetime.now(tz=timezone.utc),
            llm_model="claude-sonnet-4-6",
            tokens_used=100,
        )
        assert valid_id in chain.mitre_techniques

    @pytest.mark.parametrize("invalid_id", [
        "T123",          # only 3 digits
        "T12345",        # 5 digits
        "T1234.56",      # sub-technique only 2 digits
        "T1234.5678",    # sub-technique 4 digits
        "1589.001",      # missing T prefix
        "t1589.001",     # lowercase t
        "T1589-001",     # hyphen instead of dot
        "T1589.001.002", # double sub-technique
        "TA0001",        # tactic ID, not technique
        "",              # empty string
        "T1234 ",        # trailing space
    ])
    def test_invalid_mitre_ids_raise(
        self,
        invalid_id: str,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ValidatedChain(
                id=str(uuid4()),
                candidate_id=str(uuid4()),
                pattern_id="AP-001",
                name="Test",
                severity=Severity.HIGH,
                confidence=sample_confidence,
                steps=sample_attack_steps,
                narrative=SAMPLE_NARRATIVE,
                mitre_techniques=[invalid_id],
                owasp_category="API1:2023",
                remediation=["Fix it"],
                tool_calls_used=[],
                analyzed_at=datetime.now(tz=timezone.utc),
                llm_model="claude-sonnet-4-6",
                tokens_used=100,
            )
        assert "MITRE" in str(exc_info.value) or "T1234" in str(exc_info.value)

    def test_empty_mitre_list_accepted(
        self,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        chain = ValidatedChain(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            pattern_id="AP-001",
            name="Test",
            severity=Severity.MEDIUM,
            confidence=sample_confidence,
            steps=sample_attack_steps,
            narrative=SAMPLE_NARRATIVE,
            mitre_techniques=[],
            owasp_category="API1:2023",
            remediation=["Fix it"],
            tool_calls_used=[],
            analyzed_at=datetime.now(tz=timezone.utc),
            llm_model="claude-sonnet-4-6",
            tokens_used=100,
        )
        assert chain.mitre_techniques == []


class TestValidatedChainNarrativeValidator:
    """Narrative must be at least 100 characters to prevent truncated LLM outputs."""

    def test_narrative_exactly_100_chars_accepted(
        self,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        narrative_100 = "A" * 100
        chain = ValidatedChain(
            id=str(uuid4()),
            candidate_id=str(uuid4()),
            pattern_id="AP-001",
            name="Test",
            severity=Severity.LOW,
            confidence=sample_confidence,
            steps=sample_attack_steps,
            narrative=narrative_100,
            mitre_techniques=[],
            owasp_category="API1:2023",
            remediation=["Fix it"],
            tool_calls_used=[],
            analyzed_at=datetime.now(tz=timezone.utc),
            llm_model="claude-sonnet-4-6",
            tokens_used=50,
        )
        assert len(chain.narrative) == 100

    def test_narrative_99_chars_raises(
        self,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            ValidatedChain(
                id=str(uuid4()),
                candidate_id=str(uuid4()),
                pattern_id="AP-001",
                name="Test",
                severity=Severity.LOW,
                confidence=sample_confidence,
                steps=sample_attack_steps,
                narrative="A" * 99,
                mitre_techniques=[],
                owasp_category="API1:2023",
                remediation=["Fix it"],
                tool_calls_used=[],
                analyzed_at=datetime.now(tz=timezone.utc),
                llm_model="claude-sonnet-4-6",
                tokens_used=50,
            )
        assert "too short" in str(exc_info.value)

    def test_empty_narrative_raises(
        self,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        with pytest.raises(ValidationError):
            ValidatedChain(
                id=str(uuid4()),
                candidate_id=str(uuid4()),
                pattern_id="AP-001",
                name="Test",
                severity=Severity.LOW,
                confidence=sample_confidence,
                steps=sample_attack_steps,
                narrative="",
                mitre_techniques=[],
                owasp_category="API1:2023",
                remediation=["Fix it"],
                tool_calls_used=[],
                analyzed_at=datetime.now(tz=timezone.utc),
                llm_model="claude-sonnet-4-6",
                tokens_used=50,
            )


class TestValidatedChainImmutability:
    def test_validated_chain_is_frozen(self, sample_validated_chain: ValidatedChain) -> None:
        with pytest.raises(ValidationError):
            sample_validated_chain.severity = Severity.CRITICAL  # type: ignore[misc]

    def test_steps_minimum_length_two_enforced(
        self,
        sample_confidence: ConfidenceBreakdown,
        sample_attack_steps: list[AttackStep],
    ) -> None:
        with pytest.raises(ValidationError):
            ValidatedChain(
                id=str(uuid4()),
                candidate_id=str(uuid4()),
                pattern_id="AP-001",
                name="Test",
                severity=Severity.LOW,
                confidence=sample_confidence,
                steps=[sample_attack_steps[0]],  # only one step
                narrative=SAMPLE_NARRATIVE,
                mitre_techniques=[],
                owasp_category="API1:2023",
                remediation=["Fix it"],
                tool_calls_used=[],
                analyzed_at=datetime.now(tz=timezone.utc),
                llm_model="claude-sonnet-4-6",
                tokens_used=50,
            )

    def test_remediation_minimum_length_one_enforced(
        self,
        sample_attack_steps: list[AttackStep],
        sample_confidence: ConfidenceBreakdown,
    ) -> None:
        with pytest.raises(ValidationError):
            ValidatedChain(
                id=str(uuid4()),
                candidate_id=str(uuid4()),
                pattern_id="AP-001",
                name="Test",
                severity=Severity.LOW,
                confidence=sample_confidence,
                steps=sample_attack_steps,
                narrative=SAMPLE_NARRATIVE,
                mitre_techniques=[],
                owasp_category="API1:2023",
                remediation=[],  # empty — must have at least one
                tool_calls_used=[],
                analyzed_at=datetime.now(tz=timezone.utc),
                llm_model="claude-sonnet-4-6",
                tokens_used=50,
            )


class TestCandidateChainConstruction:
    def test_valid_candidate_chain(self, sample_candidate_chain: CandidateChain) -> None:
        assert sample_candidate_chain.hop_count == 1
        assert sample_candidate_chain.crosses_auth_boundary is True
        assert sample_candidate_chain.sensitivity_delta == 2

    def test_endpoint_ids_minimum_two_enforced(self) -> None:
        with pytest.raises(ValidationError):
            CandidateChain(
                id=str(uuid4()),
                pattern_id="AP-001",
                pattern_name="Test",
                owasp_category="API1:2023",
                confidence_base=0.7,
                endpoint_ids=["GET:/single"],  # only one endpoint
                hop_count=0,
                entry_endpoint_id="GET:/single",
                exit_endpoint_id="GET:/single",
                crosses_auth_boundary=False,
                sensitivity_delta=0,
                rank_score=1.0,
                entry_summary="GET /single (PUBLIC)",
                exit_summary="GET /single (PUBLIC)",
            )

    def test_candidate_chain_is_frozen(self, sample_candidate_chain: CandidateChain) -> None:
        with pytest.raises(ValidationError):
            sample_candidate_chain.rank_score = 999.9  # type: ignore[misc]
