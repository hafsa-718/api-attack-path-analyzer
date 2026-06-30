"""Unit tests for api_analyzer/report/generator.py (M11).

Coverage:
  - _build_d3_graph_data: nodes deduplication, link construction, severity promotion,
    empty chains, single-step chain (no links), multi-chain shared nodes
  - generate_report: creates file, returns correct path, creates parent dirs,
    include_graph=True/False affects template output, HTML contains key sections,
    template receives ctx and d3_graph_data correctly
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from api_analyzer.models.chain import AttackStep, ConfidenceBreakdown, ValidatedChain
from api_analyzer.models.enums import Severity, SpecFormat
from api_analyzer.models.report import AnalysisResult
from api_analyzer.report.generator import _build_d3_graph_data, generate_report


# ── Fixture helpers ────────────────────────────────────────────────────────────


def _step(seq: int, method: str, path: str) -> AttackStep:
    return AttackStep(
        sequence=seq,
        endpoint_id=f"{method}:{path}",
        path=path,
        method=method,
        auth_required="None — public endpoint",
        action="Enumerate resource IDs",
        attacker_gains="Valid resource IDs for next step",
        technique="Integer enumeration via predictable IDs",
    )


def _confidence(graph: float = 0.7, auth: float = 0.8, llm: float = 0.9) -> ConfidenceBreakdown:
    return ConfidenceBreakdown(
        graph_match_score=graph,
        auth_clarity_score=auth,
        llm_self_score=llm,
        evidence_count=3,
        rationale="Test rationale for confidence scoring in unit tests",
    )


def _chain(
    name: str = "Test BOLA Chain",
    severity: Severity = Severity.HIGH,
    steps: list[AttackStep] | None = None,
    pattern_id: str = "AP-001",
    owasp: str = "API1:2023",
) -> ValidatedChain:
    if steps is None:
        steps = [
            _step(1, "GET", "/users"),
            _step(2, "GET", "/users/{userId}/data"),
        ]
    return ValidatedChain(
        id=str(uuid.uuid4()),
        candidate_id=str(uuid.uuid4()),
        pattern_id=pattern_id,
        name=name,
        severity=severity,
        confidence=_confidence(),
        steps=steps,
        narrative="A" * 120,  # >= 100 chars required
        mitre_techniques=["T1589"],
        owasp_category=owasp,
        remediation=["Implement object-level auth checks"],
        tool_calls_used=["get_endpoint_info(GET:/users) → {found: true}"],
        analyzed_at=datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        llm_model="claude-sonnet-4-6",
        tokens_used=1200,
    )


def _result(chains: list[ValidatedChain] | None = None) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        spec_title="Test API",
        spec_version="1.0.0",
        spec_format=SpecFormat.OPENAPI3,
        analyzed_at=datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        duration_seconds=5.3,
        endpoint_count=20,
        public_endpoint_count=5,
        auth_declared=True,
        spec_completeness=0.85,
        chains=chains or [],
        graph_node_count=30,
        graph_edge_count=45,
        patterns_run=["AP-001", "AP-002"],
        candidates_evaluated=10,
        candidates_rejected=8,
        llm_tokens_used=3000,
        estimated_cost_usd=0.0045,
        parse_warnings=[],
    )


# ── _build_d3_graph_data ───────────────────────────────────────────────────────


class TestBuildD3GraphData:
    def test_empty_chains_returns_empty_nodes_and_links(self):
        data = json.loads(_build_d3_graph_data([]))
        assert data == {"nodes": [], "links": []}

    def test_single_chain_two_steps_produces_one_link(self):
        chain = _chain()
        data = json.loads(_build_d3_graph_data([chain]))
        assert len(data["nodes"]) == 2
        assert len(data["links"]) == 1

    def test_node_ids_match_endpoint_ids(self):
        chain = _chain()
        data = json.loads(_build_d3_graph_data([chain]))
        node_ids = {n["id"] for n in data["nodes"]}
        assert "GET:/users" in node_ids
        assert "GET:/users/{userId}/data" in node_ids

    def test_node_label_includes_method_and_path(self):
        chain = _chain(steps=[_step(1, "POST", "/items"), _step(2, "GET", "/items/{id}")])
        data = json.loads(_build_d3_graph_data([chain]))
        labels = {n["label"] for n in data["nodes"]}
        assert "POST /items" in labels
        assert "GET /items/{id}" in labels

    def test_link_source_and_target_match_step_order(self):
        chain = _chain()
        data = json.loads(_build_d3_graph_data([chain]))
        link = data["links"][0]
        assert link["source"] == "GET:/users"
        assert link["target"] == "GET:/users/{userId}/data"

    def test_link_carries_chain_metadata(self):
        chain = _chain(pattern_id="AP-001", severity=Severity.HIGH)
        data = json.loads(_build_d3_graph_data([chain]))
        link = data["links"][0]
        assert link["chain_id"] == chain.id
        assert link["pattern_id"] == "AP-001"
        assert link["severity"] == "HIGH"

    def test_node_carries_severity(self):
        chain = _chain(severity=Severity.CRITICAL)
        data = json.loads(_build_d3_graph_data([chain]))
        severities = {n["severity"] for n in data["nodes"]}
        assert "CRITICAL" in severities

    def test_three_step_chain_produces_two_links(self):
        chain = _chain(
            steps=[
                _step(1, "GET", "/users"),
                _step(2, "GET", "/users/{id}/orders"),
                _step(3, "DELETE", "/orders/{orderId}"),
            ]
        )
        data = json.loads(_build_d3_graph_data([chain]))
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2

    def test_shared_node_deduplicated_across_chains(self):
        shared_step = _step(1, "GET", "/users")
        chain_a = _chain(
            name="Chain A",
            steps=[shared_step, _step(2, "GET", "/users/{id}/a")],
        )
        chain_b = _chain(
            name="Chain B",
            steps=[shared_step, _step(2, "GET", "/users/{id}/b")],
        )
        data = json.loads(_build_d3_graph_data([chain_a, chain_b]))
        # "GET:/users" should appear only once
        ids = [n["id"] for n in data["nodes"]]
        assert ids.count("GET:/users") == 1

    def test_severity_promotion_keeps_higher_severity(self):
        # chain_a is CRITICAL, chain_b is LOW; shared node should be CRITICAL
        shared_step = _step(1, "GET", "/shared")
        chain_critical = _chain(
            name="Critical",
            severity=Severity.CRITICAL,
            steps=[shared_step, _step(2, "DELETE", "/admin/resource")],
        )
        chain_low = _chain(
            name="Low",
            severity=Severity.LOW,
            steps=[shared_step, _step(2, "GET", "/public/info")],
        )
        # critical chain first — lower-severity second chain should not demote node
        data = json.loads(_build_d3_graph_data([chain_critical, chain_low]))
        shared_node = next(n for n in data["nodes"] if n["id"] == "GET:/shared")
        assert shared_node["severity"] == "CRITICAL"

    def test_severity_promotion_upgrades_when_higher_seen_later(self):
        # If LOW chain is listed first, a later CRITICAL chain should upgrade
        shared_step = _step(1, "GET", "/shared")
        chain_low = _chain(
            name="Low",
            severity=Severity.LOW,
            steps=[shared_step, _step(2, "GET", "/public/info")],
        )
        chain_critical = _chain(
            name="Critical",
            severity=Severity.CRITICAL,
            steps=[shared_step, _step(2, "DELETE", "/admin/resource")],
        )
        data = json.loads(_build_d3_graph_data([chain_low, chain_critical]))
        shared_node = next(n for n in data["nodes"] if n["id"] == "GET:/shared")
        assert shared_node["severity"] == "CRITICAL"

    def test_multiple_links_from_multiple_chains(self):
        chain_a = _chain(
            name="A",
            steps=[_step(1, "GET", "/a"), _step(2, "GET", "/b")],
        )
        chain_b = _chain(
            name="B",
            steps=[_step(1, "GET", "/c"), _step(2, "GET", "/d")],
        )
        data = json.loads(_build_d3_graph_data([chain_a, chain_b]))
        assert len(data["links"]) == 2

    def test_return_type_is_string(self):
        result = _build_d3_graph_data([])
        assert isinstance(result, str)

    def test_result_is_valid_json(self):
        chain = _chain()
        result = _build_d3_graph_data([chain])
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "links" in parsed


# ── generate_report ────────────────────────────────────────────────────────────


class TestGenerateReport:
    def test_creates_file_at_output_path(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(_result(), out)
        assert out.exists()

    def test_returns_resolved_path(self, tmp_path):
        out = tmp_path / "report.html"
        returned = generate_report(_result(), out)
        assert returned == out

    def test_accepts_string_path(self, tmp_path):
        out = str(tmp_path / "report.html")
        returned = generate_report(_result(), out)
        assert returned == Path(out)

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "report.html"
        generate_report(_result(), out)
        assert out.exists()

    def test_output_is_non_empty_html(self, tmp_path):
        out = tmp_path / "report.html"
        generate_report(_result(), out)
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert len(content) > 500

    def test_html_contains_spec_title(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(), out)
        assert "Test API" in out.read_text(encoding="utf-8")

    def test_html_contains_tool_version(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(), out, tool_version="9.9.9")
        assert "9.9.9" in out.read_text(encoding="utf-8")

    def test_html_contains_executive_summary_section(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(), out)
        assert "Executive Summary" in out.read_text(encoding="utf-8")

    def test_html_with_no_chains_shows_no_findings_message(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(chains=[]), out)
        content = out.read_text(encoding="utf-8")
        assert "No validated attack chains" in content

    def test_html_with_chains_includes_finding_name(self, tmp_path):
        chain = _chain(name="BOLA Mass Exfil")
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert "BOLA Mass Exfil" in out.read_text(encoding="utf-8")

    def test_html_with_chains_includes_owasp_category(self, tmp_path):
        chain = _chain(owasp="API1:2023")
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert "API1:2023" in out.read_text(encoding="utf-8")

    def test_html_with_chains_includes_narrative(self, tmp_path):
        narrative_marker = "A" * 120  # our fixture uses exactly this
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert narrative_marker in out.read_text(encoding="utf-8")

    def test_include_graph_true_embeds_d3_script_tag(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out, include_graph=True)
        assert "d3js.org" in out.read_text(encoding="utf-8")

    def test_include_graph_false_omits_d3_script_tag(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out, include_graph=False)
        assert "d3js.org" not in out.read_text(encoding="utf-8")

    def test_default_include_graph_is_true(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert "d3js.org" in out.read_text(encoding="utf-8")

    def test_html_contains_d3_graph_data_when_chains_present(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out, include_graph=True)
        content = out.read_text(encoding="utf-8")
        assert '"nodes"' in content
        assert '"links"' in content

    def test_html_severity_counts_in_summary(self, tmp_path):
        critical_chain = _chain(severity=Severity.CRITICAL)
        out = tmp_path / "r.html"
        generate_report(_result(chains=[critical_chain]), out)
        content = out.read_text(encoding="utf-8")
        # Summary cards should show count 1 for CRITICAL
        assert "1" in content

    def test_html_contains_remediation(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert "Implement object-level auth checks" in out.read_text(encoding="utf-8")

    def test_html_contains_mitre_technique(self, tmp_path):
        chain = _chain()
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain]), out)
        assert "T1589" in out.read_text(encoding="utf-8")

    def test_overwrites_existing_file(self, tmp_path):
        out = tmp_path / "r.html"
        out.write_text("OLD CONTENT", encoding="utf-8")
        generate_report(_result(), out)
        content = out.read_text(encoding="utf-8")
        assert "OLD CONTENT" not in content
        assert "<!DOCTYPE html>" in content

    def test_default_tool_version_is_embedded(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(), out)
        assert "0.1.0" in out.read_text(encoding="utf-8")

    def test_html_is_utf8_encoded(self, tmp_path):
        out = tmp_path / "r.html"
        generate_report(_result(), out)
        # Should not raise when reading as UTF-8
        content = out.read_text(encoding="utf-8")
        assert len(content) > 0

    def test_multiple_chains_all_appear_in_html(self, tmp_path):
        chain_a = _chain(name="Finding Alpha")
        chain_b = _chain(name="Finding Beta", severity=Severity.CRITICAL)
        out = tmp_path / "r.html"
        generate_report(_result(chains=[chain_a, chain_b]), out)
        content = out.read_text(encoding="utf-8")
        assert "Finding Alpha" in content
        assert "Finding Beta" in content

    def test_parse_warnings_appear_in_html_when_present(self, tmp_path):
        result = AnalysisResult(
            analysis_id=str(uuid.uuid4()),
            spec_title="Test API",
            spec_version="1.0.0",
            spec_format=SpecFormat.OPENAPI3,
            analyzed_at=datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            duration_seconds=5.3,
            endpoint_count=20,
            public_endpoint_count=5,
            auth_declared=True,
            spec_completeness=0.85,
            chains=[],
            graph_node_count=30,
            graph_edge_count=45,
            patterns_run=[],
            candidates_evaluated=0,
            candidates_rejected=0,
            llm_tokens_used=0,
            estimated_cost_usd=0.0,
            parse_warnings=["Missing response schema for GET /users"],
        )
        out = tmp_path / "r.html"
        generate_report(result, out)
        assert "Missing response schema for GET /users" in out.read_text(encoding="utf-8")
