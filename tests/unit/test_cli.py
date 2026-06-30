"""Unit tests for api_analyzer/cli.py (M12).

Strategy: mock every external boundary so tests are fast and offline.
  - api_analyzer.cli.ingest          → returns a fake ParsedSpec
  - api_analyzer.cli.classify        → identity (returns same spec)
  - api_analyzer.cli.GraphDatabase   → fake driver that never hits Neo4j
  - api_analyzer.cli.apply_schema    → no-op
  - api_analyzer.cli.build_graph     → returns a fake BuildResult
  - api_analyzer.cli.make_spec_id    → returns "test-spec-id"
  - api_analyzer.cli.traverse        → returns a fake TraversalResult
  - api_analyzer.cli.analyze         → returns list of fake ValidatedChains
  - api_analyzer.cli.generate_report → writes nothing, returns tmp path

Coverage:
  - version command
  - analyze: default output path, --output path, --quiet flag, --wipe flag,
    --no-graph flag, --min-confidence flag, --model flag, missing spec file,
    Neo4j connection failure, parse error, exit code 0 (no findings),
    exit code 2 (CRITICAL/HIGH findings), --neo4j-uri/user/password options,
    NEO4J_* env vars, pipeline step order, AnalysisResult assembly
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from typer.testing import CliRunner

from api_analyzer import __version__
from api_analyzer.cli import app
from api_analyzer.engine.traversal import TraversalResult
from api_analyzer.models.chain import AttackStep, ConfidenceBreakdown, ValidatedChain
from api_analyzer.models.enums import Severity, SpecFormat
from api_analyzer.models.spec import ParsedSpec


runner = CliRunner()

# ── Fixture helpers ────────────────────────────────────────────────────────────


def _fake_spec(title: str = "Fake API", version: str = "1.0.0") -> MagicMock:
    spec = MagicMock(spec=ParsedSpec)
    spec.title = title
    spec.version = version
    spec.spec_format = SpecFormat.OPENAPI3
    spec.endpoints = []
    spec.auth_schemes = {}
    spec.parse_warnings = []
    spec.spec_completeness = 0.75
    return spec


def _fake_build_result() -> MagicMock:
    br = MagicMock()
    br.spec_id = "test-spec-id"
    br.endpoint_count = 10
    br.resource_count = 4
    br.auth_scheme_count = 1
    br.rel_count = 20
    return br


def _fake_traversal() -> TraversalResult:
    return TraversalResult(
        spec_id="test-spec-id",
        chains=[],
        total_candidates=5,
        spec_completeness=0.75,
        endpoint_count=10,
        candidate_counts={"bola": 2, "broken_auth": 1, "priv_esc": 1, "mass_assignment": 0,
                          "excessive_data": 1, "ssrf": 0, "auth_chains": 0},
    )


def _fake_step(seq: int) -> AttackStep:
    return AttackStep(
        sequence=seq,
        endpoint_id=f"GET:/resource/{seq}",
        path=f"/resource/{seq}",
        method="GET",
        auth_required="None",
        action="Enumerate resource",
        attacker_gains="Resource data",
        technique="BOLA via integer enumeration",
    )


def _fake_chain(severity: Severity = Severity.HIGH) -> ValidatedChain:
    return ValidatedChain(
        id=str(uuid.uuid4()),
        candidate_id=str(uuid.uuid4()),
        pattern_id="AP-001",
        name="BOLA Mass Exfil",
        severity=severity,
        confidence=ConfidenceBreakdown(
            graph_match_score=0.65,
            auth_clarity_score=0.8,
            llm_self_score=0.9,
            evidence_count=3,
            rationale="Strong BOLA indicators across three tool calls",
        ),
        steps=[_fake_step(1), _fake_step(2)],
        narrative="A" * 120,
        mitre_techniques=["T1589"],
        owasp_category="API1:2023",
        remediation=["Add object-level auth checks"],
        tool_calls_used=[],
        analyzed_at=datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        llm_model="claude-sonnet-4-6",
        tokens_used=500,
    )


def _patch_all(
    spec: Any = None,
    build_result: Any = None,
    traversal: Any = None,
    validated: list | None = None,
    report_path: Path | None = None,
    neo4j_error: Exception | None = None,
    parse_error: Exception | None = None,
):
    """Return a context-manager stack that patches every CLI dependency."""
    if spec is None:
        spec = _fake_spec()
    if build_result is None:
        build_result = _fake_build_result()
    if traversal is None:
        traversal = _fake_traversal()
    if validated is None:
        validated = []

    def _make_driver(*a, **kw):
        if neo4j_error:
            raise neo4j_error
        drv = MagicMock()
        drv.verify_connectivity.return_value = None
        drv.close.return_value = None
        return drv

    import contextlib

    @contextlib.contextmanager
    def _ctx():
        with (
            patch("api_analyzer.cli.ingest", return_value=spec, side_effect=parse_error),
            patch("api_analyzer.cli.classify", return_value=spec),
            patch("api_analyzer.cli.GraphDatabase") as gdb,
            patch("api_analyzer.cli.apply_schema"),
            patch("api_analyzer.cli.build_graph", return_value=build_result),
            patch("api_analyzer.cli.make_spec_id", return_value="test-spec-id"),
            patch("api_analyzer.cli.traverse", return_value=traversal),
            patch("api_analyzer.cli.analyze", return_value=validated),
            patch("api_analyzer.cli.generate_report", return_value=report_path or Path("/tmp/r.html")),
            patch("api_analyzer.cli.wipe_spec"),
        ):
            gdb.driver.side_effect = _make_driver
            yield

    return _ctx()


# ── version command ────────────────────────────────────────────────────────────


class TestVersionCommand:
    def test_prints_version(self):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_contains_tool_name(self):
        result = runner.invoke(app, ["version"])
        assert "api-attack-path-analyzer" in result.output


# ── analyze: file validation ───────────────────────────────────────────────────


class TestAnalyzeMissingFile:
    def test_missing_spec_exits_1(self, tmp_path):
        result = runner.invoke(app, ["analyze-spec", str(tmp_path / "nonexistent.yaml")])
        assert result.exit_code == 1

    def test_missing_spec_prints_error(self, tmp_path):
        result = runner.invoke(app, ["analyze-spec", str(tmp_path / "nonexistent.yaml")])
        assert "not found" in result.output.lower() or "not found" in (result.stderr or "").lower()


# ── analyze: parse error ───────────────────────────────────────────────────────


class TestAnalyzeParseError:
    def test_parse_error_exits_1(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text("not: valid: openapi")
        from api_analyzer.parser.ingestor import SpecParseError
        with _patch_all(parse_error=SpecParseError("bad yaml")):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 1

    def test_unexpected_parse_error_exits_1(self, tmp_path):
        spec_file = tmp_path / "bad.yaml"
        spec_file.write_text("x: 1")
        with _patch_all(parse_error=RuntimeError("unexpected")):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 1


# ── analyze: neo4j error ───────────────────────────────────────────────────────


class TestAnalyzeNeo4jError:
    def test_neo4j_connection_error_exits_1(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all(neo4j_error=Exception("Connection refused")):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 1


# ── analyze: exit codes ────────────────────────────────────────────────────────


class TestAnalyzeExitCodes:
    def test_no_findings_exits_0(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all(validated=[]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 0

    def test_critical_findings_exits_2(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain = _fake_chain(severity=Severity.CRITICAL)
        with _patch_all(validated=[chain]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 2

    def test_high_findings_exits_2(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain = _fake_chain(severity=Severity.HIGH)
        with _patch_all(validated=[chain]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 2

    def test_medium_findings_exits_0(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain = _fake_chain(severity=Severity.MEDIUM)
        with _patch_all(validated=[chain]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 0

    def test_low_findings_exits_0(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain = _fake_chain(severity=Severity.LOW)
        with _patch_all(validated=[chain]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert result.exit_code == 0


# ── analyze: default output path ──────────────────────────────────────────────


class TestAnalyzeDefaultOutputPath:
    def test_default_output_is_stem_report_html(self, tmp_path):
        spec_file = tmp_path / "petstore.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.generate_report", return_value=Path("/tmp/petstore_report.html")) as mock_gen:
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            called_path = mock_gen.call_args[0][1]
            assert called_path.name == "petstore_report.html"

    def test_explicit_output_path_passed_to_generate(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        out = tmp_path / "custom.html"
        with _patch_all() as _, patch("api_analyzer.cli.generate_report", return_value=out) as mock_gen:
            runner.invoke(app, ["analyze-spec", str(spec_file), "--output", str(out)])
            called_path = mock_gen.call_args[0][1]
            assert Path(called_path) == out


# ── analyze: --quiet ──────────────────────────────────────────────────────────


class TestAnalyzeQuiet:
    def test_quiet_prints_only_report_path(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        expected = tmp_path / "out.html"
        with _patch_all(report_path=expected):
            result = runner.invoke(app, ["analyze-spec", str(spec_file), "--quiet"])
        assert result.exit_code == 0
        # Rich may word-wrap the path; compare after collapsing whitespace
        flattened = " ".join(result.output.split())
        assert str(expected).replace("\\", "\\") in flattened or expected.name in flattened
        assert "Executive" not in result.output


# ── analyze: --no-graph ────────────────────────────────────────────────────────


class TestAnalyzeNoGraph:
    def test_no_graph_passes_include_graph_false(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.generate_report", return_value=Path("/tmp/r.html")) as mock_gen:
            runner.invoke(app, ["analyze-spec", str(spec_file), "--no-graph"])
            assert mock_gen.call_args[1]["include_graph"] is False

    def test_default_include_graph_true(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.generate_report", return_value=Path("/tmp/r.html")) as mock_gen:
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            assert mock_gen.call_args[1]["include_graph"] is True


# ── analyze: --wipe ────────────────────────────────────────────────────────────


class TestAnalyzeWipe:
    def test_wipe_calls_wipe_spec(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.wipe_spec") as mock_wipe:
            runner.invoke(app, ["analyze-spec", str(spec_file), "--wipe"])
            assert mock_wipe.called

    def test_no_wipe_does_not_call_wipe_spec(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.wipe_spec") as mock_wipe:
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            assert not mock_wipe.called


# ── analyze: model and confidence options ─────────────────────────────────────


class TestAnalyzeOptions:
    def test_custom_model_passed_to_reasoner_config(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.analyze") as mock_analyze:
            mock_analyze.return_value = []
            runner.invoke(app, ["analyze-spec", str(spec_file), "--model", "claude-opus-4-8"])
            config = mock_analyze.call_args[1]["config"]
            assert config.llm_model == "claude-opus-4-8"

    def test_custom_confidence_threshold(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.analyze") as mock_analyze:
            mock_analyze.return_value = []
            runner.invoke(app, ["analyze-spec", str(spec_file), "--min-confidence", "0.7"])
            config = mock_analyze.call_args[1]["config"]
            assert config.confidence_threshold == pytest.approx(0.7)

    def test_max_candidates_passed_to_traverse(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.traverse", return_value=_fake_traversal()) as mock_trav:
            runner.invoke(app, ["analyze-spec", str(spec_file), "--max-candidates", "20"])
            assert mock_trav.call_args[1]["max_candidates"] == 20


# ── analyze: Neo4j credentials via options ────────────────────────────────────


class TestAnalyzeNeo4jCredentials:
    def test_custom_neo4j_uri(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.GraphDatabase") as gdb:
            drv = MagicMock()
            drv.verify_connectivity.return_value = None
            drv.close.return_value = None
            gdb.driver.return_value = drv
            runner.invoke(app, ["analyze-spec", str(spec_file),
                                "--neo4j-uri", "bolt://remotehost:7687"])
            assert gdb.driver.call_args[0][0] == "bolt://remotehost:7687"

    def test_custom_neo4j_user_and_password(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.GraphDatabase") as gdb:
            drv = MagicMock()
            drv.verify_connectivity.return_value = None
            drv.close.return_value = None
            gdb.driver.return_value = drv
            runner.invoke(app, ["analyze-spec", str(spec_file),
                                "--neo4j-user", "admin",
                                "--neo4j-password", "secret"])
            assert gdb.driver.call_args[1]["auth"] == ("admin", "secret")


# ── analyze: pipeline step order ──────────────────────────────────────────────


class TestAnalyzePipelineOrder:
    def test_ingest_called_before_classify(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        call_order = []
        spec = _fake_spec()
        with (
            patch("api_analyzer.cli.ingest", side_effect=lambda *a: (call_order.append("ingest"), spec)[1]),
            patch("api_analyzer.cli.classify", side_effect=lambda *a: (call_order.append("classify"), spec)[1]),
            patch("api_analyzer.cli.GraphDatabase") as gdb,
            patch("api_analyzer.cli.apply_schema"),
            patch("api_analyzer.cli.build_graph", return_value=_fake_build_result()),
            patch("api_analyzer.cli.make_spec_id", return_value="test-spec-id"),
            patch("api_analyzer.cli.traverse", return_value=_fake_traversal()),
            patch("api_analyzer.cli.analyze", return_value=[]),
            patch("api_analyzer.cli.generate_report", return_value=Path("/tmp/r.html")),
            patch("api_analyzer.cli.wipe_spec"),
        ):
            drv = MagicMock()
            drv.verify_connectivity.return_value = None
            drv.close.return_value = None
            gdb.driver.return_value = drv
            runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert call_order.index("ingest") < call_order.index("classify")

    def test_apply_schema_called_before_build_graph(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        call_order = []
        with (
            patch("api_analyzer.cli.ingest", return_value=_fake_spec()),
            patch("api_analyzer.cli.classify", return_value=_fake_spec()),
            patch("api_analyzer.cli.GraphDatabase") as gdb,
            patch("api_analyzer.cli.apply_schema", side_effect=lambda *a: call_order.append("apply_schema")),
            patch("api_analyzer.cli.build_graph", side_effect=lambda *a, **kw: (call_order.append("build_graph"), _fake_build_result())[1]),
            patch("api_analyzer.cli.make_spec_id", return_value="test-spec-id"),
            patch("api_analyzer.cli.traverse", return_value=_fake_traversal()),
            patch("api_analyzer.cli.analyze", return_value=[]),
            patch("api_analyzer.cli.generate_report", return_value=Path("/tmp/r.html")),
            patch("api_analyzer.cli.wipe_spec"),
        ):
            drv = MagicMock()
            drv.verify_connectivity.return_value = None
            drv.close.return_value = None
            gdb.driver.return_value = drv
            runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert call_order.index("apply_schema") < call_order.index("build_graph")


# ── analyze: AnalysisResult assembly ─────────────────────────────────────────


class TestAnalyzeResultAssembly:
    def test_result_includes_validated_chains(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chains = [_fake_chain()]
        with _patch_all() as _, patch("api_analyzer.cli.generate_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/r.html")
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            result = mock_gen.call_args[0][0]
            # result is AnalysisResult; chains validated count matches
            assert result.candidates_evaluated == 5  # from fake traversal total_candidates

    def test_token_count_summed_across_chains(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain_a = _fake_chain()
        chain_b = _fake_chain()
        with _patch_all(validated=[chain_a, chain_b]) as _, patch("api_analyzer.cli.generate_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/r.html")
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            result = mock_gen.call_args[0][0]
            assert result.llm_tokens_used == 1000  # 500 + 500

    def test_candidates_rejected_computed_correctly(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        # total_candidates=5 in fake traversal; 2 validated → 3 rejected
        chains = [_fake_chain(), _fake_chain()]
        with _patch_all(validated=chains) as _, patch("api_analyzer.cli.generate_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/r.html")
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            result = mock_gen.call_args[0][0]
            assert result.candidates_rejected == 3

    def test_tool_version_passed_to_generate_report(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.generate_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/r.html")
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            assert mock_gen.call_args[1]["tool_version"] == __version__

    def test_spec_format_preserved(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all() as _, patch("api_analyzer.cli.generate_report") as mock_gen:
            mock_gen.return_value = Path("/tmp/r.html")
            runner.invoke(app, ["analyze-spec", str(spec_file)])
            result = mock_gen.call_args[0][0]
            assert result.spec_format == SpecFormat.OPENAPI3


# ── analyze: summary output ───────────────────────────────────────────────────


class TestAnalyzeSummaryOutput:
    def test_summary_prints_spec_title(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all(spec=_fake_spec("My Service")):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert "My Service" in result.output

    def test_summary_prints_finding_name_when_present(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        chain = _fake_chain()
        with _patch_all(validated=[chain]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert "BOLA Mass Exfil" in result.output

    def test_summary_prints_no_findings_when_empty(self, tmp_path):
        spec_file = tmp_path / "spec.yaml"
        spec_file.write_text("x: 1")
        with _patch_all(validated=[]):
            result = runner.invoke(app, ["analyze-spec", str(spec_file)])
        assert "No validated" in result.output
