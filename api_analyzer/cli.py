"""Typer CLI for the API Attack Path Analyzer.

Entry point: ``api-analyzer`` (declared in pyproject.toml [project.scripts]).

Commands
--------
  analyze   Full pipeline: parse → classify → graph → traverse → reason → report
  version   Print version and exit
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Optional

import typer
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).parent.parent / ".env")
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from api_analyzer import __version__
from api_analyzer.agent.reasoner import ReasonerConfig, analyze
from api_analyzer.graph.builder import build_graph
from api_analyzer.graph.schema import apply_schema, make_spec_id, wipe_spec
from api_analyzer.models.enums import Severity
from api_analyzer.models.report import AnalysisResult
from api_analyzer.parser.classifier import classify
from api_analyzer.parser.ingestor import SpecParseError, ingest
from api_analyzer.report.generator import generate_report
from api_analyzer.engine.traversal import traverse

app = typer.Typer(
    name="api-analyzer",
    help="AI-powered API attack path analysis using Neo4j + Claude.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

_console = Console(stderr=True)
_out = Console()

_SEVERITY_COLORS: dict[str, str] = {
    "CRITICAL": "bold red",
    "HIGH": "red",
    "MEDIUM": "yellow",
    "LOW": "blue",
    "INFO": "dim",
}

# Sonnet 4.6 pricing (per 1M tokens, as of 2026-06)
_COST_PER_INPUT_TOKEN: float = 3.0 / 1_000_000
_COST_PER_OUTPUT_TOKEN: float = 15.0 / 1_000_000


def _estimate_cost(tokens: int) -> float:
    # Approximate: assume 60% input / 40% output split
    return tokens * (0.6 * _COST_PER_INPUT_TOKEN + 0.4 * _COST_PER_OUTPUT_TOKEN)


def _print_summary(result: AnalysisResult, report_path: Path | None) -> None:
    """Print a Rich summary table to stdout."""
    sev = result.highest_severity
    sev_color = _SEVERITY_COLORS.get(sev.value if sev else "", "dim")

    panel_title = (
        f"[{sev_color}]{sev.value}[/{sev_color}] — {result.chain_count} finding(s)"
        if result.chain_count
        else "No validated attack chains found"
    )
    _out.print(Panel(panel_title, title="[bold]API Attack Path Analyzer[/bold]", expand=False))

    # Stats table
    stats = Table(show_header=False, box=None, padding=(0, 1))
    stats.add_column(style="dim", width=26)
    stats.add_column()
    stats.add_row("Spec", f"{result.spec_title} v{result.spec_version}")
    stats.add_row("Endpoints", str(result.endpoint_count))
    stats.add_row("Spec completeness", f"{result.spec_completeness * 100:.0f}%")
    stats.add_row("Candidates evaluated", str(result.candidates_evaluated))
    stats.add_row("Candidates rejected", str(result.candidates_rejected))
    stats.add_row("Duration", f"{result.duration_seconds:.1f}s")
    stats.add_row("Tokens used", str(result.llm_tokens_used))
    stats.add_row("Estimated cost", f"${result.estimated_cost_usd:.4f}")
    if report_path:
        stats.add_row("Report", str(report_path))
    _out.print(stats)

    if not result.chains:
        return

    # Findings table
    _out.print()
    findings = Table(
        "#", "Name", "Severity", "OWASP", "Confidence",
        title="Findings", show_lines=False,
    )
    for i, chain in enumerate(result.chains, 1):
        color = _SEVERITY_COLORS.get(chain.severity.value, "")
        findings.add_row(
            str(i),
            chain.name,
            f"[{color}]{chain.severity.value}[/{color}]",
            chain.owasp_category,
            f"{chain.confidence.final_score * 100:.0f}%",
        )
    _out.print(findings)


# ── Commands ───────────────────────────────────────────────────────────────────


@app.command()
def analyze_spec(
    spec_path: Annotated[Path, typer.Argument(help="Path to OpenAPI/Swagger spec (YAML or JSON)")],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="HTML report output path (default: <spec_stem>_report.html)"),
    ] = None,
    neo4j_uri: Annotated[
        str,
        typer.Option("--neo4j-uri", envvar="NEO4J_URI", help="Neo4j bolt URI"),
    ] = "bolt://localhost:7687",
    neo4j_user: Annotated[
        str,
        typer.Option("--neo4j-user", envvar="NEO4J_USER", help="Neo4j username"),
    ] = "neo4j",
    neo4j_password: Annotated[
        str,
        typer.Option("--neo4j-password", envvar="NEO4J_PASSWORD", help="Neo4j password"),
    ] = "attackpath",
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Claude model ID"),
    ] = "claude-sonnet-4-6",
    max_candidates: Annotated[
        int,
        typer.Option("--max-candidates", help="Max attack chains passed to LLM"),
    ] = 50,
    confidence_threshold: Annotated[
        float,
        typer.Option("--min-confidence", help="Minimum final confidence score (0–1)"),
    ] = 0.4,
    no_graph: Annotated[
        bool,
        typer.Option("--no-graph", help="Omit D3.js attack path graph from HTML report"),
    ] = False,
    wipe: Annotated[
        bool,
        typer.Option("--wipe", help="Wipe previous data for this spec from Neo4j before analysis"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output; print only the report path"),
    ] = False,
) -> None:
    """Analyze an API specification for attack paths.

    Runs the full pipeline:
      1. Parse & classify the OpenAPI/Swagger spec
      2. Build a Neo4j knowledge graph
      3. Traverse the graph with OWASP-pattern queries
      4. Validate candidates with Claude LLM reasoning
      5. Render an HTML report with D3.js attack path visualization
    """
    if not spec_path.exists():
        _console.print(f"[red]Error:[/red] spec file not found: {spec_path}")
        raise typer.Exit(1)

    if output is None:
        output = spec_path.parent / f"{spec_path.stem}_report.html"

    t_start = perf_counter()

    progress_kwargs: dict = {
        "SpinnerColumn": SpinnerColumn(),
        "TextColumn": TextColumn("[progress.description]{task.description}"),
        "console": _console,
        "disable": quiet,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=_console,
        disable=quiet,
    ) as progress:

        # Step 1 — Parse
        task = progress.add_task("Parsing specification…", total=None)
        try:
            spec = ingest(spec_path)
        except SpecParseError as exc:
            _console.print(f"[red]Parse error:[/red] {exc}")
            raise typer.Exit(1) from exc
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]Unexpected parse error:[/red] {exc}")
            raise typer.Exit(1) from exc
        progress.update(task, description=f"Parsed {len(spec.endpoints)} endpoints")

        # Step 2 — Classify
        progress.update(task, description="Classifying endpoints…")
        spec = classify(spec)

        # Step 3 — Build graph
        progress.update(task, description="Connecting to Neo4j…")
        try:
            driver = GraphDatabase.driver(
                neo4j_uri, auth=(neo4j_user, neo4j_password)
            )
            driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001
            _console.print(f"[red]Neo4j connection error:[/red] {exc}")
            _console.print(
                "[dim]Tip: set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD env vars "
                "or use --neo4j-uri / --neo4j-user / --neo4j-password[/dim]"
            )
            raise typer.Exit(1) from exc

        progress.update(task, description="Applying graph schema…")
        apply_schema(driver)

        if wipe:
            progress.update(task, description="Wiping previous data…")
            wipe_spec(driver, make_spec_id(spec.title, spec.version))

        progress.update(task, description="Building knowledge graph…")
        build_result = build_graph(spec, driver)
        spec_id = make_spec_id(spec.title, spec.version)

        # Step 4 — Traverse
        progress.update(task, description="Traversing attack paths…")
        traversal = traverse(driver, spec_id, max_candidates=max_candidates)

        # Step 5 — Reason
        progress.update(
            task,
            description=f"Validating {traversal.total_candidates} candidates with {model}…",
        )
        config = ReasonerConfig(
            llm_model=model,
            confidence_threshold=confidence_threshold,
        )
        validated_chains = analyze(traversal, driver, config=config)

        progress.update(task, description="Generating report…")

    # ── Assemble AnalysisResult ────────────────────────────────────────────────
    t_end = perf_counter()
    total_tokens = sum(c.tokens_used for c in validated_chains)

    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        spec_title=spec.title,
        spec_version=spec.version,
        spec_format=spec.spec_format,
        analyzed_at=datetime.now(tz=timezone.utc),
        duration_seconds=round(t_end - t_start, 2),
        endpoint_count=len(spec.endpoints),
        public_endpoint_count=sum(1 for ep in spec.endpoints if ep.is_public),
        auth_declared=bool(spec.auth_schemes),
        spec_completeness=traversal.spec_completeness,
        chains=validated_chains,
        graph_node_count=build_result.endpoint_count + build_result.resource_count + build_result.auth_scheme_count + 1,
        graph_edge_count=build_result.rel_count,
        patterns_run=list(traversal.candidate_counts.keys()),
        candidates_evaluated=traversal.total_candidates,
        candidates_rejected=traversal.total_candidates - len(validated_chains),
        llm_tokens_used=total_tokens,
        estimated_cost_usd=_estimate_cost(total_tokens),
        parse_warnings=list(spec.parse_warnings),
    )

    driver.close()

    # ── Render report ──────────────────────────────────────────────────────────
    report_path = generate_report(
        result, output, tool_version=__version__, include_graph=not no_graph
    )

    # ── Output ─────────────────────────────────────────────────────────────────
    if quiet:
        _out.print(str(report_path))
    else:
        _print_summary(result, report_path)

    # Non-zero exit when critical or high findings present
    if result.highest_severity in (Severity.CRITICAL, Severity.HIGH):
        raise typer.Exit(2)


@app.command()
def version() -> None:
    """Print the tool version and exit."""
    _out.print(f"api-attack-path-analyzer {__version__}")
