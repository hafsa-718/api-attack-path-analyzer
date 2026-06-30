"""Shared pipeline logic used by both the CLI (M12) and FastAPI wrapper (M13).

Runs the full analysis:
  parse → classify → build graph → traverse → LLM validate → assemble result

The caller owns the Neo4j driver lifecycle (open before, close after).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from neo4j import Driver

from api_analyzer import __version__
from api_analyzer.agent.reasoner import ReasonerConfig, analyze
from api_analyzer.engine.prober import probe_all
from api_analyzer.engine.traversal import traverse
from api_analyzer.graph.builder import build_graph
from api_analyzer.graph.schema import apply_schema, make_spec_id, wipe_spec
from api_analyzer.models.report import AnalysisResult
from api_analyzer.parser.classifier import classify
from api_analyzer.parser.ingestor import ingest

# Sonnet 4.6 pricing per token (2026-06)
_COST_INPUT: float = 3.0 / 1_000_000
_COST_OUTPUT: float = 15.0 / 1_000_000


def estimate_cost(tokens: int) -> float:
    """Approximate cost assuming 60 % input / 40 % output token split."""
    return tokens * (0.6 * _COST_INPUT + 0.4 * _COST_OUTPUT)


@dataclass
class PipelineConfig:
    model: str = "claude-sonnet-4-6"
    max_candidates: int = 50
    confidence_threshold: float = 0.4
    wipe_before_build: bool = False
    max_workers: int = 5
    target_url: str | None = None  # When set, probe entry endpoints after LLM validation


def run_pipeline(
    spec_path: Path | str,
    driver: Driver,
    *,
    config: PipelineConfig | None = None,
) -> AnalysisResult:
    """Execute the full analysis pipeline and return an AnalysisResult.

    The caller must pass an already-connected ``driver``.  The driver is NOT
    closed by this function — the caller owns its lifecycle.

    ``apply_schema`` is called internally so it is safe to call ``run_pipeline``
    on a fresh database.
    """
    if config is None:
        config = PipelineConfig()

    t_start = perf_counter()

    # Parse + classify
    spec = ingest(spec_path)
    spec = classify(spec)

    # Graph
    apply_schema(driver)
    if config.wipe_before_build:
        wipe_spec(driver, make_spec_id(spec.title, spec.version))
    build_result = build_graph(spec, driver)
    spec_id = make_spec_id(spec.title, spec.version)

    # Traverse + reason
    traversal = traverse(driver, spec_id, max_candidates=config.max_candidates)
    reasoner_config = ReasonerConfig(
        llm_model=config.model,
        confidence_threshold=config.confidence_threshold,
        max_workers=config.max_workers,
    )
    validated_chains = analyze(traversal, driver, config=reasoner_config)

    # Optional runtime probe — fires real HTTP requests to confirm entry endpoints
    if config.target_url and validated_chains:
        probe_map = probe_all(
            config.target_url,
            validated_chains,
            max_workers=config.max_workers,
        )
        validated_chains = [
            c.model_copy(update={"probe_result": probe_map.get(c.id)})
            for c in validated_chains
        ]

    t_end = perf_counter()
    total_tokens = sum(c.tokens_used for c in validated_chains)

    return AnalysisResult(
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
        graph_node_count=(
            build_result.endpoint_count
            + build_result.resource_count
            + build_result.auth_scheme_count
            + 1  # ApiSpec node
        ),
        graph_edge_count=build_result.rel_count,
        patterns_run=list(traversal.candidate_counts.keys()),
        candidates_evaluated=traversal.total_candidates,
        candidates_rejected=traversal.total_candidates - len(validated_chains),
        llm_tokens_used=total_tokens,
        estimated_cost_usd=estimate_cost(total_tokens),
        parse_warnings=list(spec.parse_warnings),
    )
