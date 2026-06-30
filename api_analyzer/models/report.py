"""Data models for analysis results and report generation.

These models are produced at the end of the analysis pipeline and consumed
by the report generator (M11) and the optional API wrapper (M13).

All report models are immutable (frozen=True) — they represent a completed
analysis snapshot and must not be mutated after construction.

Module dependency: imports from api_analyzer.models.chain and .enums.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, computed_field

from api_analyzer.models.chain import ValidatedChain
from api_analyzer.models.enums import Severity, SpecFormat


class ChainSummary(BaseModel):
    """Compact representation of a validated chain for the executive summary section.

    Contains only the fields needed for the summary table — no narrative, no
    step details, no full confidence breakdown.  Using a separate model for the
    executive view enforces at the type level that summary templates cannot
    accidentally render technical detail intended for the full report.

    Constructed via ``ChainSummary.from_chain`` rather than directly, so the
    derivation logic stays in one place.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0, description="final_score from ConfidenceBreakdown")
    entry: str = Field(description="Human-readable entry point, e.g. 'GET /users (PUBLIC)'")
    exit_point: str = Field(
        description="Human-readable exit point, e.g. 'DELETE /admin/users/{id} (CRITICAL)'"
    )
    hop_count: int = Field(ge=1, description="Number of edges traversed in the chain")
    mitre_ids: list[str] = Field(default_factory=list)
    owasp: str

    @classmethod
    def from_chain(cls, chain: ValidatedChain) -> ChainSummary:
        """Derive a compact summary from a full ValidatedChain.

        Entry and exit strings are derived from the first and last AttackStep.
        ``hop_count`` is ``len(steps) - 1`` (edges, not nodes), consistent with
        ``CandidateChain.hop_count``.

        If steps is empty (which ValidatedChain's min_length=2 constraint
        prevents in practice), falls back to safe placeholder strings rather
        than raising, so a corrupt finding does not crash report generation.
        """
        entry_step = chain.steps[0] if chain.steps else None
        exit_step = chain.steps[-1] if chain.steps else None

        return cls(
            id=chain.id,
            name=chain.name,
            severity=chain.severity,
            confidence=chain.confidence.final_score,
            entry=f"{entry_step.method} {entry_step.path}" if entry_step else "unknown",
            exit_point=f"{exit_step.method} {exit_step.path}" if exit_step else "unknown",
            hop_count=max(len(chain.steps) - 1, 1),
            mitre_ids=list(chain.mitre_techniques),
            owasp=chain.owasp_category,
        )


class AnalysisResult(BaseModel):
    """Top-level output of the complete analysis pipeline.

    Produced after the reasoning agent (M10) has validated all candidate chains.
    Consumed by the report generator (M11), the CLI summary printer, and the
    optional REST API (M13).

    Computed fields
    ---------------
    ``chain_count``, ``critical_count``, ``high_count``, ``medium_count``,
    ``low_count``, and ``highest_severity`` are derived from ``chains`` and
    cannot be set independently.  This prevents the common report-generation
    bug where a severity count is manually incremented and falls out of sync
    with the actual chain list.

    Diagnostic fields
    -----------------
    ``candidates_evaluated`` and ``candidates_rejected`` are the primary signal
    for evaluating tool behaviour on a new spec:

      - High rejection rate (>60%) → either a well-secured API, or patterns
        are too broad and producing low-quality candidates.
      - Zero rejections → either many genuine vulnerabilities, or the LLM
        agent is over-validating.  Warrants manual review of the chains.
    """

    model_config = ConfigDict(frozen=True)

    analysis_id: str = Field(description="UUID generated at analysis start")
    spec_title: str
    spec_version: str
    spec_format: SpecFormat
    analyzed_at: datetime = Field(description="UTC timestamp when analysis completed")
    duration_seconds: float = Field(ge=0.0)
    endpoint_count: int = Field(ge=0)
    public_endpoint_count: int = Field(ge=0)
    auth_declared: bool
    spec_completeness: float = Field(ge=0.0, le=1.0)
    chains: list[ValidatedChain] = Field(default_factory=list)
    graph_node_count: int = Field(ge=0, description="Total nodes in the Neo4j knowledge graph")
    graph_edge_count: int = Field(ge=0, description="Total edges in the Neo4j knowledge graph")
    patterns_run: list[str] = Field(
        default_factory=list,
        description="Pattern IDs executed during this analysis run",
    )
    candidates_evaluated: int = Field(
        ge=0,
        description="CandidateChains passed to the LLM for validation",
    )
    candidates_rejected: int = Field(
        ge=0,
        description="Candidates the LLM agent determined were not valid findings",
    )
    llm_tokens_used: int = Field(ge=0, description="Total tokens consumed across all chain analyses")
    estimated_cost_usd: float = Field(
        ge=0.0,
        description="Estimated LLM cost in USD, calculated by the reasoning agent (M10)",
    )
    parse_warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings propagated from ParsedSpec.parse_warnings",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chain_count(self) -> int:
        """Total number of validated attack chains."""
        return len(self.chains)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def critical_count(self) -> int:
        """Number of CRITICAL severity chains."""
        return sum(1 for c in self.chains if c.severity == Severity.CRITICAL)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def high_count(self) -> int:
        """Number of HIGH severity chains."""
        return sum(1 for c in self.chains if c.severity == Severity.HIGH)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def medium_count(self) -> int:
        """Number of MEDIUM severity chains."""
        return sum(1 for c in self.chains if c.severity == Severity.MEDIUM)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def low_count(self) -> int:
        """Number of LOW severity chains."""
        return sum(1 for c in self.chains if c.severity == Severity.LOW)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def highest_severity(self) -> Severity | None:
        """The highest (most severe) level found among all validated chains.

        Uses the Severity enum's declaration order: CRITICAL(0) is the highest
        and INFO(4) is the lowest.  Returns None when the chains list is empty
        so callers can distinguish 'no findings' from 'INFO-only findings'.
        """
        if not self.chains:
            return None
        severity_order = list(Severity)
        return min(
            (c.severity for c in self.chains),
            key=lambda s: severity_order.index(s),
        )


class ReportContext(BaseModel):
    """Context object passed to Jinja2 report templates.

    Wraps an ``AnalysisResult`` with rendering-time metadata and a pre-computed
    ``chain_summaries`` list for the executive summary section.  Using separate
    ``ChainSummary`` objects for the executive template enforces that it cannot
    accidentally render fields (narratives, step details, full confidence
    breakdowns) that belong in the technical report section.

    Constructed via ``ReportContext.from_result`` rather than directly.
    """

    model_config = ConfigDict(frozen=True)

    analysis: AnalysisResult
    chain_summaries: list[ChainSummary] = Field(
        description="Compact summaries derived from analysis.chains, ordered by rank_score desc"
    )
    generated_at: datetime = Field(description="UTC timestamp when the report was rendered")
    tool_version: str = Field(description="api-attack-path-analyzer version string")
    include_graph: bool = Field(
        default=True,
        description="Whether to embed D3.js attack path graph in HTML output",
    )

    @classmethod
    def from_result(cls, result: AnalysisResult, tool_version: str) -> ReportContext:
        """Construct a ReportContext from a completed AnalysisResult.

        Derives ``chain_summaries`` automatically.
        Sets ``generated_at`` to the current UTC time.
        """
        return cls(
            analysis=result,
            chain_summaries=[ChainSummary.from_chain(c) for c in result.chains],
            generated_at=datetime.now(tz=timezone.utc),
            tool_version=tool_version,
        )
