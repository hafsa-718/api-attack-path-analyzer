"""HTML report generator using Jinja2 templates and D3.js visualisations."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from api_analyzer.models.chain import ValidatedChain
from api_analyzer.models.report import AnalysisResult, ReportContext

_TEMPLATE_DIR: Path = Path(__file__).parent / "templates"
_TEMPLATE_NAME: str = "report.html.j2"


def _build_d3_graph_data(chains: list[ValidatedChain]) -> str:
    """Return a JSON string describing nodes and directed links for D3 force graph.

    Nodes are deduplicated by endpoint_id; each node retains the highest severity
    seen across all chains that pass through it (earlier chains rank higher
    because they come from a sorted list).
    """
    _SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}

    nodes: dict[str, dict] = {}
    links: list[dict] = []

    for chain in chains:
        sev = chain.severity.value
        for step in chain.steps:
            eid = step.endpoint_id
            if eid not in nodes:
                nodes[eid] = {
                    "id": eid,
                    "label": f"{step.method} {step.path}",
                    "severity": sev,
                    "pattern_id": chain.pattern_id,
                }
            else:
                if _SEVERITY_RANK.get(sev, 99) < _SEVERITY_RANK.get(nodes[eid]["severity"], 99):
                    nodes[eid]["severity"] = sev

        for i in range(len(chain.steps) - 1):
            links.append(
                {
                    "source": chain.steps[i].endpoint_id,
                    "target": chain.steps[i + 1].endpoint_id,
                    "chain_id": chain.id,
                    "pattern_id": chain.pattern_id,
                    "severity": sev,
                }
            )

    return json.dumps({"nodes": list(nodes.values()), "links": links})


def generate_report(
    result: AnalysisResult,
    output_path: Path | str,
    *,
    tool_version: str = "0.1.0",
    include_graph: bool = True,
) -> Path:
    """Render an HTML security report to *output_path* and return the resolved path.

    Creates parent directories as needed.  Overwrites any existing file at
    *output_path*.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    context = ReportContext.from_result(result, tool_version=tool_version)
    # include_graph flag is immutable on the frozen model; rebuild if caller overrides
    if not include_graph:
        context = ReportContext(
            analysis=context.analysis,
            chain_summaries=context.chain_summaries,
            generated_at=context.generated_at,
            tool_version=context.tool_version,
            include_graph=False,
        )

    d3_data = _build_d3_graph_data(result.chains)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    template = env.get_template(_TEMPLATE_NAME)
    html = template.render(ctx=context, d3_graph_data=d3_data)

    output_path.write_text(html, encoding="utf-8")
    return output_path
