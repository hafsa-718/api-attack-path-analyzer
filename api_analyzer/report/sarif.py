"""SARIF 2.1.0 export for API Attack Path Analyzer findings.

SARIF (Static Analysis Results Interchange Format) is the OASIS standard
consumed natively by GitHub Advanced Security, GitLab, Azure DevOps, and
VS Code.  Uploading the output of this function to GitHub via
``github/codeql-action/upload-sarif`` makes findings appear as inline
PR annotations and in the Security → Code Scanning tab.

Usage
-----
    from api_analyzer.report.sarif import generate_sarif
    sarif_dict = generate_sarif(result, spec_filename="openapi.yaml")
    import json
    Path("results.sarif").write_text(json.dumps(sarif_dict, indent=2))

GitHub Actions integration
--------------------------
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: results.sarif
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api_analyzer import __version__
from api_analyzer.models.enums import Severity
from api_analyzer.models.report import AnalysisResult

# SARIF level mapping
_SEVERITY_TO_LEVEL: dict[Severity, str] = {
    Severity.CRITICAL: "error",
    Severity.HIGH:     "error",
    Severity.MEDIUM:   "warning",
    Severity.LOW:      "note",
}

# Rule metadata per attack pattern
_RULES: dict[str, dict[str, Any]] = {
    "AP-001": {
        "name": "BrokenObjectLevelAuthorization",
        "short": "BOLA / Broken Object Level Authorization",
        "full": (
            "An attacker substitutes the ID of a resource they own with another "
            "user's resource ID. The API does not verify that the requesting user "
            "has access to the target resource."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa1-broken-object-level-authorization/",
        "tags": ["OWASP-API1-2023", "BOLA", "IDOR"],
    },
    "AP-002": {
        "name": "BrokenAuthentication",
        "short": "Broken Authentication",
        "full": (
            "Authentication mechanisms are implemented incorrectly, allowing attackers "
            "to compromise authentication tokens or exploit implementation flaws to "
            "assume other users' identities."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa2-broken-authentication/",
        "tags": ["OWASP-API2-2023", "BrokenAuth"],
    },
    "AP-003": {
        "name": "PrivilegeEscalation",
        "short": "Privilege Escalation via Role Parameter Injection",
        "full": (
            "A low-privilege user can escalate to higher privileges by manipulating "
            "role or permission parameters in API requests that are not properly "
            "validated server-side."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa5-broken-function-level-authorization/",
        "tags": ["OWASP-API5-2023", "PrivEsc"],
    },
    "AP-004": {
        "name": "MassAssignment",
        "short": "Mass Assignment",
        "full": (
            "The API automatically binds client-provided data to internal object "
            "properties, allowing attackers to modify fields they should not have "
            "access to, such as role, admin status, or balance."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa6-unrestricted-access-to-sensitive-business-flows/",
        "tags": ["OWASP-API6-2023", "MassAssignment"],
    },
    "AP-005": {
        "name": "ExcessiveDataExposure",
        "short": "Excessive Data Exposure",
        "full": (
            "The API returns more data than the client needs, exposing sensitive "
            "fields that a filtering layer should remove. Attackers can harvest "
            "credentials, PII, or internal state from oversized responses."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/",
        "tags": ["OWASP-API3-2023", "ExcessiveData"],
    },
    "AP-006": {
        "name": "ServerSideRequestForgery",
        "short": "SSRF / Server-Side Request Forgery",
        "full": (
            "The API accepts a user-supplied URL or network location and makes "
            "a server-side request to it, allowing attackers to probe internal "
            "services, cloud metadata endpoints, or internal APIs."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa7-server-side-request-forgery/",
        "tags": ["OWASP-API7-2023", "SSRF"],
    },
    "AP-007": {
        "name": "AuthChainCredentialTheft",
        "short": "Auth Chain: Credential Theft to Sensitive Data Access",
        "full": (
            "A multi-hop attack chain where an attacker exploits a weakness in "
            "the authentication flow (e.g. OTP brute-force, token theft) and "
            "leverages the gained credential to access sensitive data endpoints."
        ),
        "help_uri": "https://owasp.org/API-Security/editions/2023/en/0xa2-broken-authentication/",
        "tags": ["OWASP-API2-2023", "AuthChain", "CredentialTheft"],
    },
}

_FALLBACK_RULE: dict[str, Any] = {
    "name": "ApiSecurityFinding",
    "short": "API Security Finding",
    "full": "A multi-hop API attack chain was identified by graph traversal and LLM reasoning.",
    "help_uri": "https://owasp.org/API-Security/",
    "tags": ["OWASP-API-Top-10"],
}


def _make_rules(result: AnalysisResult) -> list[dict[str, Any]]:
    """Emit one rule entry per unique pattern_id found in chains."""
    seen: set[str] = set()
    rules: list[dict[str, Any]] = []
    for chain in result.chains:
        pid = chain.pattern_id
        if pid in seen:
            continue
        seen.add(pid)
        meta = _RULES.get(pid, _FALLBACK_RULE)
        rules.append({
            "id": pid,
            "name": meta["name"],
            "shortDescription": {"text": meta["short"]},
            "fullDescription":  {"text": meta["full"]},
            "helpUri": meta["help_uri"],
            "properties": {
                "tags":      meta["tags"],
                "precision": "medium",
                "problem.severity": "error",
            },
        })
    return rules


def _chain_message(chain: Any) -> str:
    steps_summary = " → ".join(
        f"{s.method} {s.path}" for s in chain.steps
    )
    conf_pct = int(chain.confidence.final_score * 100)
    probe_note = ""
    if chain.probe_result:
        probe_note = f" Runtime probe: {chain.probe_result.outcome.value}."
    return (
        f"{chain.name}. "
        f"Confidence: {conf_pct}%. "
        f"Attack path: {steps_summary}. "
        f"{chain.owasp_category}.{probe_note} "
        f"Remediation: {chain.remediation[0] if chain.remediation else 'See full report.'}"
    )


def generate_sarif(
    result: AnalysisResult,
    spec_filename: str = "openapi.yaml",
    tool_version: str | None = None,
) -> dict[str, Any]:
    """Convert an AnalysisResult to a SARIF 2.1.0 dict.

    Args:
        result:         Completed AnalysisResult from the pipeline.
        spec_filename:  The spec file name shown in finding locations.
                        Defaults to ``openapi.yaml``.
        tool_version:   Override for the tool version string.

    Returns:
        A dict that serialises to valid SARIF 2.1.0 JSON.
    """
    version = tool_version or __version__

    sarif_results: list[dict[str, Any]] = []
    for chain in result.chains:
        level = _SEVERITY_TO_LEVEL.get(chain.severity, "warning")

        # Related locations: one per attack step
        related: list[dict[str, Any]] = [
            {
                "message": {"text": f"Step {s.sequence}: {s.action} — {s.attacker_gains}"},
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": spec_filename,
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": {"startLine": 1},
                },
            }
            for s in chain.steps
        ]

        sarif_results.append({
            "ruleId":  chain.pattern_id,
            "level":   level,
            "message": {"text": _chain_message(chain)},
            "locations": [
                {
                    "physicalLocation": {
                        "artifactLocation": {
                            "uri": spec_filename,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "region": {"startLine": 1},
                    },
                    "logicalLocations": [
                        {
                            "name": f"{chain.steps[0].method} {chain.steps[0].path}",
                            "kind": "function",
                        }
                    ],
                }
            ],
            "relatedLocations": related,
            "properties": {
                "confidence":      chain.confidence.final_score,
                "severity":        chain.severity.value,
                "owasp_category":  chain.owasp_category,
                "mitre_techniques": chain.mitre_techniques,
                "hop_count":       len(chain.steps) - 1,
                "rationale":       chain.confidence.rationale,
                "remediation":     chain.remediation,
                "analyzed_at":     chain.analyzed_at.isoformat(),
                "llm_model":       chain.llm_model,
                **(
                    {"probe_outcome": chain.probe_result.outcome.value,
                     "probe_url":     chain.probe_result.probed_url}
                    if chain.probe_result else {}
                ),
            },
        })

    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name":           "API Attack Path Analyzer",
                        "version":        version,
                        "informationUri": "https://github.com/your-org/api-attack-path-analyzer",
                        "rules":          _make_rules(result),
                    }
                },
                "results":   sarif_results,
                "artifacts": [
                    {
                        "location": {
                            "uri":       spec_filename,
                            "uriBaseId": "%SRCROOT%",
                        },
                        "mimeType": (
                            "application/json"
                            if spec_filename.endswith(".json")
                            else "application/yaml"
                        ),
                    }
                ],
                "properties": {
                    "spec_title":          result.spec_title,
                    "spec_version":        result.spec_version,
                    "analysis_id":         result.analysis_id,
                    "analyzed_at":         result.analyzed_at.isoformat(),
                    "spec_completeness":   result.spec_completeness,
                    "endpoint_count":      result.endpoint_count,
                    "candidates_evaluated": result.candidates_evaluated,
                    "estimated_cost_usd":  result.estimated_cost_usd,
                    "duration_seconds":    result.duration_seconds,
                },
            }
        ],
    }


def write_sarif(
    result: AnalysisResult,
    output_path: Path | str,
    spec_filename: str = "openapi.yaml",
    tool_version: str | None = None,
) -> None:
    """Serialise SARIF to a file. Convenience wrapper around generate_sarif."""
    sarif = generate_sarif(result, spec_filename=spec_filename, tool_version=tool_version)
    Path(output_path).write_text(json.dumps(sarif, indent=2), encoding="utf-8")
