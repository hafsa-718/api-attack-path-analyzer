# AI-Powered API Attack Path Analyzer

> Transforms an OpenAPI specification into a ranked list of multi-hop exploit chains — combining Neo4j knowledge graph traversal with Claude LLM reasoning to discover what isolated vulnerability scanners miss.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?logo=neo4j&logoColor=white)](https://neo4j.com)
[![Claude](https://img.shields.io/badge/Claude-claude--sonnet--4--6-orange)](https://anthropic.com)
[![OWASP API Top 10](https://img.shields.io/badge/OWASP-API%20Top%2010%202023-A30000)](https://owasp.org/API-Security/)
[![MITRE ATT&CK](https://img.shields.io/badge/MITRE-ATT%26CK%20v14-red)](https://attack.mitre.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## The Problem

API vulnerability scanners find **isolated weaknesses** — a missing auth header here, an IDOR there. Real attackers think in **chains**:

```
GET /users?page=1          → enumerate user IDs (no auth, returns integers)
        ↓
GET /users/4821/profile    → fetch PII for any ID (BOLA, no ownership check)
        ↓
POST /auth/password-reset  → trigger reset for any account (no rate limit)

Three findings: CVSS 4.3, CVSS 5.4, CVSS 5.9
One chain:      Account Takeover at Scale — CVSS 9.1
```

No existing open-source tool connects those dots automatically.

This platform does.

---

## How It Works

```
  OpenAPI Spec (.yaml / .json / URL)
            │
            ▼
  ┌─────────────────────┐
  │    Spec Parser      │  Extracts endpoints, parameters, auth
  │    + Classifier     │  schemes, and infers resource relationships
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Knowledge Graph    │  Builds a Neo4j property graph:
  │  Builder (Neo4j)    │  endpoints → resources → auth → schemas
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  Attack Path Engine │  Traverses the graph with OWASP API Top 10
  │  (Cypher + YAML)    │  pattern templates to find candidate chains
  └──────────┬──────────┘
             │
             ▼
  ┌─────────────────────┐
  │  LLM Reasoning      │  Single Claude agent with graph tool calling.
  │  Agent (Claude)     │  Validates feasibility. Maps to MITRE ATT&CK.
  └──────────┬──────────┘  Scores confidence. Writes narrative.
             │
             ▼
  ┌─────────────────────────────────────────────┐
  │              Report                         │
  │  HTML: D3.js attack path graph + findings   │
  │  Markdown: developer remediation guide      │
  └─────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Start Neo4j (only external dependency)
docker compose up -d

# 2. Install the tool
pip install -e .

# 3. Analyze an API
api-analyzer analyze your-openapi-spec.yaml --output report.html

# 4. Open the report
open report.html
```

**Or scan without LLM (graph-only, instant, free):**
```bash
api-analyzer scan your-spec.yaml --format markdown
```

---

## Demo

*[Demo GIF will go here — recorded against crAPI showing BOLA → PII harvest → Account Takeover chain]*

**Example output for crAPI:**
```
Analysis complete — 3 attack chains discovered

CRITICAL  [confidence: 0.87]  BOLA → Mass Data Exfiltration
          GET /community/api/v2/community/posts/recent (public)
          → GET /community/api/v2/community/posts/{postId} (BOLA, integer ID)
          → GET /identity/api/v2/user/dashboard (PII exposure)
          MITRE: T1589.001 → T1530
          Remediation: Enforce ownership check on postId; validate caller owns resource

HIGH      [confidence: 0.79]  Broken Function-Level Auth → Admin Operations
          ...

MEDIUM    [confidence: 0.65]  Mass Assignment via Profile Update
          ...
```

---

## Attack Patterns

Six built-in YAML templates covering the highest-impact OWASP API Security categories.
Add custom patterns with no code changes — drop a YAML file in `patterns/`.

| Pattern | OWASP Category | MITRE Techniques |
|---|---|---|
| BOLA / IDOR via sequential identifiers | API1:2023 | T1589.001, T1530 |
| Broken authentication (public → auth-required adjacency) | API2:2023 | T1078, T1110 |
| Privilege escalation via role parameters | API5:2023 | T1078.001 |
| Mass assignment (sensitive fields in PATCH/PUT) | API3:2023 | T1565.001 |
| Data exfiltration (listing → detail → PII chain) | API1:2023 | T1567 |
| SSRF via URL/callback parameters | API7:2023 | T1090 |

**Pattern format** — adding a new pattern takes under 10 minutes:

```yaml
pattern_id: AP-007
name: "Unauthenticated Webhook Registration"
owasp_category: "API8:2023"
mitre_techniques: ["T1071.001"]
cypher_template: |
  MATCH (e:Endpoint)
  WHERE e.is_public = true
    AND e.accepts_url_param = true
    AND e.method = 'POST'
  RETURN e LIMIT 10
confidence_base: 0.60
severity_floor: "MEDIUM"
```

---

## AI Design: Grounded Reasoning

The LLM agent can only claim a vulnerability exists if its graph tool calls confirm it.

```
Agent receives: candidate chain (ordered endpoint IDs from graph traversal)

Agent calls:    get_endpoint("GET:/users/{userId}")
                → { is_public: false, path_param_type: "INTEGER",
                    sensitivity_class: "SENSITIVE", returns_pii: true }

Agent calls:    get_endpoint("GET:/users")
                → { is_public: true, returns_collection: true }

Agent reasons:  "Public collection endpoint returns integer IDs.
                 Detail endpoint accepts integer ID parameter with no auth.
                 Returns PII. This is a confirmed BOLA chain."

Agent returns:  ValidatedChain with confidence: 0.84, narrative, MITRE TTPs
```

If the graph data does not support the claim, the agent says so and returns `confidence: 0.0`.
No hallucinated vulnerabilities. No black-box scores.

---

## What's Different

| Capability | This Tool | Burp Suite | 42Crunch | OWASP ZAP |
|---|:---:|:---:|:---:|:---:|
| Multi-hop attack chain discovery | ✅ | ❌ | ❌ | ❌ |
| Neo4j knowledge graph from spec | ✅ | ❌ | ❌ | ❌ |
| LLM reasoning grounded in graph data | ✅ | ❌ | ❌ | ❌ |
| MITRE ATT&CK kill chain mapping | ✅ | ❌ | ❌ | ❌ |
| Transparent confidence score breakdown | ✅ | ❌ | Partial | ❌ |
| No dynamic testing / traffic required | ✅ | ❌ | ✅ | ❌ |
| Open source + self-hostable | ✅ | ❌ | ❌ | ✅ |
| Extensible via YAML (no code changes) | ✅ | ❌ | Partial | ❌ |

---

## CLI Reference

```bash
# Full analysis (parser → graph → patterns → LLM → report)
api-analyzer analyze <spec> [--output report.html] [--format html|markdown]

# Graph-only scan (no LLM, instant, free)
api-analyzer scan <spec> [--pattern bola_idor] [--format json|markdown]

# Visualize the knowledge graph in browser (static HTML)
api-analyzer graph <spec>

# Start minimal API server for CI/CD integration
api-analyzer serve [--port 8000]

# Check Neo4j connection and spec validity
api-analyzer check <spec>
```

**Global flags:**
```bash
--verbose      Show Cypher queries, LLM prompts, token usage
--no-llm       Skip LLM reasoning (graph-only mode)
--model        LLM model override (default: claude-sonnet-4-6)
```

---

## Optional REST API

```bash
api-analyzer serve  # starts FastAPI on :8000
```

```
POST  /api/v1/analyze              Submit spec for analysis
GET   /api/v1/analyses/{id}        Get status and summary
GET   /api/v1/analyses/{id}/chains Get ranked attack chains
GET   /api/v1/analyses/{id}/report Get HTML report
GET   /health                      Health check
```

Full OpenAPI docs at `http://localhost:8000/docs`.

---

## Architecture

### Technology Stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | Best LLM + graph ecosystem |
| CLI | Typer | Clean argument parsing, auto-generated help |
| Spec parsing | prance | Handles `$ref` resolution, multi-file specs |
| Graph database | Neo4j 5.x | Native graph traversal; O(1) per hop vs O(log n) in SQL |
| LLM reasoning | Anthropic SDK (Claude) | 200K context, reliable structured JSON output |
| Data validation | Pydantic v2 | Hallucination firewall on LLM outputs |
| Report templates | Jinja2 | Zero JS build step, self-contained HTML output |
| Storage | SQLite (via SQLAlchemy) | Zero ops; sufficient for single-user research tool |
| Container | Docker Compose (Neo4j only) | `docker compose up` → database ready |
| Tests | pytest | Unit + integration against intentionally vulnerable APIs |

### Project Structure

```
api_analyzer/
├── parser/           Spec ingestion, normalization, sensitivity classification
├── graph/            Neo4j client, graph builder, Cypher schema, query files
├── attack_path/      Pattern engine, Cypher traversal, ranking, deduplication
│   └── patterns/     YAML attack pattern templates (extensible, no code changes)
├── reasoning/        Claude agent, tool definitions, prompt templates, validator
│   └── mitre/        Local MITRE ATT&CK bundle (offline-capable)
├── reporting/        Jinja2 HTML + Markdown report generation
│   └── templates/    report.html.j2, report.md.j2
├── models/           Shared Pydantic models across all modules
├── api/              Optional thin FastAPI wrapper
└── cli.py            Typer CLI entry point
```

---

## Development

### Requirements
- Python 3.11+
- Docker (for Neo4j)
- An [Anthropic API key](https://console.anthropic.com)

### Setup

```bash
git clone https://github.com/yourusername/api-attack-path-analyzer
cd api-attack-path-analyzer

# Start Neo4j
docker compose up -d

# Install with dev dependencies
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# Verify setup
api-analyzer check tests/fixtures/petstore.yaml
```

### Run Tests

```bash
# Unit tests (no Neo4j, no LLM required)
pytest tests/unit/ -v

# Integration tests (requires Neo4j running, LLM is mocked)
pytest tests/integration/ -v

# Full suite with coverage
pytest tests/ --cov=api_analyzer --cov-fail-under=80
```

### Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...          # Required for LLM reasoning
NEO4J_URI=bolt://localhost:7687       # Default from docker-compose
NEO4J_PASSWORD=your-password          # Set in docker-compose.yml
LLM_MODEL=claude-sonnet-4-6           # Override model
LLM_CACHE_ENABLED=true                # Cache LLM responses (saves cost)
LOG_LEVEL=INFO                        # DEBUG shows Cypher queries + prompts
```

---

## Evaluation

Tested against intentionally vulnerable APIs:

| API | Vulnerabilities | Chains Found | False Positives |
|---|---|---|---|
| [crAPI](https://github.com/OWASP/crAPI) | BOLA, Broken Auth, Mass Assignment | *TBD* | *TBD* |
| [vAPI](https://github.com/roottusk/vapi) | All OWASP API Top 10 | *TBD* | *TBD* |
| Petstore (clean baseline) | None | 0 critical/high | 0 |

*Results will be populated after M2 and M3 completion.*

---

## Roadmap

| Version | Status | Focus |
|---|---|---|
| **V1 — Research MVP** | In Progress | CLI + Neo4j graph + YAML patterns + Claude agent + HTML report |
| **V2 — Community** | Planned | React frontend + Cytoscape.js graph + SARIF export + GitHub Actions |
| **V3 — Platform** | Planned | Multi-tenancy + research white paper + conference submission |

Detailed milestone plan: [docs/architecture.md](docs/architecture.md)

---

## Research

**Research question:**
> *Can a knowledge graph built from an OpenAPI specification automatically discover multi-hop API attack chains and use LLM reasoning to explain their business impact?*

**White paper** (targeting V3): *"AI-Augmented API Attack Chain Discovery: A Knowledge Graph Approach"*

**Evaluation datasets:** crAPI, vAPI, dvAPI, and a curated set of real-world public API specifications.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Good first issues:

- **Add an attack pattern** — write a YAML file in `attack_path/patterns/`
- **Improve sensitivity classification** — add rules to `parser/classifier.py`
- **Add a vulnerable API fixture** — add a spec to `tests/fixtures/` with a test
- **Improve report template** — enhance `reporting/templates/report.html.j2`
- **Write a MITRE technique lookup** — expand the local MITRE bundle tooling

Pre-commit hooks (ruff + mypy + pytest unit) run automatically on commit.

---

## Security

This tool processes API specifications that may reveal internal architecture. See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

Do not open public issues for security vulnerabilities in this tool.

---

## License

[Apache License 2.0](LICENSE) — permissive, enterprise-friendly, compatible with commercial use.

---

*A security research project demonstrating knowledge graph analysis, LLM grounding techniques, and automated threat modeling for API security.*
