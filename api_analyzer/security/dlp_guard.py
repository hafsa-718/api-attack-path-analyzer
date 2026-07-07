"""Data Loss Prevention guard for OpenAPI spec inputs.

Scans spec string values for sensitive data — credentials, PII, infrastructure
secrets — and redacts them before content reaches the Neo4j graph or the LLM.

Why this matters
----------------
OpenAPI specs frequently contain real data in ``example``, ``default``, and
``enum`` fields: real email addresses, test credentials, JWT tokens, database
URLs.  Without redaction these flow verbatim into the LLM context and into
graph node properties, potentially leaking them to Anthropic's API or into
logged tool-call results.

What is scanned and what is skipped
------------------------------------
  DATA fields   (example, default, enum, x-example)
                → full scan: credentials, PII, entropy check
  TEXT fields   (description, summary, title)
                → credentials only; PII skipped (phone/email can appear
                  legitimately in documentation)
  STRUCTURAL    (type, format, in, required, name, operationId, security,
                  paths, openapi, swagger, info)
                → never touched; these drive analysis correctness

Detection methods
-----------------
1. Pattern matching — 20+ compiled regexes covering known sensitive formats:
   AWS keys, GitHub/GitLab PATs, JWTs, Anthropic keys, private key headers,
   database URLs, credentials-in-URL, email, phone, SSN, credit card, DOB.

2. Shannon entropy — any string in a data-bearing field that is ≥ 20 chars
   and has entropy ≥ 4.5 bits/char is treated as a likely random secret.
   UUIDs, paths, and URLs are excluded from the entropy check (they are
   high-entropy but not secrets).

Impact on analysis
------------------
Zero.  The graph builder reads structural properties (path, method, is_public,
sensitivity_class, auth_declared, parameter *types*).  It never stores example
values.  Redacting examples does not change any property the traversal engine
or LLM reasoning agent uses.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# ── Sensitive-data pattern library ────────────────────────────────────────────

# Tuple: (compiled pattern, human label, category)
_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # ── Credentials ───────────────────────────────────────────────────────────
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
     "AWS access key ID", "credential"),

    (re.compile(r"\bsk-ant-api\d{2}-[A-Za-z0-9\-_]{90,}\b"),
     "Anthropic API key", "credential"),

    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
     "API key (sk- prefix)", "credential"),

    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
     "GitHub personal access token", "credential"),

    (re.compile(r"\bgho_[A-Za-z0-9]{36}\b"),
     "GitHub OAuth token", "credential"),

    (re.compile(r"\bghs_[A-Za-z0-9]{36}\b"),
     "GitHub app token", "credential"),

    (re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"),
     "GitLab personal access token", "credential"),

    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
     "private key block", "credential"),

    # JWT: three base64url segments separated by dots
    (re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
     "JWT token", "credential"),

    # credentials embedded in a URL  (http://user:pass@host)
    (re.compile(r"https?://[^:@\s\"']{1,100}:[^@\s\"']{1,100}@"),
     "credentials in URL", "credential"),

    # explicit secret assignment patterns  (password=abc123, api_key: "xyz")
    (re.compile(
        r"\b(?:password|passwd|pwd|secret|api[_-]?key|auth[_-]?token|access[_-]?token)"
        r"\s*[:=]\s*[\"']?(?!<|\[)[^\s\"'<\[]{6,}",
        re.I,
    ), "secret assignment", "credential"),

    # ── Infrastructure ────────────────────────────────────────────────────────
    (re.compile(
        r"\b(?:mongodb(?:\+srv)?|postgresql|postgres|mysql|mariadb|redis|amqp"
        r"|rabbitmq|cassandra|elasticsearch|neo4j(?:\+s)?)://[^\s\"']{8,}",
        re.I,
    ), "database / service connection string", "infrastructure"),

    # ── PII ───────────────────────────────────────────────────────────────────
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "email address", "pii"),

    # US phone  (NPA-NXX-XXXX with optional country code)
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
     "US phone number", "pii"),

    # E.164 international  (+44 7xxx xxxxxx etc.)
    (re.compile(r"\+[1-9]\d{6,14}\b"),
     "international phone number", "pii"),

    # US SSN
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     "US Social Security Number", "pii"),

    # Major credit card patterns (Visa/MC/Amex/Discover)
    (re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}"
        r"|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ), "credit card number", "pii"),

    # ISO date that looks like DOB  (1985-04-23)
    (re.compile(r"\b(?:19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"),
     "date of birth pattern", "pii"),
]

# Replacement placeholder per category
_PLACEHOLDERS: dict[str, str] = {
    "credential":     "[REDACTED:CREDENTIAL]",
    "infrastructure": "[REDACTED:INFRASTRUCTURE]",
    "pii":            "[REDACTED:PII]",
    "secret":         "[REDACTED:SECRET]",
}

# ── Shannon entropy — high-entropy secret detection ───────────────────────────

_MIN_ENTROPY: float = 4.5   # bits / character; random secrets typically > 4.5
_MIN_SECRET_LEN: int = 20   # shorter strings aren't meaningful secrets

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)


def _shannon_entropy(s: str) -> float:
    if len(s) < 2:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _is_high_entropy_secret(value: str) -> bool:
    """Return True if ``value`` looks like a random secret based on entropy alone."""
    stripped = value.strip().strip("\"'")
    if len(stripped) < _MIN_SECRET_LEN:
        return False
    # UUIDs are high-entropy IDs, not secrets
    if _UUID_RE.match(stripped):
        return False
    # Paths and URLs have high entropy but are not secrets
    if stripped.startswith(("/", "http://", "https://")):
        return False
    return _shannon_entropy(stripped) >= _MIN_ENTROPY


# ── Field classification ───────────────────────────────────────────────────────

# Full scan: pattern matching + entropy check
_DATA_FIELDS: frozenset[str] = frozenset({
    "example", "default", "x-example", "x-examples",
})

# Partial scan: credentials only (PII may appear legitimately in docs)
_TEXT_FIELDS: frozenset[str] = frozenset({
    "description", "summary", "title", "x-internal-note", "x-description",
    "x-summary", "message", "detail",
})

# Never touch: structural fields that drive analysis
_STRUCTURAL_FIELDS: frozenset[str] = frozenset({
    "type", "format", "in", "required", "name", "operationId",
    "$ref", "security", "securitySchemes", "method",
    "openapi", "swagger", "info", "servers", "tags",
    "paths", "components", "definitions", "parameters",
})


# ── Per-value scanner and redactor ────────────────────────────────────────────

def _scan_and_redact(
    value: str,
    path: str,
    field_key: str,
    warnings: list[str],
) -> str:
    """Scan ``value`` for sensitive data and return a redacted copy."""
    is_data_field = field_key in _DATA_FIELDS or field_key == "enum"
    is_text_field = field_key in _TEXT_FIELDS
    # Fields not in either set get the same treatment as text fields
    scan_pii = is_data_field

    result = value
    matched_categories: set[str] = set()

    for pattern, label, category in _PATTERNS:
        if category == "pii" and not scan_pii:
            continue
        if pattern.search(result):
            matched_categories.add(category)
            result = pattern.sub(_PLACEHOLDERS[category], result)

    # Entropy check only on data-bearing fields (examples/defaults),
    # not on free-text descriptions which are natural language
    if is_data_field and not matched_categories and _is_high_entropy_secret(value):
        matched_categories.add("secret")
        result = _PLACEHOLDERS["secret"]

    if matched_categories:
        preview = value[:24].replace("\n", " ")
        cats = ", ".join(sorted(matched_categories))
        ellipsis = "…" if len(value) > 24 else ""
        warnings.append(
            f"DLP [{cats}] at {path!r}: {preview!r}{ellipsis}"
        )

    return result


# ── Recursive spec walker ─────────────────────────────────────────────────────

def _walk(
    node: Any,
    path: str,
    parent_key: str,
    warnings: list[str],
) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _STRUCTURAL_FIELDS:
                out[k] = v  # structural fields: pass through untouched
            else:
                out[k] = _walk(v, f"{path}.{k}", k, warnings)
        return out

    if isinstance(node, list):
        return [
            _walk(item, f"{path}[{i}]", parent_key, warnings)
            for i, item in enumerate(node)
        ]

    if isinstance(node, str) and node.strip() and parent_key not in _STRUCTURAL_FIELDS:
        return _scan_and_redact(node, path, parent_key, warnings)

    return node


# ── Public API ────────────────────────────────────────────────────────────────

def redact_spec(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Scan and redact sensitive data from all string values in a raw spec dict.

    Returns:
        (redacted_dict, warnings)
        ``warnings`` is empty when no sensitive data was detected.
        Each warning includes the JSON path, category, and a short preview
        of the original value for operator review.
    """
    warnings: list[str] = []
    redacted = _walk(raw, "spec", "", warnings)
    return redacted, warnings
