"""OWASP attack pattern loader.

Reads ``data/attack_patterns.yaml`` bundled alongside this module and returns
typed ``AttackPattern`` dataclasses used by the ranker (M8), traversal engine
(M9), and LLM reasoning agent (M10).

Public API
----------
  load_patterns() -> list[AttackPattern]
      Load (and LRU-cache) all patterns from the bundled YAML file.

  get_pattern(pattern_id: str) -> AttackPattern | None
      Return a single pattern by ID (e.g. ``"AP-001"``), or ``None``.

Design notes
------------
- ``Path(__file__).parent`` locates the YAML file relative to this module so
  the path is correct whether the package is installed as a wheel or run from
  source during development.
- ``functools.lru_cache`` caches the parsed list after the first call.  The
  YAML file is read exactly once per process; subsequent calls return the same
  list object with no I/O.
- ``mitre_hints``, ``indicators``, and ``remediation`` are stored as tuples
  (not lists) so ``AttackPattern`` remains hashable — a requirement for frozen
  dataclasses that appear as dict keys or set members in M8/M9.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from api_analyzer.models.enums import Severity

_DATA_FILE: Path = Path(__file__).parent / "data" / "attack_patterns.yaml"


@dataclass(frozen=True)
class AttackPattern:
    """A single OWASP API Security attack pattern loaded from YAML.

    Instances are immutable and hashable — safe to use as dict keys and in
    sets.  All sequence fields are stored as tuples for this reason.
    """

    id: str
    name: str
    owasp_category: str
    owasp_description: str
    severity: Severity
    confidence_base: float
    mitre_hints: tuple[str, ...]
    description: str
    indicators: tuple[str, ...]
    remediation: tuple[str, ...]


@lru_cache(maxsize=1)
def load_patterns() -> list[AttackPattern]:
    """Load all attack patterns from the bundled YAML file.

    The parsed list is cached after the first call; subsequent calls return the
    same list object with no disk I/O.

    Raises:
        FileNotFoundError: If the bundled YAML file is missing (packaging error).
        yaml.YAMLError: If the YAML is malformed.
        KeyError: If a required field is missing from a pattern entry.
        ValueError: If a severity value does not match the ``Severity`` enum.
    """
    raw_text = _DATA_FILE.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = yaml.safe_load(raw_text)
    return [_parse_entry(e) for e in entries]


def get_pattern(pattern_id: str) -> AttackPattern | None:
    """Return a single ``AttackPattern`` by its ID, or ``None`` if not found.

    Args:
        pattern_id: Pattern identifier, e.g. ``"AP-001"``.

    Returns:
        The matching ``AttackPattern``, or ``None``.
    """
    return next((p for p in load_patterns() if p.id == pattern_id), None)


def _parse_entry(entry: dict[str, Any]) -> AttackPattern:
    return AttackPattern(
        id=entry["id"],
        name=entry["name"],
        owasp_category=entry["owasp_category"],
        owasp_description=entry["owasp_description"],
        severity=Severity(entry["severity"]),
        confidence_base=float(entry["confidence_base"]),
        mitre_hints=tuple(entry.get("mitre_hints") or []),
        description=entry["description"].strip(),
        indicators=tuple(entry.get("indicators") or []),
        remediation=tuple(entry.get("remediation") or []),
    )
