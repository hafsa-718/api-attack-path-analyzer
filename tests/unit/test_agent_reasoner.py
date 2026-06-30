"""Unit tests for api_analyzer.agent.reasoner (M10).

No real Anthropic API calls or Neo4j connections are made.
The ``anthropic.Anthropic`` client is injected as a MagicMock, and
``session.run()`` is mocked to control graph tool responses.

Mock response builder
---------------------
``_response(blocks, stop_reason)`` creates a minimal mock that satisfies
the attribute access pattern in the agent loop:
  - response.content  → list of block mocks
  - response.stop_reason → str
  - response.usage.input_tokens / output_tokens → int

Block mocks follow the Anthropic ToolUseBlock shape:
  - block.type  → "tool_use" | "text"
  - block.name  → tool name
  - block.id    → unique str
  - block.input → dict
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, call, patch

import pytest

from api_analyzer.agent.reasoner import (
    _SYSTEM_PROMPT,
    _TOOL_SCHEMAS,
    _analyze_chain,
    _build_user_prompt,
    _build_validated_chain,
    _check_auth_scheme,
    _execute_tool,
    _get_endpoint_info,
    _get_resource_info,
    analyze,
)
from api_analyzer.engine.traversal import TraversalResult
from api_analyzer.models.chain import (
    AttackStep,
    CandidateChain,
    ConfidenceBreakdown,
    ValidatedChain,
)
from api_analyzer.models.enums import Severity
from api_analyzer.agent.reasoner import ReasonerConfig


# ── Mock factories ─────────────────────────────────────────────────────────────


def _tool_block(name: str, input_data: dict, id: str = "tu_001") -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.id = id
    block.input = input_data
    return block


def _text_block(text: str = "thinking...") -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _usage(input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    return usage


def _response(
    blocks: list[MagicMock],
    stop_reason: str = "tool_use",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.usage = _usage(input_tokens, output_tokens)
    return resp


def _client(*responses: MagicMock) -> MagicMock:
    """Mock Anthropic client whose messages.create returns responses in order."""
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def _neo4j_session() -> MagicMock:
    session = MagicMock()
    # Default: single() returns None (not found)
    session.run.return_value.single.return_value = None
    return session


def _neo4j_driver(session: MagicMock | None = None) -> MagicMock:
    if session is None:
        session = _neo4j_session()
    driver = MagicMock()
    driver.session.return_value.__enter__ = MagicMock(return_value=session)
    driver.session.return_value.__exit__ = MagicMock(return_value=False)
    return driver


# ── Minimal CandidateChain factory ────────────────────────────────────────────

def _chain(
    pattern_id: str = "AP-001",
    entry_id: str = "GET:/users",
    exit_id: str = "GET:/users/{id}",
    sensitivity_delta: int = 2,
    crosses_auth: bool = True,
) -> CandidateChain:
    from api_analyzer.patterns.loader import get_pattern
    p = get_pattern(pattern_id)
    assert p is not None
    import uuid as _uuid
    return CandidateChain(
        id=str(_uuid.uuid4()),
        pattern_id=p.id,
        pattern_name=p.name,
        owasp_category=p.owasp_category,
        mitre_hints=list(p.mitre_hints),
        confidence_base=p.confidence_base,
        endpoint_ids=[entry_id, exit_id],
        hop_count=1,
        entry_endpoint_id=entry_id,
        exit_endpoint_id=exit_id,
        crosses_auth_boundary=crosses_auth,
        sensitivity_delta=sensitivity_delta,
        rank_score=0.7,
        entry_summary=f"GET /users (PUBLIC)",
        exit_summary=f"GET /users/{{id}} (SENSITIVE)",
    )


def _traversal_result(chains: list[CandidateChain] | None = None) -> TraversalResult:
    return TraversalResult(
        spec_id="api:1.0",
        chains=chains or [],
        total_candidates=len(chains or []),
        spec_completeness=0.8,
        endpoint_count=10,
        candidate_counts={"bola": len(chains or [])},
    )


# ── Valid submit_analysis input ────────────────────────────────────────────────

def _valid_submit(llm_self_score: float = 0.8) -> dict:
    return {
        "name": "BOLA → PII Exfiltration via User Detail",
        "severity": "HIGH",
        "llm_self_score": llm_self_score,
        "rationale": "The endpoint returns full user objects including PII fields without ownership checks.",
        "steps": [
            {
                "sequence": 1,
                "endpoint_id": "GET:/users",
                "path": "/users",
                "method": "GET",
                "auth_required": "None — public endpoint",
                "action": "Enumerate user IDs from public collection endpoint",
                "attacker_gains": "List of sequential integer user IDs",
                "technique": "Integer enumeration via public collection endpoint",
            },
            {
                "sequence": 2,
                "endpoint_id": "GET:/users/{id}",
                "path": "/users/{id}",
                "method": "GET",
                "auth_required": "Bearer token (any valid user)",
                "action": "Access user records using enumerated IDs from step 1",
                "attacker_gains": "Full user PII including email, address, SSN",
                "technique": "BOLA / Insecure Direct Object Reference via predictable integer ID",
            },
        ],
        "narrative": (
            "An attacker first calls GET /users to enumerate all user IDs from the public "
            "collection endpoint. The response contains sequential integer IDs, making enumeration "
            "trivial. The attacker then iterates through these IDs calling GET /users/{id} to "
            "retrieve full user objects containing PII fields including email addresses, phone "
            "numbers, and home addresses. No ownership validation is performed server-side."
        ),
        "mitre_techniques": ["T1078", "T1530"],
        "remediation": [
            "Implement object-level authorization checks on GET /users/{id}",
            "Validate that the authenticated user owns the requested resource",
            "Use non-sequential GUIDs for user identifiers",
        ],
    }


# ── ReasonerConfig ─────────────────────────────────────────────────────────────


class TestReasonerConfig:
    def test_defaults(self) -> None:
        cfg = ReasonerConfig()
        assert cfg.llm_model == "claude-sonnet-4-6"
        assert cfg.max_tokens == 4096
        assert cfg.max_tool_calls_per_chain == 5
        assert cfg.confidence_threshold == 0.4

    def test_frozen(self) -> None:
        cfg = ReasonerConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.llm_model = "other"  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = ReasonerConfig(llm_model="claude-opus-4-8", max_tokens=8192,
                             max_tool_calls_per_chain=3, confidence_threshold=0.6)
        assert cfg.llm_model == "claude-opus-4-8"
        assert cfg.confidence_threshold == 0.6


# ── Static content ─────────────────────────────────────────────────────────────


class TestStaticContent:
    def test_system_prompt_non_empty(self) -> None:
        assert _SYSTEM_PROMPT.strip()

    def test_system_prompt_mentions_submit_analysis(self) -> None:
        assert "submit_analysis" in _SYSTEM_PROMPT

    def test_four_tools_defined(self) -> None:
        assert len(_TOOL_SCHEMAS) == 4

    def test_tool_names(self) -> None:
        names = {t["name"] for t in _TOOL_SCHEMAS}
        assert names == {
            "get_endpoint_info", "get_resource_info",
            "check_auth_scheme", "submit_analysis",
        }

    def test_all_tools_have_input_schema(self) -> None:
        for tool in _TOOL_SCHEMAS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_submit_analysis_has_required_fields(self) -> None:
        submit = next(t for t in _TOOL_SCHEMAS if t["name"] == "submit_analysis")
        required = set(submit["input_schema"]["required"])
        assert "name" in required
        assert "severity" in required
        assert "llm_self_score" in required
        assert "steps" in required
        assert "narrative" in required
        assert "remediation" in required


# ── _build_user_prompt ─────────────────────────────────────────────────────────


class TestBuildUserPrompt:
    def test_contains_pattern_id(self) -> None:
        prompt = _build_user_prompt(_chain())
        assert "AP-001" in prompt

    def test_contains_owasp_category(self) -> None:
        prompt = _build_user_prompt(_chain("AP-002"))
        assert "API2:2023" in prompt

    def test_contains_endpoint_ids(self) -> None:
        prompt = _build_user_prompt(_chain(
            entry_id="POST:/auth/login",
            exit_id="GET:/admin/users",
        ))
        assert "POST:/auth/login" in prompt
        assert "GET:/admin/users" in prompt

    def test_contains_hop_count(self) -> None:
        assert "1" in _build_user_prompt(_chain())

    def test_contains_sensitivity_delta(self) -> None:
        prompt = _build_user_prompt(_chain(sensitivity_delta=3))
        assert "3" in prompt

    def test_mentions_submit_analysis(self) -> None:
        assert "submit_analysis" in _build_user_prompt(_chain())

    def test_non_empty(self) -> None:
        assert _build_user_prompt(_chain()).strip()


# ── Graph tool handlers ────────────────────────────────────────────────────────


class TestGetEndpointInfo:
    def test_returns_found_false_when_no_record(self) -> None:
        session = _neo4j_session()
        result = _get_endpoint_info(session, "GET:/missing")
        assert result["found"] is False
        assert result["endpoint_id"] == "GET:/missing"

    def test_returns_node_props_when_found(self) -> None:
        session = _neo4j_session()
        fake_node = MagicMock()
        fake_node.__iter__ = MagicMock(return_value=iter([
            ("is_public", True), ("sensitivity_class", "SENSITIVE"),
        ]))
        fake_node.items = MagicMock(return_value=[
            ("is_public", True), ("sensitivity_class", "SENSITIVE"),
        ])
        record = MagicMock()
        record.__getitem__ = MagicMock(side_effect=lambda k: fake_node if k == "e" else None)
        # Patch dict() to return something sensible from the mock node
        session.run.return_value.single.return_value = record
        with patch("api_analyzer.agent.reasoner.dict", side_effect=lambda x: {"is_public": True, "sensitivity_class": "SENSITIVE"} if x is fake_node else dict(x)):
            result = _get_endpoint_info(session, "GET:/users")
        assert result.get("found") is True

    def test_session_run_called_with_endpoint_id(self) -> None:
        session = _neo4j_session()
        _get_endpoint_info(session, "POST:/auth/login")
        session.run.assert_called_once()
        call_kwargs = session.run.call_args.kwargs
        assert call_kwargs.get("id") == "POST:/auth/login"


class TestCheckAuthScheme:
    def test_returns_found_false_when_no_record(self) -> None:
        session = _neo4j_session()
        result = _check_auth_scheme(session, "GET:/missing", "api:1.0")
        assert result["found"] is False

    def test_session_run_uses_parameterized_query(self) -> None:
        session = _neo4j_session()
        _check_auth_scheme(session, "GET:/users/{id}", "test-api:1.0")
        session.run.assert_called_once()
        kwargs = session.run.call_args.kwargs
        assert kwargs.get("eid") == "GET:/users/{id}"
        assert kwargs.get("spec_id") == "test-api:1.0"


class TestGetResourceInfo:
    def test_returns_found_false_when_no_record(self) -> None:
        session = _neo4j_session()
        result = _get_resource_info(session, "Unknown", "api:1.0")
        assert result["found"] is False
        assert result["resource_name"] == "Unknown"

    def test_session_run_uses_parameterized_query(self) -> None:
        session = _neo4j_session()
        _get_resource_info(session, "User", "shop:2.0")
        session.run.assert_called_once()
        kwargs = session.run.call_args.kwargs
        assert kwargs.get("name") == "User"
        assert kwargs.get("spec_id") == "shop:2.0"


# ── _execute_tool ──────────────────────────────────────────────────────────────


class TestExecuteTool:
    def test_dispatches_get_endpoint_info(self) -> None:
        session = _neo4j_session()
        with patch("api_analyzer.agent.reasoner._get_endpoint_info",
                   return_value={"found": False}) as mock:
            _execute_tool("get_endpoint_info", {"endpoint_id": "GET:/x"}, session, "s:1")
            mock.assert_called_once_with(session, "GET:/x")

    def test_dispatches_get_resource_info(self) -> None:
        session = _neo4j_session()
        with patch("api_analyzer.agent.reasoner._get_resource_info",
                   return_value={"found": False}) as mock:
            _execute_tool("get_resource_info", {"resource_name": "User"}, session, "s:1")
            mock.assert_called_once_with(session, "User", "s:1")

    def test_dispatches_check_auth_scheme(self) -> None:
        session = _neo4j_session()
        with patch("api_analyzer.agent.reasoner._check_auth_scheme",
                   return_value={"found": False}) as mock:
            _execute_tool("check_auth_scheme", {"endpoint_id": "GET:/y"}, session, "s:1")
            mock.assert_called_once_with(session, "GET:/y", "s:1")

    def test_unknown_tool_returns_error(self) -> None:
        session = _neo4j_session()
        result = _execute_tool("nonexistent_tool", {}, session, "s:1")
        assert "error" in result
        assert "nonexistent_tool" in result["error"]


# ── _build_validated_chain ─────────────────────────────────────────────────────


class TestBuildValidatedChain:
    def _call(self, submit_input: dict | None = None, **kwargs) -> ValidatedChain | None:
        chain = _chain()
        return _build_validated_chain(
            chain=chain,
            submit_input=submit_input or _valid_submit(),
            spec_completeness=kwargs.get("spec_completeness", 0.8),
            evidence_count=kwargs.get("evidence_count", 2),
            tool_calls_used=kwargs.get("tool_calls_used", ["tool1", "tool2"]),
            total_tokens=kwargs.get("total_tokens", 500),
            config=kwargs.get("config", ReasonerConfig()),
        )

    def test_returns_validated_chain(self) -> None:
        result = self._call()
        assert isinstance(result, ValidatedChain)

    def test_candidate_id_carried_from_chain(self) -> None:
        chain = _chain()
        result = _build_validated_chain(
            chain=chain,
            submit_input=_valid_submit(),
            spec_completeness=0.8,
            evidence_count=2,
            tool_calls_used=[],
            total_tokens=500,
            config=ReasonerConfig(),
        )
        assert result is not None
        assert result.candidate_id == chain.id

    def test_pattern_id_carried_from_chain(self) -> None:
        result = self._call()
        assert result is not None
        assert result.pattern_id == "AP-001"

    def test_owasp_category_from_chain(self) -> None:
        result = self._call()
        assert result is not None
        assert result.owasp_category == "API1:2023"

    def test_name_from_submit_input(self) -> None:
        result = self._call()
        assert result is not None
        assert result.name == "BOLA → PII Exfiltration via User Detail"

    def test_severity_parsed_as_enum(self) -> None:
        result = self._call()
        assert result is not None
        assert result.severity == Severity.HIGH

    def test_steps_built_from_submit_input(self) -> None:
        result = self._call()
        assert result is not None
        assert len(result.steps) == 2
        assert all(isinstance(s, AttackStep) for s in result.steps)

    def test_step_sequence_numbers_correct(self) -> None:
        result = self._call()
        assert result is not None
        assert result.steps[0].sequence == 1
        assert result.steps[1].sequence == 2

    def test_confidence_has_correct_components(self) -> None:
        chain = _chain()
        result = _build_validated_chain(
            chain=chain,
            submit_input=_valid_submit(llm_self_score=0.9),
            spec_completeness=0.75,
            evidence_count=3,
            tool_calls_used=[],
            total_tokens=300,
            config=ReasonerConfig(),
        )
        assert result is not None
        assert result.confidence.graph_match_score == pytest.approx(chain.confidence_base)
        assert result.confidence.auth_clarity_score == pytest.approx(0.75)
        assert result.confidence.llm_self_score == pytest.approx(0.9)
        assert result.confidence.evidence_count == 3

    def test_mitre_techniques_from_submit_input(self) -> None:
        result = self._call()
        assert result is not None
        assert "T1078" in result.mitre_techniques

    def test_remediation_from_submit_input(self) -> None:
        result = self._call()
        assert result is not None
        assert len(result.remediation) >= 1

    def test_tool_calls_used_propagated(self) -> None:
        result = self._call(tool_calls_used=["call_a", "call_b"])
        assert result is not None
        assert result.tool_calls_used == ["call_a", "call_b"]

    def test_tokens_used_propagated(self) -> None:
        result = self._call(total_tokens=1234)
        assert result is not None
        assert result.tokens_used == 1234

    def test_llm_model_from_config(self) -> None:
        cfg = ReasonerConfig(llm_model="claude-opus-4-8")
        result = self._call(config=cfg)
        assert result is not None
        assert result.llm_model == "claude-opus-4-8"

    def test_analyzed_at_is_datetime(self) -> None:
        result = self._call()
        assert result is not None
        assert isinstance(result.analyzed_at, datetime)

    def test_id_is_new_uuid(self) -> None:
        r1 = self._call()
        r2 = self._call()
        assert r1 is not None and r2 is not None
        assert r1.id != r2.id

    def test_returns_none_when_llm_self_score_zero(self) -> None:
        result = self._call(_valid_submit(llm_self_score=0.0))
        assert result is None

    def test_returns_none_when_confidence_below_threshold(self) -> None:
        cfg = ReasonerConfig(confidence_threshold=0.9)
        result = self._call(config=cfg, submit_input=_valid_submit(llm_self_score=0.1))
        assert result is None

    def test_passes_when_confidence_above_threshold(self) -> None:
        cfg = ReasonerConfig(confidence_threshold=0.3)
        result = self._call(config=cfg, submit_input=_valid_submit(llm_self_score=0.9))
        assert result is not None

    def test_returns_none_on_missing_required_field(self) -> None:
        bad = _valid_submit()
        del bad["narrative"]
        result = self._call(bad)
        assert result is None

    def test_returns_none_on_invalid_severity(self) -> None:
        bad = _valid_submit()
        bad["severity"] = "ULTRA_CRITICAL"
        result = self._call(bad)
        assert result is None


# ── _analyze_chain ─────────────────────────────────────────────────────────────


class TestAnalyzeChain:
    def _run(
        self,
        responses: list[MagicMock],
        chain: CandidateChain | None = None,
        config: ReasonerConfig | None = None,
    ) -> ValidatedChain | None:
        client = _client(*responses)
        session = _neo4j_session()
        return _analyze_chain(
            chain=chain or _chain(),
            client=client,
            session=session,
            spec_id="api:1.0",
            spec_completeness=0.8,
            config=config or ReasonerConfig(),
        )

    def test_calls_messages_create(self) -> None:
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use")
        )
        session = _neo4j_session()
        _analyze_chain(_chain(), client, session, "api:1.0", 0.8, ReasonerConfig())
        client.messages.create.assert_called_once()

    def test_passes_system_prompt(self) -> None:
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use")
        )
        session = _neo4j_session()
        _analyze_chain(_chain(), client, session, "api:1.0", 0.8, ReasonerConfig())
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["system"] == _SYSTEM_PROMPT

    def test_passes_tool_schemas(self) -> None:
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use")
        )
        session = _neo4j_session()
        _analyze_chain(_chain(), client, session, "api:1.0", 0.8, ReasonerConfig())
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["tools"] == _TOOL_SCHEMAS

    def test_direct_submit_returns_validated_chain(self) -> None:
        result = self._run([
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use")
        ])
        assert isinstance(result, ValidatedChain)

    def test_returns_none_on_end_turn_without_submit(self) -> None:
        result = self._run([
            _response([_text_block("I cannot determine exploitability.")], "end_turn")
        ])
        assert result is None

    def test_returns_none_when_no_tool_calls(self) -> None:
        result = self._run([
            _response([], "end_turn")
        ])
        assert result is None

    def test_two_turn_conversation_tool_then_submit(self) -> None:
        # Turn 1: graph tool call
        # Turn 2: submit_analysis
        turn1 = _response(
            [_tool_block("get_endpoint_info", {"endpoint_id": "GET:/users"}, id="tu_1")],
            stop_reason="tool_use",
        )
        turn2 = _response(
            [_tool_block("submit_analysis", _valid_submit())],
            stop_reason="tool_use",
        )
        with patch("api_analyzer.agent.reasoner._execute_tool",
                   return_value={"found": True, "is_public": True}):
            result = self._run([turn1, turn2])
        assert isinstance(result, ValidatedChain)

    def test_evidence_count_incremented_per_tool_call(self) -> None:
        turn1 = _response(
            [_tool_block("get_endpoint_info", {"endpoint_id": "GET:/users"}, id="tu_1")],
            stop_reason="tool_use",
        )
        turn2 = _response(
            [_tool_block("submit_analysis", _valid_submit(llm_self_score=0.9))],
            stop_reason="tool_use",
        )
        with patch("api_analyzer.agent.reasoner._execute_tool",
                   return_value={"found": True}):
            result = self._run([turn1, turn2])
        assert result is not None
        assert result.confidence.evidence_count == 1

    def test_tokens_accumulated_across_turns(self) -> None:
        turn1 = _response(
            [_tool_block("get_endpoint_info", {"endpoint_id": "GET:/x"}, id="t1")],
            stop_reason="tool_use",
            input_tokens=200, output_tokens=100,
        )
        turn2 = _response(
            [_tool_block("submit_analysis", _valid_submit())],
            stop_reason="tool_use",
            input_tokens=300, output_tokens=150,
        )
        with patch("api_analyzer.agent.reasoner._execute_tool", return_value={"found": False}):
            result = self._run([turn1, turn2])
        assert result is not None
        assert result.tokens_used == 750  # 200+100+300+150

    def test_budget_exhausted_returns_none(self) -> None:
        # max_tool_calls_per_chain=2 means 3 turns max; never call submit_analysis
        tool_resp = _response(
            [_tool_block("get_endpoint_info", {"endpoint_id": "GET:/x"}, id="t1")],
            stop_reason="tool_use",
        )
        cfg = ReasonerConfig(max_tool_calls_per_chain=2)
        with patch("api_analyzer.agent.reasoner._execute_tool", return_value={"found": False}):
            result = self._run([tool_resp, tool_resp, tool_resp], config=cfg)
        assert result is None

    def test_text_blocks_ignored(self) -> None:
        # First turn has text + submit_analysis
        turn = _response(
            [_text_block("Let me analyze..."),
             _tool_block("submit_analysis", _valid_submit())],
            stop_reason="tool_use",
        )
        result = self._run([turn])
        assert isinstance(result, ValidatedChain)

    def test_returns_none_when_submit_below_threshold(self) -> None:
        turn = _response(
            [_tool_block("submit_analysis", _valid_submit(llm_self_score=0.0))],
            stop_reason="tool_use",
        )
        result = self._run([turn])
        assert result is None


# ── analyze() ─────────────────────────────────────────────────────────────────


class TestAnalyze:
    def _make_client_and_driver(self, submit_input=None):
        submit = submit_input or _valid_submit()
        client = _client(
            _response([_tool_block("submit_analysis", submit)], "tool_use")
        )
        driver = _neo4j_driver()
        return client, driver

    def test_returns_list(self) -> None:
        client, driver = self._make_client_and_driver()
        result = analyze(_traversal_result([_chain()]), driver, client=client)
        assert isinstance(result, list)

    def test_empty_chains_returns_empty_list(self) -> None:
        driver = _neo4j_driver()
        result = analyze(_traversal_result([]), driver, client=MagicMock())
        assert result == []

    def test_validated_chain_returned_for_good_candidate(self) -> None:
        client, driver = self._make_client_and_driver()
        result = analyze(_traversal_result([_chain()]), driver, client=client)
        assert len(result) == 1
        assert isinstance(result[0], ValidatedChain)

    def test_failed_analysis_filtered_out(self) -> None:
        # llm_self_score=0.0 → confidence kills the chain → None → filtered out
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit(llm_self_score=0.0))], "tool_use")
        )
        driver = _neo4j_driver()
        result = analyze(_traversal_result([_chain()]), driver, client=client)
        assert result == []

    def test_multiple_chains_all_processed(self) -> None:
        chains = [_chain("AP-001"), _chain("AP-002"), _chain("AP-007")]
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use"),
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use"),
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use"),
        )
        driver = _neo4j_driver()
        result = analyze(_traversal_result(chains), driver, client=client)
        assert len(result) == 3

    def test_driver_session_opened_per_chain(self) -> None:
        chains = [_chain(), _chain()]
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use"),
            _response([_tool_block("submit_analysis", _valid_submit())], "tool_use"),
        )
        driver = _neo4j_driver()
        analyze(_traversal_result(chains), driver, client=client)
        assert driver.session.call_count == 2

    def test_uses_default_config_when_none_provided(self) -> None:
        client, driver = self._make_client_and_driver()
        result = analyze(_traversal_result([_chain()]), driver, client=client, config=None)
        assert isinstance(result, list)

    def test_spec_completeness_from_traversal_result(self) -> None:
        tr = TraversalResult(
            spec_id="api:1.0", chains=[_chain()], total_candidates=1,
            spec_completeness=0.99, endpoint_count=5,
            candidate_counts={"bola": 1},
        )
        client = _client(
            _response([_tool_block("submit_analysis", _valid_submit(llm_self_score=0.8))], "tool_use")
        )
        driver = _neo4j_driver()
        result = analyze(tr, driver, client=client)
        assert len(result) == 1
        assert result[0].confidence.auth_clarity_score == pytest.approx(0.99)

    def test_creates_default_client_when_none(self) -> None:
        with patch("api_analyzer.agent.reasoner.anthropic.Anthropic") as MockAnthropic:
            mock_client = _client(
                _response([_tool_block("submit_analysis", _valid_submit())], "tool_use")
            )
            MockAnthropic.return_value = mock_client
            result = analyze(_traversal_result([_chain()]), _neo4j_driver(), client=None)
            MockAnthropic.assert_called_once()
        assert isinstance(result, list)
