"""
Phase 8 可观测性模块测试
覆盖：logger / langsmith_tracer / eval_metrics / hallucination_detector
"""
import pytest
from observability.logger import NodeLogger, get_logger, _summarize
from observability.tracer import (
    TraceCollector, get_tracer, reset_tracer, trace_node,
    _compute_diff, _extract_tool_calls,
)
from reflection.eval_metrics import compute_metrics
from reflection.hallucination_detector import detect


# ============================================================
# 1. Logger 测试
# ============================================================

class TestLogger:
    def test_node_start_end_contract(self, capsys):
        logger = NodeLogger("test-session")
        state = {"user_input": "你好", "mode": "learn", "error": None}
        logger.node_start("PLANNER", state)
        logger.node_end("PLANNER", state)

        captured = capsys.readouterr().out
        lines = [l for l in captured.strip().split("\n") if l]
        assert len(lines) == 2
        assert "node_start" in lines[0]
        assert "node_end" in lines[1]
        assert "test-session" in lines[0]
        assert "PLANNER" in lines[0]

    def test_node_error(self, capsys):
        logger = NodeLogger()
        logger.node_error("RETRIEVE", "向量检索超时")
        captured = capsys.readouterr().out
        assert "node_error" in captured
        assert "向量检索超时" in captured

    def test_info(self, capsys):
        logger = NodeLogger()
        logger.info("测试消息", key="value")
        captured = capsys.readouterr().out
        assert "info" in captured
        assert "测试消息" in captured

    def test_duration_ms_auto_computed(self, capsys):
        logger = NodeLogger()
        logger.node_start("EXECUTE", {})
        logger.node_end("EXECUTE", {})
        captured = capsys.readouterr().out
        assert "duration_ms" in captured

    def test_singleton_same_session(self):
        l1 = get_logger("s1")
        l2 = get_logger("s1")
        assert l1 is l2

    def test_singleton_different_session(self):
        l1 = get_logger("s1")
        l2 = get_logger("s2")
        assert l1 is not l2

    def test_summarize_extracts_governance(self):
        state = {
            "mode": "qa", "retrieval_score": 0.9,
            "groundedness_score": 0.8, "completeness_score": 0.7,
            "answer_source": "rag", "retry_reason": None,
            "retry_count": 1, "error": "timeout",
            "current_step": "REFLECT",
            "user_input": "a" * 100,  # 超过 80 字符会截断
        }
        s = _summarize(state)
        assert s["mode"] == "qa"
        assert s["retrieval_score"] == 0.9
        assert s["current_step"] == "REFLECT"
        assert len(s["user_input"]) <= 83  # 80 + "..." 约 83

    def test_summarize_handles_missing_fields(self):
        s = _summarize({})
        assert s == {}


# ============================================================
# 2. LangSmith Tracer 测试
# ============================================================

class TestTraceCollector:
    def test_record_and_retrieve(self):
        reset_tracer()
        t = get_tracer()
        t.record("planner", 100, ["plan"], {"plan": "updated"}, [])
        assert len(t.records) == 1
        assert t.records[0]["node"] == "planner"
        assert t.records[0]["duration_ms"] == 100

    def test_summary(self):
        reset_tracer()
        t = get_tracer()
        t.record("planner", 100, ["plan"], {}, [])
        t.record("execute", 200, ["tool_results"], {}, [
            {"tool_name": "search_docs", "called_at": "2024-01-01"}
        ])
        s = t.summary()
        assert s["total_duration_ms"] == 300
        assert s["node_count"] == 2
        assert s["tool_calls_total"] == 1
        assert s["nodes_visited"] == ["planner", "execute"]

    def test_reset_clears_records(self):
        t = get_tracer()
        t.record("planner", 10, [], {}, [])
        reset_tracer()
        t2 = get_tracer()
        assert len(t2.records) == 0


class TestStateDiff:
    def test_scalar_change(self):
        diff = _compute_diff({"a": 1}, {"a": 2})
        assert diff["a"] == {"before": 1, "after": 2}

    def test_list_change(self):
        diff = _compute_diff({"x": []}, {"x": [1, 2]})
        assert diff["x"] == "updated"

    def test_dict_change(self):
        diff = _compute_diff({"d": {}}, {"d": {"k": "v"}})
        assert diff["d"] == "updated"

    def test_no_change(self):
        diff = _compute_diff({"a": 1, "b": "same"}, {"a": 1, "b": "same"})
        assert diff == {}

    def test_new_field(self):
        diff = _compute_diff({}, {"new_field": 42})
        assert diff["new_field"] == {"before": None, "after": 42}

    def test_messages_diff(self):
        diff = _compute_diff(
            {"messages": [{"role": "human"}]},
            {"messages": [{"role": "human"}, {"role": "ai"}]},
        )
        assert "messages" in diff
        assert "1 → 2" in diff["messages"]


class TestExtractToolCalls:
    def test_empty(self):
        assert _extract_tool_calls({}) == []

    def test_normal(self):
        result = {
            "tool_calls": [
                {"tool_name": "search_docs", "called_at": "2024-01-01T00:00:00", "arguments": {}},
            ]
        }
        extracted = _extract_tool_calls(result)
        assert len(extracted) == 1
        assert extracted[0]["tool_name"] == "search_docs"


class TestTraceNodeWrapper:
    def test_wrapper_returns_result(self):
        reset_tracer()

        def fake_node(state):
            return {"new_field": 42}

        wrapped = trace_node("test_node", fake_node)
        result = wrapped({})
        assert result == {"new_field": 42}

    def test_wrapper_records_trace(self):
        reset_tracer()

        def fake_node(state):
            return {"step": "done"}

        wrapped = trace_node("test_node", fake_node)
        wrapped({"input": "hello"})
        records = get_tracer().records
        assert len(records) == 1
        assert records[0]["node"] == "test_node"
        assert "step" in records[0]["fields_produced"]


# ============================================================
# 3. Eval Metrics 测试
# ============================================================

class TestEvalMetrics:
    def test_rag_metrics_with_context(self):
        state = {
            "retrieved_context": [{"doc_id": "1"}, {"doc_id": "2"}],
            "retrieval_score": 0.85,
        }
        m = compute_metrics(state)
        assert m["rag"]["docs_retrieved"] == 2
        assert m["rag"]["retrieval_score"] == 0.85
        assert m["rag"]["has_context"] is True

    def test_rag_metrics_empty(self):
        m = compute_metrics({})
        assert m["rag"]["docs_retrieved"] == 0
        assert m["rag"]["has_context"] is False

    def test_tool_metrics_success(self):
        state = {
            "tool_calls": [
                {"tool_name": "search_docs", "called_at": "2024-01-01"},
            ],
            "tool_results": [
                {"tool_name": "search_docs", "status": "success", "duration_ms": 150},
            ],
        }
        m = compute_metrics(state)
        assert m["tool"]["tool_success_rate"] == 1.0
        assert m["tool"]["total_tool_duration_ms"] == 150
        assert "search_docs" in m["tool"]["tools_used"]

    def test_tool_metrics_mixed(self):
        state = {
            "tool_calls": [{"tool_name": "a"}, {"tool_name": "b"}],
            "tool_results": [
                {"tool_name": "a", "status": "success"},
                {"tool_name": "b", "status": "failed"},
            ],
        }
        m = compute_metrics(state)
        assert m["tool"]["tool_success_rate"] == 0.5
        assert m["tool"]["tool_failure_count"] == 1

    def test_tool_metrics_empty(self):
        m = compute_metrics({})
        assert m["tool"]["tool_success_rate"] is None
        assert m["tool"]["tool_calls_total"] == 0

    def test_answer_metrics_rag(self):
        state = {"final_output": "RAG 是检索增强生成...", "answer_source": "rag"}
        m = compute_metrics(state)
        assert m["answer"]["has_answer"] is True
        assert m["answer"]["is_rag_sourced"] is True
        assert m["answer"]["answer_length_chars"] > 0

    def test_answer_metrics_fallback(self):
        state = {"final_output": "fallback 回答", "answer_source": "llm_fallback"}
        m = compute_metrics(state)
        assert m["answer"]["is_fallback"] is True
        assert m["answer"]["is_rag_sourced"] is False

    def test_answer_metrics_empty(self):
        m = compute_metrics({})
        assert m["answer"]["has_answer"] is False

    def test_flow_metrics_nodes_visited(self):
        state = {
            "plan": {"topic": "AI"},
            "decision": {"tool_name": "search_docs"},
            "retrieval_score": 0.5,
            "tool_results": [{"status": "success"}],
            "reflection": {"confidence": 0.8},
        }
        m = compute_metrics(state)
        assert m["flow"]["nodes_visited"] == [
            "PLANNER", "DECIDE", "RETRIEVE", "EXECUTE", "REFLECT"
        ]
        assert m["flow"]["nodes_visited_count"] == 5

    def test_flow_metrics_partial(self):
        state = {"plan": {"topic": "AI"}, "error": "timeout"}
        m = compute_metrics(state)
        assert m["flow"]["nodes_visited"] == ["PLANNER"]
        assert m["flow"]["has_error"] is True

    def test_flow_metrics_retry(self):
        state = {"retry_count": 2, "max_retries": 3}
        m = compute_metrics(state)
        assert m["flow"]["did_retry"] is True
        assert m["flow"]["retries_exhausted"] is False

    def test_flow_metrics_retries_exhausted(self):
        state = {"retry_count": 3, "max_retries": 3}
        m = compute_metrics(state)
        assert m["flow"]["retries_exhausted"] is True

    def test_governance_passthrough(self):
        state = {
            "mode": "qa", "retrieval_score": 0.9,
            "groundedness_score": 0.8, "completeness_score": 0.7,
            "answer_source": "rag", "retry_reason": None,
        }
        m = compute_metrics(state)
        g = m["governance"]
        assert g["mode"] == "qa"
        assert g["groundedness_score"] == 0.8
        assert g["completeness_score"] == 0.7


# ============================================================
# 4. Hallucination Detector 测试
# ============================================================

class TestHallucinationDetector:
    def test_normal_no_flag(self):
        """正常情况不应触发幻觉告警"""
        result = detect({
            "groundedness_score": 0.8,
            "completeness_score": 0.7,
            "answer_source": "rag",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "正常回答",
        })
        assert result["flag"] is False
        assert result["severity"] == "none"

    def test_rule1_rag_hallucination(self):
        """主规则：接地低 + RAG来源 + 有上下文 + 完整度尚可"""
        result = detect({
            "groundedness_score": 0.3,
            "completeness_score": 0.6,
            "answer_source": "rag",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "回答内容",
        })
        assert result["flag"] is True
        assert "rag_hallucination" in result["rules_triggered"]
        assert result["severity"] == "high"

    def test_rule2_empty_context_mismatch(self):
        """规则2：声称RAG但检索上下文为空"""
        result = detect({
            "groundedness_score": 0.8,
            "completeness_score": 0.7,
            "answer_source": "rag",
            "retrieved_context": [],
            "final_output": "回答",
        })
        assert result["flag"] is True
        assert "empty_context_mismatch" in result["rules_triggered"]
        assert result["severity"] == "medium"

    def test_rule3_long_answer_low_groundedness(self):
        """规则3：长回答 + 接地极低"""
        result = detect({
            "groundedness_score": 0.15,
            "completeness_score": 0.6,
            "answer_source": "rag",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "x" * 250,
        })
        assert result["flag"] is True
        triggered = result["rules_triggered"]
        assert "long_answer_low_groundedness" in triggered or "rag_hallucination" in triggered

    def test_low_groundedness_but_llm_fallback(self):
        """非RAG来源，即使接地低也不触发主规则"""
        result = detect({
            "groundedness_score": 0.3,
            "completeness_score": 0.7,
            "answer_source": "llm_fallback",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "回答",
        })
        # 规则1不应触发（answer_source != rag），规则3可能触发如果答案够长
        # 这里答案短，不触发任何规则
        assert "rag_hallucination" not in result.get("rules_triggered", [])

    def test_evidence_included(self):
        result = detect({
            "groundedness_score": 0.3,
            "completeness_score": 0.6,
            "answer_source": "rag",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "回答",
        })
        assert "evidence" in result
        assert result["evidence"]["groundedness_score"] == 0.3

    def test_empty_state(self):
        """空 state 不应崩溃"""
        result = detect({})
        assert result["flag"] is False
        assert result["severity"] == "none"

    def test_reason_combined_when_multiple_rules(self):
        """多规则触发时 reason 应合并"""
        result = detect({
            "groundedness_score": 0.15,
            "completeness_score": 0.6,
            "answer_source": "rag",
            "retrieved_context": [{"doc_id": "1"}],
            "final_output": "x" * 250,
        })
        if len(result.get("rules_triggered", [])) >= 2:
            assert "；" in result["reason"]
