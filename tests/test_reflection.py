"""
Phase 9-10 反思模块测试
覆盖：error_analysis / improvement_suggestion / self_reflection / 经验记忆闭环
"""
import pytest
from reflection.error_analysis import analyze, _classify_error_string
from reflection.improvement_suggestion import suggest, _infer_target_from_fallback
from reflection.self_reflection import evaluate


# ============================================================
# 1. error_analysis 测试
# ============================================================

class TestErrorAnalysis:
    def test_no_error(self):
        r = analyze()
        assert r["error_type"] == "none"
        assert r["recoverable"] is False
        assert r["severity"] == "none"

    def test_max_retries_exhausted(self):
        r = analyze(retry_count=3, max_retries=3)
        assert r["error_type"] == "max_retries_exhausted"
        assert r["recoverable"] is False
        assert r["severity"] == "high"

    def test_max_retries_takes_priority_over_other_errors(self):
        """已达上限时，即使有其他错误也优先返回 max_retries"""
        r = analyze(
            fallback_triggered=True, retry_count=3, max_retries=3,
            fallback_reason="PLANNER: error"
        )
        assert r["error_type"] == "max_retries_exhausted"

    def test_fallback_triggered(self):
        r = analyze(fallback_triggered=True, fallback_reason="DECIDE: Timeout")
        assert r["error_type"] == "llm_error"
        assert r["recoverable"] is True
        assert r["severity"] == "medium"

    def test_explicit_error(self):
        r = analyze(error="timeout occurred")
        assert r["error_type"] == "llm_error"
        assert r["recoverable"] is True

    def test_tool_failure(self):
        r = analyze(tool_results=[
            {"tool_name": "search_docs", "status": "failed"},
            {"tool_name": "memory_write", "status": "success"},
        ])
        assert r["error_type"] == "tool_failure"
        assert r["recoverable"] is True

    def test_tool_all_success(self):
        r = analyze(tool_results=[
            {"tool_name": "search_docs", "status": "success"},
        ])
        assert r["error_type"] == "none"

    def test_retrieval_empty_when_attempted(self):
        r = analyze(retrieval_attempted=True, retrieved_context=[])
        assert r["error_type"] == "retrieval_empty"
        assert r["recoverable"] is True
        assert r["severity"] == "low"

    def test_retrieval_not_flagged_when_not_attempted(self):
        """未尝试检索时不应标记为 retrieval_empty"""
        r = analyze(retrieval_attempted=False, retrieved_context=[])
        assert r["error_type"] == "none"

    def test_classify_string_timeout(self):
        assert _classify_error_string("ConnectionError: timed out") == "llm_error"

    def test_classify_string_tool(self):
        assert _classify_error_string("ToolExecutor: execution failed") == "tool_failure"

    def test_classify_string_retrieval(self):
        assert _classify_error_string("VectorRetriever: search error") == "retrieval_empty"

    def test_classify_string_parse(self):
        assert _classify_error_string("JSON parse error in structured output") == "llm_error"

    def test_classify_string_unknown(self):
        assert _classify_error_string("something unexpected") == "unknown"


# ============================================================
# 2. improvement_suggestion 测试
# ============================================================

class TestImprovementSuggestion:
    def test_unrecoverable_ends(self):
        err = {"error_type": "max_retries_exhausted", "recoverable": False, "severity": "high", "summary": "max"}
        s = suggest(error_analysis=err)
        assert s["next_action"] == "end"
        assert s["reason_code"] == "MAX_RETRIES"
        assert s["retry_target_node"] is None

    def test_llm_error_retries_planner(self):
        err = {"error_type": "llm_error", "recoverable": True, "severity": "medium", "summary": "llm fail"}
        s = suggest(error_analysis=err, fallback_reason="PLANNER: ConnectionError")
        assert s["next_action"] == "retry"
        assert s["reason_code"] == "LLM_ERROR"
        assert s["retry_target_node"] == "PLANNER"

    def test_decide_fallback_routes_to_planner(self):
        """DECIDE LLM 失败时回退到 PLANNER（DECIDE 不是合法重试目标）"""
        assert _infer_target_from_fallback("DECIDE: Timeout") == "PLANNER"

    def test_reflect_fallback_routes_to_planner(self):
        """REFLECT 自身失败保守回退到 PLANNER"""
        assert _infer_target_from_fallback("REFLECT: error") == "PLANNER"

    def test_unknown_fallback_defaults_planner(self):
        assert _infer_target_from_fallback(None) == "PLANNER"

    def test_tool_failure_retries_execute(self):
        err = {"error_type": "tool_failure", "recoverable": True, "severity": "medium", "summary": "tool fail"}
        s = suggest(error_analysis=err)
        assert s["next_action"] == "retry"
        assert s["reason_code"] == "TOOL_FAILURE"
        assert s["retry_target_node"] == "EXECUTE"

    def test_retrieval_empty_retries_retrieve(self):
        err = {"error_type": "retrieval_empty", "recoverable": True, "severity": "low", "summary": "empty"}
        s = suggest(error_analysis=err)
        assert s["next_action"] == "retry"
        assert s["reason_code"] == "EMPTY_RETRIEVAL"
        assert s["retry_target_node"] == "RETRIEVE"

    def test_low_groundedness_qa_with_hallucination_ends(self):
        """QA 模式接地低 + 幻觉告警 → 应结束而非重试"""
        err = {"error_type": "none", "recoverable": False, "severity": "none", "summary": ""}
        s = suggest(
            error_analysis=err, mode="qa",
            groundedness_score=0.3, hallucination_flag=True
        )
        assert s["next_action"] == "end"
        assert s["reason_code"] == "LOW_GROUNDEDNESS"

    def test_low_groundedness_learn_retries(self):
        """learn 模式接地低无幻觉 → 重试 RETRIEVE"""
        err = {"error_type": "none", "recoverable": False, "severity": "none", "summary": ""}
        s = suggest(
            error_analysis=err, mode="learn",
            groundedness_score=0.3, hallucination_flag=False
        )
        assert s["next_action"] == "retry"
        assert s["retry_target_node"] == "RETRIEVE"

    def test_no_signals_ends(self):
        err = {"error_type": "none", "recoverable": False, "severity": "none", "summary": ""}
        s = suggest(error_analysis=err)
        assert s["next_action"] == "end"
        assert s["reason_code"] == "NONE"

    def test_rationale_included(self):
        err = {"error_type": "tool_failure", "recoverable": True, "severity": "medium", "summary": "x"}
        s = suggest(error_analysis=err)
        assert "rationale" in s
        assert len(s["rationale"]) > 0


# ============================================================
# 3. self_reflection 测试
# ============================================================

class TestSelfReflection:
    def test_perfect_scores(self):
        r = evaluate(groundedness_score=1.0, completeness_score=1.0)
        assert r["confidence"] == 0.7
        assert r["factors"]["groundedness_contrib"] == 0.4
        assert r["factors"]["completeness_contrib"] == 0.3

    def test_missing_scores_default_to_neutral(self):
        """缺失分用 0.5 中性默认"""
        r = evaluate()
        assert r["confidence"] == 0.35  # 0.2 + 0.15
        assert r["hallucination_penalty_applied"] is False

    def test_fallback_penalty(self):
        r_base = evaluate(groundedness_score=0.8, completeness_score=0.8)
        r_fb = evaluate(groundedness_score=0.8, completeness_score=0.8, fallback_triggered=True)
        assert r_fb["confidence"] < r_base["confidence"]
        assert r_fb["factors"]["fallback_penalty"] == -0.2

    def test_hallucination_penalty(self):
        r = evaluate(groundedness_score=0.8, completeness_score=0.8, hallucination_flag=True)
        assert r["confidence"] < 0.5
        assert r["factors"]["hallucination_penalty"] == -0.3
        assert r["hallucination_penalty_applied"] is True

    def test_confidence_clamped_to_zero(self):
        """信心分不低于 0"""
        r = evaluate(groundedness_score=0.1, completeness_score=0.1,
                     fallback_triggered=True, hallucination_flag=True)
        assert r["confidence"] >= 0.0

    def test_confidence_clamped_to_one(self):
        """信心分不超过 1"""
        r = evaluate(groundedness_score=1.0, completeness_score=1.0)
        assert r["confidence"] <= 1.0


# ============================================================
# 4. 经验记忆闭环 测试
# ============================================================

class TestExperienceMemory:
    def test_load_recent_experiences(self):
        """验证能读取到历史经验记录"""
        from memory.long_term_memory import load_recent_experiences
        records = load_recent_experiences(limit=3)
        assert isinstance(records, list)
        # 数据库中应该有之前运行留下的经验
        assert len(records) >= 1

    def test_load_experience_summaries_format(self):
        """验证摘要格式：每行以时间戳开头，含决策信息"""
        from memory.long_term_memory import load_experience_summaries
        summaries = load_experience_summaries(limit=2)
        assert isinstance(summaries, list)
        assert len(summaries) >= 1
        for s in summaries:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_experience_summaries_empty_db(self):
        """空库时返回友好提示"""
        from memory.long_term_memory import load_experience_summaries
        summaries = load_experience_summaries(mode="nonexistent_mode", limit=3)
        assert summaries == ["无历史经验"]

    def test_experience_write_then_read(self):
        """写入后立即可读到"""
        from memory.long_term_memory import save, MemoryRecord, load_recent_experiences
        import time
        key = f"test_exp_{int(time.time() * 1000)}"
        save(MemoryRecord(
            key=key,
            value={"mode": "learn", "reflection": {
                "next_action": "retry", "error_root_cause": "test error",
                "is_satisfactory": False
            }},
            category="experience",
        ))
        records = load_recent_experiences(limit=10)
        keys = [r["key"] for r in records]
        assert key in keys

    def test_mode_filter(self):
        """按模式筛选正确"""
        from memory.long_term_memory import load_recent_experiences
        import time
        learn_records = load_recent_experiences(mode="learn", limit=10)
        qa_records = load_recent_experiences(mode="qa", limit=10)
        for r in learn_records:
            assert r["value"].get("mode") == "learn"
        for r in qa_records:
            assert r["value"].get("mode") == "qa"
