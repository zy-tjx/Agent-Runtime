"""
LLM 异常压测
验证 PLANNER / DECIDE / REFLECT 的降级路径在真实异常下是否按预期工作
"""
import pytest
from unittest.mock import patch, Mock
from engine.model_manager import ModelManager
from runtime.state_graph import run_graph


# ── 模拟异常工厂 ──

def _make_failing_generate(failures: list[Exception | str], final: str):
    """
    返回一个 generate 替代函数，前 N 次抛出指定异常，最后返回正常结果
    failures: 异常列表
    final: 正常返回的文本
    """
    calls = [0]

    def _mock(self, prompt: str) -> str:
        idx = calls[0]
        calls[0] += 1
        if idx < len(failures):
            err = failures[idx]
            if isinstance(err, Exception):
                raise err
            return err
        return final

    return _mock


# ============================================================
# 1. PLANNER 降级测试
# ============================================================

class TestPlannerFallback:
    def test_llm_connection_error_triggers_fallback(self):
        """PLANNER LLM 超时 → 应走模板降级"""
        mock_gen = _make_failing_generate(
            [ConnectionError("API connection timeout")],
            '{"mode": "qa", "topic": "测试", "goals": ["测试"], "modules": [], "estimated_total_minutes": 0}',
        )
        with patch.object(ModelManager, 'generate', mock_gen):
            result = run_graph("测试问题", session_id="fail-planner")
        assert result.get("fallback_triggered") is True
        plan = result.get("plan", {})
        assert plan.get("topic") is not None

    def test_planner_dirty_json_fallback(self):
        """PLANNER 返回脏 JSON → 应走模板降级"""
        mock_gen = _make_failing_generate(
            ["{not valid json!!}"],
            '{"mode": "qa", "topic": "测试", "goals": ["测试"], "modules": [], "estimated_total_minutes": 0}',
        )
        with patch.object(ModelManager, 'generate', mock_gen):
            result = run_graph("测试问题", session_id="fail-planner-dirty")
        assert result.get("fallback_triggered") is True

    def test_planner_empty_response_fallback(self):
        """PLANNER 返回空 → 应走模板降级"""
        mock_gen = _make_failing_generate(
            [""],
            '{"mode": "qa", "topic": "测试", "goals": ["测试"], "modules": [], "estimated_total_minutes": 0}',
        )
        with patch.object(ModelManager, 'generate', mock_gen):
            result = run_graph("", session_id="fail-planner-empty")
        # 空输入 + 空响应 → 降级路径仍应产生 plan
        plan = result.get("plan", {})
        assert plan.get("topic") is not None


# ============================================================
# 2. DECIDE 降级测试
# ============================================================

class TestDecideFallback:
    def test_decide_llm_failure_uses_keyword(self):
        """DECIDE LLM 失败 → 应走关键词匹配"""
        mock_gen = _make_failing_generate(
            [ConnectionError("timeout")],
            '{"tool_name": "search_docs", "reason": "test", "confidence": 1.0}',
        )
        with patch.object(ModelManager, 'generate', mock_gen):
            result = run_graph("学习 AI 系统", session_id="fail-decide")
        decision = result.get("decision", {})
        assert decision.get("tool_name") is not None


# ============================================================
# 3. REFLECT 降级测试
# ============================================================

class TestReflectFallback:
    def test_reflect_llm_failure_uses_rule_engine(self):
        """REFLECT LLM 失败 → 应走规则引擎降级（非硬编码 end）"""
        # PLANNER 正常, DECIDE 正常, 但 REFLECT 崩溃
        fail_count = [0]

        def _mock(self, prompt: str) -> str:
            fail_count[0] += 1
            # 前两次调用（PLANNER + DECIDE）正常
            if fail_count[0] <= 2:
                if "规划模块" in prompt or "planner" in prompt.lower():
                    return '{"mode": "learn", "topic": "压力测试", "goals": ["测试"], "modules": [{"title": "基础", "content": "内容", "duration_minutes": 30}], "estimated_total_minutes": 30}'
                if "决策模块" in prompt or "工具" in prompt:
                    return '{"tool_name": "search_docs", "reason": "测试", "confidence": 1.0}'
            # REFLECT 调用失败
            raise ConnectionError("REFLECT timeout")

        with patch.object(ModelManager, 'generate', _mock):
            result = run_graph("学习压力测试", session_id="fail-reflect")

        assert result.get("fallback_triggered") is True
        assert result.get("fallback_reason", "").startswith("REFLECT:")
        reflection = result.get("reflection", {})
        assert reflection.get("next_action") is not None


# ============================================================
# 4. 恢复后继续正常
# ============================================================

class TestRecoveryThenNormal:
    def test_failed_planner_then_normal(self):
        """一次 PLANNER 失败后，下次正常调用应恢复"""
        call_count = [0]

        def _mock(self, prompt: str) -> str:
            call_count[0] += 1
            if call_count[0] == 1 and "规划模块" in prompt:
                raise ConnectionError("timeout")
            if "规划模块" in prompt:
                return '{"mode": "learn", "topic": "恢复测试", "goals": ["测试恢复"], "modules": [{"title": "模块", "content": "内容", "duration_minutes": 30}], "estimated_total_minutes": 30}'
            if "决策模块" in prompt:
                return '{"tool_name": "search_docs", "reason": "恢复", "confidence": 1.0}'
            return '{"confidence": 0.8, "is_satisfactory": true, "next_action": "end", "retry_target_node": null, "hallucination_flag": false, "error_root_cause": null, "improvement_suggestion": null}'

        with patch.object(ModelManager, 'generate', _mock):
            result = run_graph("恢复测试", session_id="recovery")
        assert result.get("fallback_triggered") is True
        assert result.get("current_step") == "REFLECT"


# ============================================================
# 5. 全面崩溃 → 仍应返回结果不抛异常
# ============================================================

class TestGracefulDegradation:
    def test_all_llm_calls_fail_still_returns_state(self):
        """所有 LLM 调用全部失败 → 系统不应崩溃"""
        def _mock(self, prompt: str) -> str:
            raise ConnectionError("API 不可用")

        with patch.object(ModelManager, 'generate', _mock):
            result = run_graph("任何问题", session_id="total-failure")
        # 系统不应崩溃，应有最终 state
        assert result.get("current_step") is not None
        assert result.get("reflection", {}).get("next_action") is not None
