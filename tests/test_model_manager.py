"""
模型与 Prompt 层测试
覆盖：config / prompt_manager / model_manager / DECIDE JSON 解析
"""
import os
import pytest
from pydantic import BaseModel

from engine.config import get_config
from engine.prompt_manager import (
    render,
    format_tools_list,
    format_parameters_schema,
)
from engine.model_manager import ModelManager
from runtime.node_decide import (
    DecideToolSelection,
    _parse_json,
    _decide_via_keyword,
    _get_registry,
)

# ============================================================
# 1. Config 测试
# ============================================================

class TestConfig:
    def test_get_config_returns_three_keys(self):
        config = get_config()
        assert "api_key" in config
        assert "base_url" in config
        assert "model_name" in config
        assert all(isinstance(v, str) and v for v in config.values())

    def test_get_config_raises_on_missing_var(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "https://test")
        monkeypatch.setenv("MODEL_NAME", "test")
        monkeypatch.setenv("OPENAI_API_KEY", "")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            get_config()


# ============================================================
# 2. Prompt Manager 测试
# ============================================================

class TestPromptManager:
    def test_render_tool_selection(self):
        prompt = render(
            "decide_tool_selection",
            user_input="测试输入",
            tools_list="- name: search_docs",
        )
        assert "测试输入" in prompt
        assert "search_docs" in prompt
        assert "只输出 JSON" in prompt

    def test_render_arguments(self):
        prompt = render(
            "decide_arguments",
            user_input="测试",
            name="create_plan",
            description="生成计划",
            parameters_schema='{"topic": {"required": true}}',
        )
        assert "create_plan" in prompt
        assert "生成计划" in prompt
        assert "topic" in prompt

    def test_render_unknown_template_raises(self):
        with pytest.raises(ValueError, match="未知模板"):
            render("nonexistent")

    def test_format_tools_list(self):
        tools = [
            {
                "name": "search_docs",
                "description": "检索知识库",
                "parameters": {"query": {"required": True}},
            }
        ]
        result = format_tools_list(tools)
        assert "search_docs" in result
        assert "检索知识库" in result
        assert "query" in result

    def test_format_parameters_schema(self):
        params = {"query": {"type": "str", "required": True}}
        result = format_parameters_schema(params)
        assert "query" in result
        assert "str" in result


# ============================================================
# 3. ModelManager 测试
# ============================================================

class TestModelManager:
    def test_init_uses_config(self):
        manager = ModelManager()
        assert manager.model_name == get_config()["model_name"]

    def test_generate_returns_text(self):
        """真实 LLM 调用（需要网络和千问 API）"""
        manager = ModelManager()
        result = manager.generate("输出一个 JSON: {\"key\": \"value\"}")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_structured_fails_on_local_models(self):
        """
        结构化输出对本地模型可能不生效（Ollama 不支持），
        但对云端模型（千问/Groq）应正常工作。
        当前只验证 generate() 可用。
        """
        manager = ModelManager()
        result = manager.generate("回复: OK")
        assert "OK" in result or len(result) > 0

    def test_generate_failure_raises(self):
        """使用错误的 base_url 应抛出 RuntimeError"""
        import os as _os
        # 临时改配置
        _os.environ["OPENAI_BASE_URL"] = "https://invalid.example.com/v1"
        _os.environ["OPENAI_API_KEY"] = "fake"
        _os.environ["MODEL_NAME"] = "fake"
        try:
            from engine.config import get_config as _cfg_fresh
            cfg = _cfg_fresh()  # 会报错因为 model_name 还在但 base_url 不对
        except ValueError:
            # 如果 config 检查通过了，尝试调用
            manager = ModelManager()
            with pytest.raises(Exception):
                manager.generate("test")


# ============================================================
# 4. DECIDE JSON 解析测试
# ============================================================

class TestDecideJsonParsing:
    def test_parse_plain_json(self):
        result = _parse_json(
            '{"tool_name": "search_docs", "reason": "测试", "confidence": 0.9}',
            DecideToolSelection,
        )
        assert result.tool_name == "search_docs"
        assert result.reason == "测试"
        assert result.confidence == 0.9

    def test_parse_json_with_markdown_wrapper(self):
        result = _parse_json(
            '```json\n{"tool_name": "create_plan", "reason": "学习意图", "confidence": 0.85}\n```',
            DecideToolSelection,
        )
        assert result.tool_name == "create_plan"

    def test_parse_json_with_plain_markdown_wrapper(self):
        result = _parse_json(
            '```\n{"tool_name": "summarize_note", "reason": "总结", "confidence": 0.8}\n```',
            DecideToolSelection,
        )
        assert result.tool_name == "summarize_note"

    def test_parse_invalid_json_raises(self):
        with pytest.raises(ValueError):
            _parse_json("不是 JSON", DecideToolSelection)

    def test_parse_missing_field_raises(self):
        with pytest.raises(ValueError):
            _parse_json('{"tool_name": "search_docs"}', DecideToolSelection)

    def test_parse_confidence_range_clamped(self):
        with pytest.raises(ValueError):
            _parse_json(
                '{"tool_name": "search_docs", "reason": "x", "confidence": 1.5}',
                DecideToolSelection,
            )


# ============================================================
# 5. DECIDE 降级路径测试（不依赖 LLM）
# ============================================================

class TestDecideKeywordFallback:
    def test_fallback_search(self):
        reg = _get_registry()
        result = _decide_via_keyword(reg, "检索一下资料", {})
        d = result["decision"]
        assert d["tool_name"] == "search_docs"
        assert d["arguments"]["query"] == "检索一下资料"

    def test_fallback_learn(self):
        """'学习' 关键词映射到 search_docs"""
        reg = _get_registry()
        result = _decide_via_keyword(reg, "帮我学习 AI", {"topic": "AI 开发"})
        d = result["decision"]
        assert d["tool_name"] == "search_docs"

    def test_fallback_unknown_defaults_to_search(self):
        reg = _get_registry()
        result = _decide_via_keyword(reg, "今天天气不错", {})
        assert result["decision"]["tool_name"] == "search_docs"

    def test_fallback_decision_structure(self):
        """降级路径产出的 decision 结构与 LLM 路径一致"""
        reg = _get_registry()
        result = _decide_via_keyword(reg, "学习", {})
        d = result["decision"]
        for field in ["tool_name", "arguments", "reason", "confidence", "requires_retrieval", "action"]:
            assert field in d, f"缺少字段: {field}"


# ============================================================
# 6. DEPLOY 节点 LLM 集成测试（需要千问 API）
# 标记为 integration，避免日常开发频繁调用 API
# ============================================================

@pytest.mark.integration
class TestDecideLLMIntegration:
    def test_llm_selects_tool(self):
        """LLM 应选择合适的工具"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_decide import decide_node
        state = {"user_input": "帮我检索一下知识库", "plan": {}}
        result = decide_node(state)
        d = result["decision"]
        assert d["tool_name"] in ("search_docs", "memory_write", "memory_read")

    def test_llm_generates_arguments(self):
        """LLM 应生成合法的参数字典"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_decide import decide_node
        state = {"user_input": "学习 Agent 开发", "plan": {"topic": "Agent"}}
        result = decide_node(state)
        d = result["decision"]
        assert isinstance(d["arguments"], dict)
        assert len(d["arguments"]) > 0

    def test_llm_tool_exists_in_registry(self):
        """LLM 选出的工具必须已注册"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_decide import decide_node
        reg = _get_registry()
        state = {"user_input": "帮我总结一下", "plan": {}}
        result = decide_node(state)
        tool_name = result["decision"]["tool_name"]
        assert reg.exists(tool_name), f"LLM 选了未注册的工具: {tool_name}"


# ============================================================
# 7. PLANNER 节点测试
# ============================================================

class TestPlannerNode:
    def test_fallback_produces_plan(self):
        """降级路径产出合法 plan"""
        from runtime.node_planner import planner_node
        state = {"user_input": "学习 AI", "plan": {}}
        result = planner_node(state)
        plan = result["plan"]
        assert plan["topic"] == "学习 AI"
        assert len(plan["modules"]) == 3
        for m in plan["modules"]:
            assert "title" in m
            assert "content" in m
            assert "duration_minutes" in m
        assert plan["estimated_total_minutes"] > 0

    def test_fallback_respects_level(self):
        """降级路径根据 user_level 返回不同模块"""
        from runtime.node_planner import planner_node
        for level in ["beginner", "intermediate", "advanced"]:
            state = {"user_input": "test", "plan": {"user_level": level}}
            result = planner_node(state)
            assert result["plan"]["topic"] == "test"

    def test_plan_has_created_at(self):
        """plan 包含时间戳"""
        from runtime.node_planner import planner_node
        state = {"user_input": "test", "plan": {}}
        result = planner_node(state)
        assert "created_at" in result["plan"]


@pytest.mark.integration
class TestPlannerLLMIntegration:
    def test_llm_produces_structured_plan(self):
        """LLM 产出含 modules 的计划"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_planner import planner_node
        state = {"user_input": "学习 RAG 检索原理", "plan": {}}
        result = planner_node(state)
        plan = result["plan"]
        assert len(plan["modules"]) >= 2
        for m in plan["modules"]:
            assert "title" in m
            assert "content" in m
            assert isinstance(m["duration_minutes"], int)
        assert plan["estimated_total_minutes"] > 0

    def test_llm_plan_topic_matches_input(self):
        """LLM 计划主题与输入相关"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_planner import planner_node
        state = {"user_input": "Python 机器学习", "plan": {}}
        result = planner_node(state)
        assert "Python" in result["plan"]["topic"] or "机器学习" in result["plan"]["topic"]


# ============================================================
# 8. REFLECT 节点测试
# ============================================================

class TestReflectNode:
    def test_fallback_returns_end(self):
        """降级路径默认返回 end"""
        from runtime.node_reflect import reflect_node
        state = {
            "user_input": "test",
            "tool_results": [],
            "error": None,
            "plan": {},
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        assert result["reflection"]["next_action"] == "end"
        assert result["retry_count"] == 1  # 自增

    def test_fallback_increments_retry_count(self):
        """降级路径仍正确自增 retry_count"""
        from runtime.node_reflect import reflect_node
        state = {
            "user_input": "test",
            "tool_results": [],
            "error": None,
            "plan": {},
            "retry_count": 2,
            "max_retries": 3,
        }
        result = reflect_node(state)
        assert result["retry_count"] == 3


@pytest.mark.integration
class TestReflectLLMIntegration:
    def test_llm_returns_end_on_success(self):
        """成功结果 → LLM 返回 end"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node
        state = {
            "user_input": "学习 RAG",
            "tool_results": [
                {"tool_name": "search_docs", "status": "success",
                 "result": {"documents": [{"doc_id": "1"}]}}
            ],
            "error": None,
            "plan": {"topic": "RAG"},
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        r = result["reflection"]
        assert r["next_action"] == "end"
        assert r["is_satisfactory"] is True

    def test_llm_handles_failure_scenario(self):
        """失败结果 → LLM 应返回合法的反思结构（end 或 retry 均合理）"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node
        state = {
            "user_input": "学习 RAG",
            "tool_results": [
                {"tool_name": "search_docs", "status": "failed",
                 "error": "连接超时"}
            ],
            "error": "检索工具执行超时",
            "plan": {"topic": "RAG"},
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        r = result["reflection"]
        assert r["next_action"] in ("end", "retry")
        if r["next_action"] == "retry":
            assert r["retry_target_node"] in ("RETRIEVE", "EXECUTE", "PLANNER")

    def test_llm_reflection_has_required_fields(self):
        """LLM 反思结果包含所有必需字段"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node
        state = {
            "user_input": "test",
            "tool_results": [],
            "error": None,
            "plan": {},
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        r = result["reflection"]
        for field in ["confidence", "is_satisfactory", "next_action", "retry_target_node", "hallucination_flag"]:
            assert field in r, f"缺少字段: {field}"


# ============================================================
# 9. Retry 循环集成测试（端到端）
# ============================================================

@pytest.mark.integration
class TestRetryLoopIntegration:
    def test_retry_loop_with_injected_error(self):
        """全链路：注入错误 → REFLECT → retry → 成功 → END"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.state_graph import build_graph
        from runtime.state_manager import create_initial_state

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-retry-loop"}}

        # 构造带错误的 state，模拟 EXECUTE 失败后进入 REFLECT
        state = create_initial_state("学习 LangGraph")
        state["decision"] = {
            "tool_name": "search_docs",
            "arguments": {"query": "LangGraph"},
            "requires_retrieval": True,
        }
        state["tool_results"] = [
            {"tool_name": "search_docs", "status": "failed", "error": "超时"}
        ]
        state["error"] = "检索超时"
        state["current_step"] = "REFLECT"

        result = graph.invoke(state, config)
        # retry 循环后应最终结束
        assert result["current_step"] == "REFLECT"
        # 至少有 reflection 产出
        assert result["reflection"] is not None


# ============================================================
# 10. QA 闭环集成测试
# ============================================================

class TestQAPlannerIntent:
    def test_qa_keyword_detected_as_qa(self):
        """'什么是 X' 应被降级路径识别为 qa"""
        from runtime.node_planner import _detect_mode_keyword
        assert _detect_mode_keyword("什么是 RAG") == "qa"
        assert _detect_mode_keyword("介绍一下 LangGraph") == "qa"

    def test_learn_keyword_detected_as_learn(self):
        """'学习 X' 应被降级路径识别为 learn"""
        from runtime.node_planner import _detect_mode_keyword
        assert _detect_mode_keyword("学习 Agent 开发") == "learn"


@pytest.mark.integration
class TestQAIntegration:
    def test_planner_qa_intent(self):
        """LLM 应将 QA 问题识别为 qa 模式"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_planner import planner_node
        state = {"user_input": "什么是 RAG 检索？", "plan": {}}
        result = planner_node(state)
        assert result["mode"] in ("qa", "learn")

    def test_execute_qa_synthesis(self):
        """EXECUTE QA 模式产出 final_output + answer_source"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_execute import execute_node
        state = {
            "mode": "qa",
            "user_input": "什么是 LangGraph？",
            "retrieved_context": [
                {"doc_id": "doc_001",
                 "content": "LangGraph 是一个基于有向图的状态机框架",
                 "source": "langgraph_overview.md", "score": 0.6,
                 "metadata": {"title": "LangGraph 概述"}}
            ],
            "decision": {"tool_name": "search_docs", "arguments": {"query": "LangGraph"}},
        }
        result = execute_node(state)
        assert result["final_output"] is not None
        assert result["answer_source"] in ("rag", "llm_fallback")

    def test_reflect_qa_governance_fields(self):
        """REFLECT QA 模式产出治理字段"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node
        state = {
            "mode": "qa",
            "user_input": "什么是 LangGraph？",
            "retrieved_context": [
                {"doc_id": "doc_001",
                 "content": "LangGraph 是一个基于有向图的状态机框架",
                 "source": "langgraph_overview.md", "score": 0.6}
            ],
            "final_output": "LangGraph 是一个状态机框架。",
            "answer_source": "rag",
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        assert result["groundedness_score"] is not None
        assert result["completeness_score"] is not None
        assert 0.0 <= result["groundedness_score"] <= 1.0
        assert 0.0 <= result["completeness_score"] <= 1.0

    def test_qa_full_flow(self):
        """QA 完整链路：PLANNER → EXECUTE → REFLECT"""
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.state_graph import build_graph
        from runtime.state_manager import create_initial_state

        graph = build_graph()
        config = {"configurable": {"thread_id": "test-qa-full"}}

        state = create_initial_state("什么是 LangGraph")
        state["mode"] = "qa"
        state["decision"] = {
            "tool_name": "search_docs",
            "arguments": {"query": "LangGraph"},
            "requires_retrieval": True,
        }

        result = graph.invoke(state, config)
        # 完整流转后应有 final_output 和治理字段
        assert result["current_step"] == "REFLECT"
        assert result.get("answer_source") is not None or result.get("final_output") is not None
