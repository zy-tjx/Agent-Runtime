"""
工具链集成测试
覆盖：Registry / Tool Schema / ToolExecutor / DECIDE / EXECUTE / 端到端

注意：DECIDE 相关测试使用 DECIDE_USE_KEYWORD=1 跳过 LLM，加速执行。
      LLM 路径的集成测试见 tests/test_model_manager.py（需 Ollama 运行）。
"""
import os
os.environ["DECIDE_USE_KEYWORD"] = "1"

import pytest
from pydantic import ValidationError

from tools.tool_registry import ToolRegistry, create_default_registry
from tools.tool_executor import ToolExecutor
from tools.search_docs import SearchDocsInput, run as search_run
from tools.memory_write import MemoryWriteInput, run as mw_run
from tools.memory_read import MemoryReadInput, run as mr_run
from tools.context_store import ContextStoreInput, run as cs_run
from tools.fallback import FallbackInput, run as fb_run
from runtime.node_decide import decide_node
from runtime.node_execute import execute_node


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def registry():
    r = ToolRegistry()
    r.register("search_docs", "检索知识库", SearchDocsInput, search_run)
    r.register("memory_write", "写入长期记忆", MemoryWriteInput, mw_run)
    r.register("memory_read", "读取长期记忆", MemoryReadInput, mr_run)
    r.register("context_store", "会话状态持久化", ContextStoreInput, cs_run)
    r.register("fallback", "统一兜底", FallbackInput, fb_run)
    return r


@pytest.fixture
def executor(registry):
    return ToolExecutor(registry)


# ============================================================
# 1. ToolRegistry 测试
# ============================================================

class TestToolRegistry:
    def test_register_five_tools(self, registry):
        """注册 5 个工具后列表返回 5 个"""
        assert len(registry) == 5
        assert len(registry.list_tools()) == 5

    def test_exists(self, registry):
        """exists 正确判断"""
        assert registry.exists("search_docs") is True
        assert registry.exists("no_such_tool") is False

    def test_get(self, registry):
        """get 返回 ToolInfo 含正确的元信息"""
        tool = registry.get("memory_write")
        assert tool is not None
        assert tool.name == "memory_write"
        assert tool.input_schema == MemoryWriteInput
        assert tool.run_func == mw_run

    def test_get_missing(self, registry):
        """获取不存在的工具返回 None"""
        assert registry.get("unknown") is None

    def test_duplicate_register_raises(self, registry):
        """重复注册抛出 ValueError"""
        with pytest.raises(ValueError, match="已存在"):
            registry.register("search_docs", "重复", SearchDocsInput, search_run)

    def test_list_tools_has_parameters(self, registry):
        """list_tools 返回的参数元信息可被 DECIDE 使用"""
        tools = registry.list_tools()
        search = next(t for t in tools if t["name"] == "search_docs")
        assert "query" in search["parameters"]
        assert search["parameters"]["query"]["required"] is True
        assert "top_k" in search["parameters"]
        assert search["parameters"]["top_k"]["required"] is False

    def test_create_default_registry(self):
        """工厂函数返回预注册 5 个工具的 registry"""
        reg = create_default_registry()
        assert len(reg) == 5
        for name in ["search_docs", "memory_write", "memory_read", "context_store", "fallback"]:
            assert reg.exists(name)


# ============================================================
# 2. 工具 run() 测试
# ============================================================

class TestSearchDocs:
    def test_keyword_match(self):
        """关键词匹配返回相关文档"""
        out = search_run(SearchDocsInput(query="RAG 检索"))
        assert out.status == "success"
        assert len(out.result["documents"]) >= 1
        assert out.result["documents"][0]["doc_id"] == "doc_002"  # RAG 相关

    def test_no_match_fallback(self):
        """无匹配时返回兜底文档"""
        out = search_run(SearchDocsInput(query="xyzzyzzy"))
        assert out.status == "success"
        assert len(out.result["documents"]) == 1
        assert out.result["documents"][0]["score"] == 0.60  # 兜底分数

    def test_top_k_limit(self):
        """top_k 限制返回数量"""
        out = search_run(SearchDocsInput(query="Agent", top_k=2))
        assert len(out.result["documents"]) <= 2

    def test_output_aligns_with_retrieved_context(self):
        """输出字段对齐 AgentState.retrieved_context 结构"""
        out = search_run(SearchDocsInput(query="LangGraph"))
        doc = out.result["documents"][0]
        for field in ["doc_id", "content", "source", "score", "metadata"]:
            assert field in doc


# 旧业务工具（create_plan / update_progress / summarize_note / generate_quiz）已移除。
# 新 Runtime 工具（memory_write / memory_read / context_store / fallback）在 Memory Phase 实现后测试。


# ============================================================
# 3. ToolExecutor 测试
# ============================================================

class TestToolExecutor:
    def test_normal_execution(self, executor):
        out = executor.execute("search_docs", {"query": "LangGraph"})
        assert out.status == "success"
        assert out.tool_name == "search_docs"
        assert out.error is None

    def test_tool_not_found(self, executor):
        out = executor.execute("no_such_tool", {})
        assert out.status == "failed"
        assert "找不到" in out.error

    def test_validation_error(self, executor):
        """缺少必填参数时返回 failed"""
        out = executor.execute("search_docs", {})
        assert out.status == "failed"
        assert "参数校验失败" in out.error

    def test_validation_with_illegal_value(self, executor):
        """参数超限时返回 failed"""
        out = executor.execute("search_docs", {"query": ""})  # 空 query → 校验失败
        assert out.status == "failed"

    def test_exception_caught(self):
        """工具内部异常被兜底为 failed"""
        def bad_run(input_data):
            raise RuntimeError("模拟崩溃")
        r = ToolRegistry()
        r.register("bad_tool", "会失败", SearchDocsInput, bad_run)
        exc = ToolExecutor(r)
        out = exc.execute("bad_tool", {"query": "test"})
        assert out.status == "failed"
        assert "工具执行异常" in out.error
        assert "模拟崩溃" in out.error

    def test_duration_recorded(self, executor):
        out = executor.execute("search_docs", {"query": "test"})
        assert out.duration_ms >= 0


# ============================================================
# 4. DECIDE 节点测试（关键词匹配 + arguments）
# ============================================================

class TestDecideNode:
    def test_search_keyword(self):
        state = {"user_input": "帮我检索一下 LangGraph", "plan": {}}
        result = decide_node(state)
        assert result["decision"]["tool_name"] == "search_docs"
        assert result["decision"]["arguments"]["query"] == "帮我检索一下 LangGraph"
        assert result["decision"]["requires_retrieval"] is True

    def test_learn_keyword_maps_to_search(self):
        """'学习' 关键词现在映射到 search_docs（学习内容通过检索获取）"""
        state = {"user_input": "我想学习 Agent 开发", "plan": {"topic": "Agent 开发"}}
        result = decide_node(state)
        assert result["decision"]["tool_name"] == "search_docs"

    def test_summary_keyword_maps_to_search(self):
        """'总结' 关键词映射到 search_docs"""
        state = {"user_input": "帮我总结一下", "plan": {}}
        result = decide_node(state)
        assert result["decision"]["tool_name"] == "search_docs"

    def test_no_match_defaults_to_search(self):
        state = {"user_input": "今天天气真好", "plan": {}}
        result = decide_node(state)
        assert result["decision"]["tool_name"] == "search_docs"

    def test_decision_structure(self):
        """DECIDE 产出的 decision 结构完整"""
        state = {"user_input": "学习 AI", "plan": {}}
        result = decide_node(state)
        d = result["decision"]
        for field in ["tool_name", "arguments", "reason", "confidence", "requires_retrieval", "action"]:
            assert field in d, f"缺少字段: {field}"


# ============================================================
# 5. EXECUTE 节点测试
# ============================================================

class TestExecuteNode:
    def test_executes_with_decision(self):
        """EXECUTE 读取 decision，调用真实工具"""
        state = {
            "user_input": "LangGraph 是什么",
            "decision": {
                "tool_name": "search_docs",
                "arguments": {"query": "LangGraph"},
            },
        }
        result = execute_node(state)
        assert result["current_step"] == "EXECUTE"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["tool_name"] == "search_docs"
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["status"] == "success"
        assert result["tool_results"][0]["result"] is not None

    def test_tool_error_propagates(self):
        """工具执行失败时 error 字段传播到 state"""
        state = {
            "user_input": "test",
            "decision": {
                "tool_name": "search_docs",
                "arguments": {},  # 缺少必填 query → 校验失败
            },
        }
        result = execute_node(state)
        assert result["tool_results"][0]["status"] == "failed"
        assert result["error"] is not None

    def test_unknown_tool(self):
        """调用未注册工具时失败"""
        state = {
            "user_input": "test",
            "decision": {
                "tool_name": "ghost_tool",
                "arguments": {},
            },
        }
        result = execute_node(state)
        assert result["tool_results"][0]["status"] == "failed"


# ============================================================
# 6. DECIDE → EXECUTE 端到端（集成测试）
# ============================================================

class TestDecideToExecute:
    def test_decide_then_execute_search(self):
        """DECIDE 产出 → 直接喂给 EXECUTE → 真实工具执行"""
        user_input = "检索 LangGraph 状态机"
        state = {"user_input": user_input, "plan": {}}
        decide_result = decide_node(state)

        # 合并 DECIDE 产出到 state
        state.update(decide_result)
        exec_result = execute_node(state)

        assert exec_result["tool_results"][0]["status"] == "success"
        assert "documents" in exec_result["tool_results"][0]["result"]

    def test_decide_selects_registered_tools_only(self):
        """DECIDE 选出的工具名必须在 registry 中存在"""
        registry = create_default_registry()
        for user_input in ["检索", "学习", "总结", "测验"]:
            state = {"user_input": user_input, "plan": {}}
            result = decide_node(state)
            tool_name = result["decision"]["tool_name"]
            assert registry.exists(tool_name), f"DECIDE 选了未注册的工具: {tool_name}"
