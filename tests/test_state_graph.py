"""
状态机流转测试
覆盖：正常流转 / 错误路由 / 重试循环 / 配额耗尽
"""
import pytest
from runtime.state_graph import build_graph
from runtime.state_manager import create_initial_state


@pytest.fixture
def graph():
    return build_graph()


def _invoke(graph, initial_state, thread_id="test"):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(initial_state, config)


# ============================================================
# 正常流转
# ============================================================

def test_normal_flow_happy_path(graph):
    """正常输入 → 完整流转 Start→End（LLM 决策可能触发 retry）"""
    state = create_initial_state("学习 RAG 检索原理")
    result = _invoke(graph, state)
    assert result["current_step"] == "REFLECT"
    assert result["retry_count"] >= 1
    assert result["reflection"]["next_action"] in ("end", "retry")
    assert result["final_output"] is not None
    assert len(result["messages"]) >= 5  # human + planner + decide + retrieve + execute + reflect


def test_normal_flow_produces_plan(graph):
    """PLANNER 节点产出 plan 字段"""
    state = create_initial_state("学习 Prompt Engineering")
    result = _invoke(graph, state)
    assert result["plan"] is not None
    assert result["plan"]["topic"] is not None
    assert len(result["plan"]["modules"]) >= 1


def test_normal_flow_produces_decision(graph):
    """DECIDE 节点产出 decision 字段"""
    state = create_initial_state("任意问题")
    result = _invoke(graph, state)
    assert result["decision"] is not None
    assert result["decision"]["action"] is not None
    assert isinstance(result["decision"]["requires_retrieval"], bool)


# ============================================================
# 错误路由：任何节点错误 → 进 REFLECT
# ============================================================

def test_error_goes_to_reflect_from_decide(graph):
    """DECIDE 发现错误 → 直接进 REFLECT，跳过检索和工具执行"""
    state = create_initial_state("测试错误")
    state["error"] = "决策模型超时"
    result = _invoke(graph, state)
    assert result["current_step"] == "REFLECT"
    ai_messages = [m for m in result["messages"] if getattr(m, "type", "") == "ai"]
    assert len(ai_messages) >= 1  # 至少有 planner + reflect


# ============================================================
# 重试循环
# ============================================================

def test_retry_loop_replan(graph):
    """REFLECT 返回 retry+PLANNER → 回到 PLANNER 重新规划"""
    state = create_initial_state("需要重试的问题")
    state["error"] = "计划不充分"
    # 预设 reflection（模拟真实 REFLECT 可能的输出）
    state["reflection"] = {
        "confidence": 0.3,
        "is_satisfactory": False,
        "next_action": "retry",
        "retry_target_node": "PLANNER",
    }
    result = _invoke(graph, state)
    # 重试后最终会进 REFLECT 再次评估
    assert result["current_step"] == "REFLECT"
    assert result["reflection"]["next_action"] in ("end", "retry")


def test_retry_loop_retrieve(graph):
    """REFLECT 返回 retry+RETRIEVE → 重新检索"""
    state = create_initial_state("检索结果不佳")
    state["decision"] = {"requires_retrieval": True}
    state["error"] = "检索质量低"
    state["reflection"] = {
        "confidence": 0.4,
        "is_satisfactory": False,
        "next_action": "retry",
        "retry_target_node": "RETRIEVE",
    }
    result = _invoke(graph, state)
    assert result["current_step"] == "REFLECT"


# ============================================================
# 配额耗尽
# ============================================================

def test_max_retries_exceeded_ends(graph):
    """retry_count 达到上限 → 强制终止"""
    state = create_initial_state("超限问题")
    state["retry_count"] = 3  # == max_retries
    state["error"] = "持续失败"
    state["reflection"] = {
        "confidence": 0.1,
        "is_satisfactory": False,
        "next_action": "retry",
        "retry_target_node": "EXECUTE",
    }
    result = _invoke(graph, state)
    # retry_count >= max_retries → 即使 reflection 建议 retry 也终止
    assert result["retry_count"] >= 3


# ============================================================
# State 字段完整性
# ============================================================

def test_all_state_fields_present(graph):
    """完整流转后所有 State 字段均存在"""
    state = create_initial_state("完整性测试")
    result = _invoke(graph, state)
    expected_fields = [
        "messages", "current_step", "user_input", "final_output",
        "error", "retry_count", "max_retries", "recovery_action",
        "plan", "decision",
        "rewritten_query", "retrieved_context",
        "vector_search_results", "reranked_results", "context_assembly_info",
        "tool_calls", "tool_results",
        "reflection",
        "short_term_memory", "long_term_memory",
        "rag_metadata",
    ]
    for field in expected_fields:
        assert field in result, f"State 缺少字段: {field}"
