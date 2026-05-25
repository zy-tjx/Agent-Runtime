"""
StateGraph 组装与编译
负责创建状态图、添加节点、配置边、编译运行
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from runtime.state_manager import AgentState, create_initial_state
from runtime.edge_router import (
    router_after_decide,
    router_after_retrieve,
    router_after_execute,
    router_after_reflect,
    Route,
)
from runtime.node_planner import planner_node
from runtime.node_decide import decide_node
from runtime.node_retrieve import retrieve_node
from runtime.node_execute import execute_node
from runtime.node_reflect import reflect_node
from observability.langsmith_tracer import trace_node, reset_tracer


def build_graph() -> StateGraph:
    """
    构建 Agent Runtime 状态图

    节点（5 个）：planner → decide → retrieve → execute → reflect
    简单边（3 条）：START→planner, planner→decide, execute→reflect
    条件边（3 条）：decide→?  retrieve→?  reflect→?
    """
    graph = StateGraph(AgentState)

    # ── 注册节点（含 trace 包装） ──
    graph.add_node("planner", trace_node("planner", planner_node))
    graph.add_node("decide", trace_node("decide", decide_node))
    graph.add_node("retrieve", trace_node("retrieve", retrieve_node))
    graph.add_node("execute", trace_node("execute", execute_node))
    graph.add_node("reflect", trace_node("reflect", reflect_node))

    # ── 简单边（确定性流转） ──
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "decide")
    graph.add_edge("execute", "reflect")  # v3: 无论成败一律进反思

    # ── 条件边（分支流转） ──
    graph.add_conditional_edges(
        "decide",
        router_after_decide,
        {
            Route.RETRIEVE: "retrieve",
            Route.EXECUTE: "execute",
            Route.REFLECT: "reflect",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        router_after_retrieve,
        {
            Route.EXECUTE: "execute",
            Route.REFLECT: "reflect",
        },
    )

    graph.add_conditional_edges(
        "reflect",
        router_after_reflect,
        {
            Route.PLANNER: "planner",
            Route.RETRIEVE: "retrieve",
            Route.EXECUTE: "execute",
            Route.END: END,
        },
    )

    # ── 编译（使用内存 checkpointer 支持中断恢复） ──
    compiled = graph.compile(checkpointer=MemorySaver())
    return compiled


def run_graph(
    user_input: str,
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> dict:
    """
    便捷入口：传入用户问题，运行完整状态机，返回最终 State

    Args:
        user_input: 用户输入的问题
        session_id: 会话标识（可选，默认自动生成）
        history: 最近 N 轮对话历史 [{"role":"human/ai","content":"..."}]
    """
    import time
    if session_id is None:
        session_id = f"demo-session-{int(time.time() * 1000)}"

    reset_tracer()  # 每次运行前重置 tracer，避免跨会话污染
    graph = build_graph()
    initial_state = create_initial_state(user_input, history=history)
    # session_id 可能在 create_initial_state 中被覆盖，这里显式设置
    initial_state["short_term_memory"]["session_id"] = session_id

    # thread_id 用于 checkpointer 区分不同会话
    config = {"configurable": {"thread_id": session_id}}

    final_state = graph.invoke(initial_state, config)
    return final_state


# ============================================================
# 自测入口
# ============================================================
if __name__ == "__main__":
    result = run_graph("学习 LangGraph 状态机原理")
    print("=== 状态流转完成 ===")
    print(f"最终步骤: {result.get('current_step')}")
    print(f"final_output: {result.get('final_output')}")
    print(f"retry_count: {result.get('retry_count')}")
    print(f"messages 数量: {len(result.get('messages', []))}")
    for m in result.get("messages", []):
        role = getattr(m, "type", getattr(m, "role", "unknown"))
        content = getattr(m, "content", str(m))
        print(f"  [{role}] {content}")
