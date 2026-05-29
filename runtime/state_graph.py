"""
StateGraph 组装与编译
负责创建状态图、添加节点、配置边、编译运行
"""
"""
build_graph (总装车间)
  ├── StateGraph(AgentState) (初始化一张空白状态图)
  ├── graph.add_node (注册 5 个核心节点，并用 trace_node 包裹以支持链路追踪)
  ├── graph.add_edge (添加确定性边：START→planner, execute→reflect)
  ├── graph.add_conditional_edges (添加条件分支边：planner/decide/retrieve/reflect 后的路由判断)
  └── graph.compile (编译图，注入 MemorySaver 内存检查点，支持中断恢复)
run_graph (对外发令枪)
  ├── create_initial_state (结合用户输入与对话历史，生成初始状态)
  ├── get_user_profile (从长期记忆中加载用户画像)
  └── graph.invoke (启动编译好的图，传入初始状态与线程配置，开始流转)
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from runtime.state_manager import AgentState, create_initial_state
from runtime.edge_router import (
    router_after_planner,
    router_after_decide,
    router_after_retrieve,
    router_after_reflect,
    Route,
)
from runtime.node_planner import planner_node
from runtime.node_decide import decide_node
from runtime.node_retrieve import retrieve_node
from runtime.node_execute import execute_node
from runtime.node_reflect import reflect_node
from observability.tracer import trace_node, reset_tracer


def build_graph() -> StateGraph:
    """
    构建 Agent Runtime 状态图

    节点（5 个）：planner → decide → retrieve → execute → reflect
    简单边（2 条）：START→planner, execute→reflect
    条件边（4 条）：planner→?, decide→?, retrieve→?, reflect→?
    """
    graph = StateGraph(AgentState)

    # ── 注册节点（含 trace 包装） ──
    graph.add_node("planner", trace_node("planner", planner_node))
    graph.add_node("decide", trace_node("decide", decide_node))
    graph.add_node("retrieve", trace_node("retrieve", retrieve_node))
    graph.add_node("execute", trace_node("execute", execute_node))
    graph.add_node("reflect", trace_node("reflect", reflect_node))

    # ── 简单边 ──
    graph.add_edge(START, "planner")

    graph.add_edge("execute", "reflect")  

    # ── 条件边（分支流转） ──

    # PLANNER → DECIDE（反问水平时直接 END）
    graph.add_conditional_edges(
        "planner",
        router_after_planner,
        {Route.DECIDE: "decide", Route.END: END},
    )

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
    user_id: str = "default",
) -> dict:
    """
    便捷入口：传入用户问题，运行完整状态机，返回最终 State

    Args:
        user_input: 用户输入的问题
        session_id: 会话标识（可选，默认自动生成）
        history: 最近 N 轮对话历史 [{"role":"human/ai","content":"..."}]
        user_id: 用户标识，用于长期记忆画像读写
    """
    import time
    from memory.long_term_memory import get_user_profile

    if session_id is None:
        session_id = f"demo-session-{int(time.time() * 1000)}"

    reset_tracer()
    graph = build_graph()
    initial_state = create_initial_state(user_input, history=history)
    initial_state["short_term_memory"]["session_id"] = session_id
    # 加载用户画像到长期记忆字段
    initial_state["long_term_memory"] = {
        "user_id": user_id,
        "profile": get_user_profile(user_id),
    }

    # thread_id 用于 checkpointer 区分不同会话
    config = {"configurable": {"thread_id": session_id}}

    final_state = graph.invoke(initial_state, config)
    return final_state


