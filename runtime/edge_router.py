"""
条件路由函数
每个函数接收 AgentState，返回下一个节点名称（str）

路由全景（v3 — 所有错误立即进入 REFLECT，节点不做原地重试）：

    START ──▶ PLANNER ──▶ DECIDE ──▶ RETRIEVE ──▶ EXECUTE ──▶ REFLECT ──▶ END
                   ▲           │          │            │            │
                   │           │          │            │            │
                   └───────────┴──────────┴────────────┘            │
                             retry (PLANNER / RETRIEVE / EXECUTE)    │
                             由 REFLECT 统一决策                      │
                                                                      │
    retry_count = REFLECT 执行次数，达到 max_retries 后强制 END ←─────┘
"""
from runtime.state_manager import AgentState


# ============================================================
# 路由常量
# ============================================================

class Route:
    """所有合法的路由目标"""
    PLANNER = "PLANNER"
    DECIDE = "DECIDE"
    RETRIEVE = "RETRIEVE"
    EXECUTE = "EXECUTE"
    REFLECT = "REFLECT"
    END = "END"


# ============================================================
# 通用辅助
# ============================================================

def _can_retry(state: "AgentState") -> bool:
    """
    重试配额是否还有剩余
    retry_count 即为 REFLECT 节点的执行次数（由 REFLECT 节点自增）
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)
    return retry_count < max_retries


def _has_error(state: "AgentState") -> bool:
    """当前是否存在未处理的错误"""
    return state.get("error") is not None and state["error"] != ""


# ============================================================
# DECIDE → RETRIEVE | EXECUTE | REFLECT
# ============================================================

def router_after_decide(state: "AgentState") -> str:
    """
    DECIDE 节点之后的条件路由

    路由逻辑：
    1. 有错误 → REFLECT（记录原因，由反思决定是否重试）
    2. 无 decision → REFLECT（异常，进反思分析）
    3. requires_retrieval == True → RETRIEVE
    4. requires_retrieval == False → EXECUTE
    """
    if _has_error(state):
        return Route.REFLECT

    decision = state.get("decision")
    if decision is None:
        return Route.REFLECT

    if decision.get("requires_retrieval", False):
        return Route.RETRIEVE
    else:
        return Route.EXECUTE


# ============================================================
# RETRIEVE → EXECUTE | REFLECT
# ============================================================

def router_after_retrieve(state: "AgentState") -> str:
    """
    RETRIEVE 节点之后的条件路由

    路由逻辑（节点不做原地重试）：
    1. 检索成功 → EXECUTE
    2. 检索失败 → REFLECT（记录错误根因，由反思统一决定重试目标）
    """
    if _has_error(state):
        return Route.REFLECT
    return Route.EXECUTE


# ============================================================
# EXECUTE → REFLECT
# ============================================================

def router_after_execute(_state: "AgentState") -> str:
    """
    EXECUTE 节点之后的条件路由

    路由逻辑（节点不做原地重试）：
    无论成功还是失败，一律进入 REFLECT。
    成功时：REFLECT 做满意度评估
    失败时：REFLECT 记录错误根因，决定是否重试
    """
    return Route.REFLECT


# ============================================================
# REFLECT → END | PLANNER | RETRIEVE | EXECUTE
# ============================================================

def router_after_reflect(state: "AgentState") -> str:
    """
    REFLECT 节点之后的条件路由 —— 整个状态机唯一的重试决策点

    路由优先级：
    1. recovery_action == "guardrail" → END（安全兜底）
    2. recovery_action == "escalate" → END（无法自动恢复）
    3. reflection 缺失 → END（异常保护）
    4. reflection.next_action == "end" → END（满意或配额用完）
    5. reflection.next_action == "retry" + 有配额 → retry_target_node
       - retry_target_node == "PLANNER" → PLANNER
       - retry_target_node == "RETRIEVE" → RETRIEVE
       - retry_target_node == "EXECUTE" → EXECUTE
    6. reflection.next_action == "retry" + 无配额 → END（强制终止）
    """
    recovery = state.get("recovery_action")

    # ── 不可恢复的终止 ──
    if recovery in ("guardrail", "escalate"):
        return Route.END

    # ── 反思决策 ──
    reflection = state.get("reflection")
    if reflection is None:
        return Route.END

    next_action = reflection.get("next_action", "end")

    if next_action == "end":
        return Route.END

    if next_action == "retry":
        if not _can_retry(state):
            # 配额用完，即使反思建议重试也强制终止
            return Route.END
        target = reflection.get("retry_target_node", Route.EXECUTE)
        if target in (Route.PLANNER, Route.RETRIEVE, Route.EXECUTE):
            return target
        return Route.EXECUTE

    # 未知 next_action，安全终止
    return Route.END


# ============================================================
# 错误恢复路由（供 recovery_handler 调用）
# ============================================================

def router_for_recovery(state: "AgentState") -> str:
    """
    异常恢复时的路由决策（由 recovery_handler 调用）

    根据 recovery_action 决定：
    - retry：返回当前节点（重新执行）
    - fallback：跳到 REFLECT（走降级逻辑）
    - guardrail / escalate：走到 END（安全终止）
    """
    recovery = state.get("recovery_action")

    if recovery == "retry":
        current = state.get("current_step", Route.REFLECT)
        return current

    if recovery == "fallback":
        return Route.REFLECT

    # guardrail / escalate / 未知
    return Route.END
