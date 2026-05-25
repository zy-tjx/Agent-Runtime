"""
API 路由
单端点 POST /chat —— 纯透传，零业务逻辑
"""
from pydantic import BaseModel, Field
from runtime.state_graph import run_graph
from reflection.eval_metrics import compute_metrics
from reflection.hallucination_detector import detect as detect_hallucination
from observability.langsmith_tracer import get_tracer


class ChatRequest(BaseModel):
    user_input: str = Field(description="用户输入的问题")
    session_id: str | None = Field(default=None, description="会话标识，不传则自动生成")


def handle_chat(req: ChatRequest) -> dict:
    """处理聊天请求，返回完整 AgentState + 指标"""
    result = run_graph(req.user_input, session_id=req.session_id)
    metrics = compute_metrics(result)
    hallucination = detect_hallucination(result)
    trace = get_tracer().summary()

    return {
        "agent_state": {
            "mode": result.get("mode"),
            "current_step": result.get("current_step"),
            "final_output": result.get("final_output"),
            "retrieval_score": result.get("retrieval_score"),
            "groundedness_score": result.get("groundedness_score"),
            "completeness_score": result.get("completeness_score"),
            "answer_source": result.get("answer_source"),
            "retry_count": result.get("retry_count"),
            "error": result.get("error"),
            "fallback_triggered": result.get("fallback_triggered", False),
            "fallback_reason": result.get("fallback_reason"),
        },
        "metrics": metrics,
        "hallucination": hallucination,
        "trace": trace,
    }
