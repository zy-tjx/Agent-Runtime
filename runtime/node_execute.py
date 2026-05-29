"""
EXECUTE 节点
learn 模式：执行工具调用，产出 tool_calls / tool_results
qa 模式：基于 retrieved_context 做受约束答案合成，产出 final_output / answer_source
"""
"""
execute_node (入口)
  └── _execute_tool (Learn 模式主逻辑)
        ├── _get_executor (获取工具执行器单例)
        └── executor.execute (实际调用底层工具，如 RAG 检索)
execute_node (入口)
  └── _execute_qa (QA 模式主逻辑)
        ├── _format_context (格式化检索到的文档上下文)
        │
        ├── [主路径] _synthesize_via_llm (LLM 约束总结答案)
        │       ├── render (生成提示词)
        │       ├── model.generate (调用大模型)
        │       └── _parse_json (解析并校验 LLM 输出的 JSON)
        │
        └── [降级路径] _execute_qa_fallback (LLM 失败时的兜底)
                └── _build_qa_result (打包最终结果)
"""
import time
from typing import Any
from pydantic import BaseModel

from tools.tool_registry import create_default_registry
from tools.tool_executor import ToolExecutor
from engine.prompt_manager import render
from engine.model_manager import ModelManager
from runtime.node_decide import _parse_json
from observability.logger import get_logger


# ── 模块级单例 ──

_executor: ToolExecutor | None = None
_model_manager: ModelManager | None = None


def _get_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        registry = create_default_registry()
        _executor = ToolExecutor(registry)
    return _executor


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

# ── 辅助 ──

def _format_context(documents: list[dict]) -> str:
    """将检索结果格式化为 Prompt 可读文本"""
    if not documents:
        return "无相关文档"
    lines = []
    for i, doc in enumerate(documents, 1):
        lines.append(
            f"[doc_{i}] {doc.get('source', 'unknown')}\n"
            f"{doc.get('content', '')}"
        )
    return "\n\n".join(lines)

# ── QA 模式：受约束答案合成 ──

class QASynthesisOutput(BaseModel):
    """QA 答案合成的输出结构"""
    answer: str
    answer_source: str = "rag"
    sources: list[int] = []

def _build_qa_result(
    final_output: str,
    answer_source: str,
    via: str,
) -> dict[str, Any]:
    """组装 QA 模式返回结构"""
    return {
        "current_step": "EXECUTE",
        "final_output": final_output,
        "answer_source": answer_source,
        "tool_calls": [],
        "tool_results": [],
        "messages": [
            {"role": "ai", "content": f"已生成回答（{via}，来源={answer_source}）"}
        ],
    }

def _synthesize_via_llm(
    user_input: str, context_text: str
) -> dict[str, Any]:
    """LLM 受约束答案合成"""
    model = _get_model_manager()
    prompt = render(
        "execute_qa_synthesis",
        user_input=user_input,
        retrieved_context=context_text,
    )
    result_text = model.generate(prompt)
    qa_result = _parse_json(result_text, QASynthesisOutput)

    return _build_qa_result(
        final_output=qa_result.answer,
        answer_source=qa_result.answer_source,
        via="LLM",
    )


def _execute_qa_fallback(
    user_input: str, retrieved_context: list
) -> dict[str, Any]:
    """LLM 失败时：取第一条文档内容包装为保守回答"""
    if retrieved_context:
        raw = retrieved_context[0].get("content", "无相关信息")
        answer = f"模型生成失败，以下为检索到的最相关资料：\n\n{raw}"
    else:
        answer = "未检索到相关文档，且模型生成不可用，请稍后重试"

    return _build_qa_result(
        final_output=answer,
        answer_source="rag" if retrieved_context else "llm_fallback",
        via="模板降级",
    )

def _execute_qa(state: dict[str, Any]) -> dict[str, Any]:
    """基于 retrieved_context 做答案合成"""
    user_input = state.get("user_input", "")
    retrieved_context = state.get("retrieved_context", [])

    # ── 格式化检索上下文 ──
    context_text = _format_context(retrieved_context)

    import os
    if os.getenv("DECIDE_USE_KEYWORD", "") == "1":
        return _execute_qa_fallback(user_input, retrieved_context)

    try:
        return _synthesize_via_llm(user_input, context_text)
    except Exception:
        return _execute_qa_fallback(user_input, retrieved_context)
    
# ── Learn 模式：工具执行 ──

def _execute_tool(state: dict[str, Any]) -> dict[str, Any]:
    """执行工具调用"""
    decision = state.get("decision", {})
    tool_name = decision.get("tool_name", "unknown_tool")
    arguments = decision.get("arguments", {})

    called_at = time.strftime("%Y-%m-%dT%H:%M:%S")

    executor = _get_executor()
    output = executor.execute(tool_name, arguments)

    tool_call = {
        "tool_name": tool_name,
        "arguments": arguments,
        "called_at": called_at,
    }

    tool_result = {
        "tool_name": output.tool_name,
        "status": output.status,
        "result": output.result,
        "error": output.error,
        "duration_ms": output.duration_ms,
    }

    error = output.error if output.status != "success" else None

    return {
        "current_step": "EXECUTE",
        "tool_calls": [tool_call],
        "tool_results": [tool_result],
        "error": error,
        "messages": [
            {
                "role": "ai",
                "content": (
                    f"工具 {tool_name} 执行{'成功' if output.status == 'success' else '失败'}"
                ),
            }
        ],
    }

# ── 节点主函数 ──

def execute_node(state: dict[str, Any]) -> dict[str, Any]:
    """执行工具调用或 QA 答案合成"""
    logger = get_logger()
    logger.node_start("EXECUTE", state)

    try:
        mode = state.get("mode", "learn")

        if mode == "qa":
            result = _execute_qa(state)
        else:
            result = _execute_tool(state)

        logger.node_end("EXECUTE", {**state, **result})
        return result
    except Exception as e:
        logger.node_error("EXECUTE", str(e))
        raise




