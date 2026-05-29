"""
Agent Runtime 治理驾驶舱
Streamlit 单页应用：输入问题 → 查看回答 + 治理指标一览
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import time as _time
# 确保 FAISS 索引已加载
import rag.vector_retriever as _vrm
_vrm._retriever = None
_vrm.get_retriever()

# 延迟导入，避免 Streamlit 热重载缓存旧版本模块
def _get_run_graph():
    from runtime.state_graph import run_graph as _rg
    return _rg

def _get_compute_metrics():
    from reflection.eval_metrics import compute_metrics as _cm
    return _cm

def _get_detect_hallucination():
    from reflection.hallucination_detector import detect as _dh
    return _dh

def _get_tracer():
    from observability.tracer import get_tracer as _gt
    return _gt()

def _get_experience_summaries(limit, mode):
    from memory.long_term_memory import load_experience_summaries as _les
    return _les(limit=limit, mode=mode)

# ── 页面配置 ──
st.set_page_config(
    page_title="Agent Runtime 治理驾驶舱",
    page_icon="A",
    layout="wide",
)

st.title("Agent Runtime 治理驾驶舱")


# ============================================================
# 组件函数
# ============================================================

def render_flow_timeline(nodes_visited: list[str], tracer_records: list[dict]) -> None:
    """节点流转时序：横向展示每个节点名称和耗时"""
    st.subheader("节点流转时序")
    if not nodes_visited:
        st.caption("无 trace 数据")
        return

    cols = st.columns(len(nodes_visited))
    for i, node in enumerate(nodes_visited):
        with cols[i]:
            duration = 0
            for r in tracer_records:
                if r["node"] == node:
                    duration = r["duration_ms"]
                    break
            st.metric(label=node.upper(), value=f"{duration}ms")


def render_governance_dashboard(
    result: dict, metrics: dict, hallucination: dict
) -> None:
    """治理仪表：模式 / 检索 / 接地 / 完整 / 流程 / 降级 / 幻觉 / 置信度"""
    st.subheader("治理仪表")

    st.metric("模式", result.get("mode", "?"))

    rag = metrics["rag"]
    st.metric("检索分数", f"{rag['retrieval_score']:.3f}" if rag["retrieval_score"] else "N/A",
              help="检索到的文档平均相似度分数，0~1")
    st.caption(f"检索到 {rag['docs_retrieved']} 条文档")

    gov = metrics["governance"]
    col_g, col_c = st.columns(2)
    with col_g:
        st.metric("接地分", f"{gov['groundedness_score']:.2f}" if gov['groundedness_score'] else "N/A",
                  help="回答内容在检索文档中的可查证程度")
    with col_c:
        st.metric("完整度", f"{gov['completeness_score']:.2f}" if gov['completeness_score'] else "N/A",
                  help="回答对用户问题各子问题的覆盖程度")

    flow = metrics["flow"]
    st.metric("节点数 / 重试", f"{flow['nodes_visited_count']} 节点 / {flow['retry_count']} 次")

    st.divider()

    st.subheader("告警")
    if gov.get("fallback_triggered"):
        st.warning(f"LLM 降级: {gov.get('fallback_reason', '未知')}")
    else:
        st.success("未触发 LLM 降级")

    if hallucination["flag"]:
        st.error(f"幻觉告警: {hallucination['reason']}")
        if hallucination.get("rules_triggered"):
            st.caption("触发规则: " + ", ".join(hallucination["rules_triggered"]))
    else:
        st.success("未检测到幻觉")

    confidence = (result.get("reflection") or {}).get("confidence")
    st.metric("综合置信度", f"{confidence:.2f}" if confidence is not None else "N/A",
              help="四因子加权: 接地×0.4 + 完整×0.3 − 降级×0.2 − 幻觉×0.3")


def render_experience_ref(mode: str) -> None:
    """历史经验参考：展示本次 REFLECT prompt 携带的经验摘要"""
    st.subheader("历史经验参考")
    st.caption("本次 REFLECT prompt 中携带的参考经验")
    experiences = _get_experience_summaries(limit=3, mode=mode)
    if experiences and experiences != ["无历史经验"]:
        for exp in experiences:
            st.text(exp)
    else:
        st.caption("无历史经验")


# ============================================================
# 多轮对话状态
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, mode?, answer_source?}]

if "session_id" not in st.session_state:
    st.session_state.session_id = f"ui-{int(_time.time() * 1000)}"


def _build_history(n: int = 10) -> list[dict]:
    """取最近 N 条消息作为对话历史"""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-n:]
    ]


# ============================================================
# 输入区
# ============================================================

col_user, col_input, col_btn, col_clear = st.columns([1, 3.5, 1, 0.8])
with col_user:
    user_id = st.text_input("用户", value="default-user", key="user_id",
                            help="用于区分不同用户的学习进度")
with col_input:
    user_input = st.text_input(
        "输入问题",
        placeholder="例如：学习 LangGraph（我是入门水平）",
        key="input_box",
    )
with col_btn:
    send = st.button("发送", type="primary", use_container_width=True)
with col_clear:
    if st.button("清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"ui-{int(_time.time() * 1000)}"
        st.rerun()

# ============================================================
# 历史消息展示
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("mode"):
            st.caption(f"mode={msg['mode']} | source={msg.get('answer_source', 'N/A')}")

# ============================================================
# 执行
# ============================================================

if send and user_input.strip():
    # 强制清除 Streamlit 缓存的旧模块，确保导入最新版本
    import sys as _sys
    for _mod in list(_sys.modules.keys()):
        if _mod.startswith('runtime') or _mod.startswith('reflection') or _mod.startswith('memory') or _mod.startswith('observability'):
            del _sys.modules[_mod]
    history = _build_history()

    with st.spinner("Agent 运行中..."):
        result = _get_run_graph()(user_input.strip(),
                           session_id=st.session_state.session_id,
                           history=history,
                           user_id=user_id.strip() or "default-user")
        metrics = _get_compute_metrics()(result)
        tracer = _get_tracer()
        trace_summary = tracer.summary()
        hallucination = _get_detect_hallucination()(result)

    # ── 追加到消息历史 ──
    st.session_state.messages.append({"role": "human", "content": user_input.strip()})
    st.session_state.messages.append({
        "role": "ai",
        "content": result.get("final_output") or "（无输出）",
        "mode": result.get("mode"),
        "answer_source": result.get("answer_source"),
    })

    # ── 左侧：本轮回答 ──
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("本轮回答")
        final_output = result.get("final_output") or "（无输出）"
        st.markdown(final_output)
        st.caption(
            f"模式: {result.get('mode', '?')}  |  "
            f"来源: {result.get('answer_source') or 'N/A'}  |  "
            f"轮次: {len(st.session_state.messages) // 2}"
        )
        st.divider()
        render_flow_timeline(trace_summary.get("nodes_visited", []), tracer.records)

    with col_right:
        render_governance_dashboard(result, metrics, hallucination)
        st.divider()
        render_experience_ref(result.get("mode", "learn"))

elif not send and not st.session_state.messages:
    st.info("输入问题后点击「发送」启动 Agent 运行。支持多轮追问。")
