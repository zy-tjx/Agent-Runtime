"""
批量评估运行器
读取 eval_questions.json → 逐条 run_graph() → 收集全量指标 → 输出 JSON
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import io
import contextlib
from runtime.state_graph import run_graph
from reflection.eval_metrics import compute_metrics
from reflection.hallucination_detector import detect as detect_hallucination
from observability.tracer import get_tracer, reset_tracer

QUESTIONS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "test_data", "eval_questions.json",
)
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def run_eval(questions_file: str = QUESTIONS_FILE, output_file: str = OUTPUT_FILE) -> list[dict]:
    """逐条运行评估，返回结果列表并写入 JSON"""
    with open(questions_file, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total = len(questions)
    start_all = time.time()

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        question = q["question"]
        print(f"[{i}/{total}] {qid}: {question[:50]}...", end=" ", flush=True)

        run_start = time.time()
        try:
            reset_tracer()
            with contextlib.redirect_stdout(io.StringIO()):
                state = run_graph(question, session_id=f"eval-{qid}")

            metrics = compute_metrics(state)
            hallucination = detect_hallucination(state)
            trace_records = get_tracer().records
            duration_ms = int((time.time() - run_start) * 1000)

            result = {
                "id": qid,
                "question": question,
                "expected": {
                    "mode": q["mode_expectation"],
                    "difficulty": q["difficulty"],
                    "should_retrieve": q["should_retrieve"],
                },
                "actual": {
                    "mode": state.get("mode"),
                    "final_output": (state.get("final_output") or "")[:200],
                    "retrieval_score": state.get("retrieval_score"),
                    "groundedness_score": state.get("groundedness_score"),
                    "completeness_score": state.get("completeness_score"),
                    "confidence": (state.get("reflection") or {}).get("confidence"),
                    "answer_source": state.get("answer_source"),
                    "retry_count": state.get("retry_count", 0),
                    "error": state.get("error"),
                    "fallback_triggered": state.get("fallback_triggered", False),
                    "fallback_reason": state.get("fallback_reason"),
                },
                "metrics": metrics,
                "hallucination": {
                    "flag": hallucination["flag"],
                    "severity": hallucination["severity"],
                    "rules_triggered": hallucination.get("rules_triggered", []),
                },
                "trace": {
                    "nodes_visited": [r["node"] for r in trace_records],
                    "total_duration_ms": sum(r["duration_ms"] for r in trace_records),
                    "node_count": len(trace_records),
                },
                "run_duration_ms": duration_ms,
                "success": True,
            }
        except Exception as e:
            duration_ms = int((time.time() - run_start) * 1000)
            import traceback
            traceback.print_exc()
            print("", flush=True)
            result = {
                "id": qid,
                "question": question,
                "expected": {
                    "mode": q["mode_expectation"],
                    "difficulty": q["difficulty"],
                    "should_retrieve": q["should_retrieve"],
                },
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "run_duration_ms": duration_ms,
            }

        results.append(result)

    total_ms = int((time.time() - start_all) * 1000)

    # ── 写入输出文件 ──
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"total_duration_ms": total_ms, "results": results}, f,
                  ensure_ascii=False, indent=2)

    success_count = sum(1 for r in results if r["success"])
    print(f"\n完成: {success_count}/{total} 成功, 总耗时 {total_ms}ms, 结果已写入 {output_file}")
    return results


if __name__ == "__main__":
    run_eval()
