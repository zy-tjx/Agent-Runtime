"""
评估报告生成器
读 results.json → 聚合统计 + 分布 + 分组分析
支持 --compare 双报告对比
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math

DEFAULT_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")


def report(results_file: str = DEFAULT_RESULTS) -> dict:
    """生成聚合报告并打印"""
    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    success = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    total = len(results)

    # ── 提取数值序列 ──
    def _nums(key, records=None):
        recs = records or success
        vals = []
        for r in recs:
            v = r.get("actual", {}).get(key)
            if v is not None and isinstance(v, (int, float)) and not math.isnan(v):
                vals.append(v)
        return vals

    retrieval_scores = _nums("retrieval_score")
    groundedness = _nums("groundedness_score")
    completeness = _nums("completeness_score")
    durations = [r["run_duration_ms"] for r in results]
    fallback_count = sum(1 for r in success if r["actual"].get("fallback_triggered"))
    hallucination_count = sum(1 for r in success if r.get("hallucination", {}).get("flag"))

    # ── 聚合 ──
    stats = {
        "total": total,
        "success": len(success),
        "failed": len(failed),
        "success_rate": len(success) / total if total else 0,
        "total_duration_ms": data.get("total_duration_ms", sum(durations)),
        "avg_duration_ms": _mean(durations),
        "retrieval_score": _dist(retrieval_scores),
        "groundedness_score": _dist(groundedness),
        "completeness_score": _dist(completeness),
        "fallback_rate": fallback_count / len(success) if success else 0,
        "hallucination_rate": hallucination_count / len(success) if success else 0,
    }

    # ── 分组分析 ──
    by_mode = _group_by(success, lambda r: r["actual"].get("mode", "?"))
    by_difficulty = _group_by(success, lambda r: r["expected"]["difficulty"])
    by_retrieve = _group_by(success, lambda r: "hit" if r["expected"]["should_retrieve"] else "miss")

    report = {
        "stats": stats,
        "by_mode": {k: _group_stats(v) for k, v in by_mode.items()},
        "by_difficulty": {k: _group_stats(v) for k, v in by_difficulty.items()},
        "by_retrieve": {k: _group_stats(v) for k, v in by_retrieve.items()},
        "failures": [
            {"id": r["id"], "error": r.get("error", "unknown")} for r in failed
        ],
    }

    _print_report(report)
    return report


def compare(file_a: str, file_b: str, label_a: str = "A", label_b: str = "B") -> None:
    """对比两次运行的聚合指标"""
    ra = report(file_a)
    rb = report(file_b)

    print("\n" + "=" * 60)
    print(f"对比: {label_a} vs {label_b}")
    print("=" * 60)

    sa = ra["stats"]
    sb = rb["stats"]

    rows = [
        ("成功率", f"{sa['success_rate']:.1%}", f"{sb['success_rate']:.1%}"),
        ("平均耗时", f"{sa['avg_duration_ms']:.0f}ms", f"{sb['avg_duration_ms']:.0f}ms"),
        ("平均检索分", f"{sa['retrieval_score']['mean']:.3f}", f"{sb['retrieval_score']['mean']:.3f}"),
        ("平均接地分", f"{sa['groundedness_score']['mean']:.3f}", f"{sb['groundedness_score']['mean']:.3f}"),
        ("平均完整度", f"{sa['completeness_score']['mean']:.3f}", f"{sb['completeness_score']['mean']:.3f}"),
        ("降级率", f"{sa['fallback_rate']:.1%}", f"{sb['fallback_rate']:.1%}"),
        ("幻觉率", f"{sa['hallucination_rate']:.1%}", f"{sb['hallucination_rate']:.1%}"),
    ]

    print(f"{'指标':<16} {label_a:<14} {label_b:<14} 差异")
    print("-" * 58)
    for name, a, b in rows:
        # 尝试计算数值差异
        diff = ""
        try:
            va = float(a.replace("ms", "").replace("%", ""))
            vb = float(b.replace("ms", "").replace("%", ""))
            if "%" in a:
                d = (vb - va)
                diff = f"{d:+.1f}pp"
            else:
                d = vb - va
                diff = f"{d:+.0f}"
        except ValueError:
            pass
        print(f"{name:<16} {a:<14} {b:<14} {diff}")


# ── 内部 ──

def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _p50(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


def _p95(vals: list[float]) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return s[int(len(s) * 0.95)]


def _dist(vals: list[float]) -> dict:
    return {
        "count": len(vals),
        "mean": round(_mean(vals), 3) if vals else None,
        "p50": round(_p50(vals), 3) if vals else None,
        "p95": round(_p95(vals), 3) if vals else None,
    }


def _group_by(records: list, key_fn) -> dict[str, list]:
    groups: dict[str, list] = {}
    for r in records:
        k = key_fn(r)
        groups.setdefault(k, []).append(r)
    return groups


def _group_stats(records: list) -> dict:
    if not records:
        return {"count": 0}
    retrieval = [r["actual"].get("retrieval_score") for r in records
                 if r["actual"].get("retrieval_score") is not None]
    grounded = [r["actual"].get("groundedness_score") for r in records
                if r["actual"].get("groundedness_score") is not None]
    fallback = sum(1 for r in records if r["actual"].get("fallback_triggered"))
    return {
        "count": len(records),
        "avg_retrieval": round(_mean(retrieval), 3) if retrieval else None,
        "avg_groundedness": round(_mean(grounded), 3) if grounded else None,
        "fallback_count": fallback,
    }


def _print_report(r: dict) -> None:
    s = r["stats"]
    print("=" * 56)
    print("Agent Runtime 评估报告")
    print("=" * 56)
    print(f"成功率: {s['success']}/{s['total']} ({s['success_rate']:.1%})")
    print(f"总耗时: {s['total_duration_ms']}ms  |  平均: {s['avg_duration_ms']:.0f}ms/题")
    print()

    # 分数分布
    print("── 分数分布（mean / P50 / P95） ──")
    for label, key in [("检索分", "retrieval_score"), ("接地分", "groundedness_score"), ("完整度", "completeness_score")]:
        d = s[key]
        if d["count"] > 0:
            print(f"  {label}: {d['mean']} / {d['p50']} / {d['p95']}  (n={d['count']})")
        else:
            print(f"  {label}: N/A")
    print()

    # 失败分析
    print(f"降级率: {s['fallback_rate']:.1%}  |  幻觉率: {s['hallucination_rate']:.1%}")
    if r["failures"]:
        print(f"失败 ({len(r['failures'])}):")
        for f in r["failures"]:
            print(f"  - {f['id']}: {f['error'][:80]}")
    print()

    # 分组
    print("── 按模式 ──")
    for mode, gs in r["by_mode"].items():
        print(f"  {mode}: {gs['count']}题, 检索均分={gs['avg_retrieval']}, "
              f"接地均分={gs['avg_groundedness']}, 降级={gs['fallback_count']}")

    print("── 按难度 ──")
    for diff, gs in r["by_difficulty"].items():
        print(f"  {diff}: {gs['count']}题, 检索均分={gs['avg_retrieval']}, "
              f"接地均分={gs['avg_groundedness']}")

    print("── 按检索命中预期 ──")
    for hit, gs in r["by_retrieve"].items():
        print(f"  {hit}: {gs['count']}题, 检索均分={gs['avg_retrieval']}")
    print()


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3],
                label_a=sys.argv[4] if len(sys.argv) > 4 else "A",
                label_b=sys.argv[5] if len(sys.argv) > 5 else "B")
    elif len(sys.argv) >= 2:
        report(sys.argv[1])
    else:
        report()
