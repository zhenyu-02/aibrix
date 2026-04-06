import argparse
import json
import os
import re

import numpy as np
import pandas as pd


def jain_fairness_index(values):
    arr = np.array([float(v) for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    s = float(np.sum(arr))
    if s == 0.0:
        return 0.0
    denom = float(arr.size * np.sum(arr * arr))
    if denom == 0.0:
        return 0.0
    return (s * s) / denom


def _percentile(values, q):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float64), q))


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v, default=0):
    try:
        return int(v)
    except Exception:
        return int(default)


def main():
    parser = argparse.ArgumentParser(description="Analyze sessioned benchmark trace (separate from analyze.py)")
    parser.add_argument("--trace", required=True, help="Path to benchmark trace jsonl")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory or summary csv path. If ends with .csv/.tsv, writes summary to that file and per_user to *_per_user.csv",
    )
    args = parser.parse_args()

    input_file = args.trace
    output_arg = args.output
    is_file_output = bool(re.search(r"\.(csv|tsv)$", str(output_arg), flags=re.IGNORECASE))
    if is_file_output:
        output_dir = os.path.dirname(os.path.abspath(output_arg)) or os.getcwd()
        summary_path = os.path.abspath(output_arg)
        base, _ = os.path.splitext(summary_path)
        per_user_path = base + "_per_user.csv"
    else:
        output_dir = os.path.abspath(output_arg)
        summary_path = os.path.join(output_dir, "summary.csv")
        per_user_path = os.path.join(output_dir, "per_user.csv")
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        pd.DataFrame([]).to_csv(summary_path, index=False)
        pd.DataFrame([]).to_csv(per_user_path, index=False)
        return

    # Basic arrays
    start_times = [_safe_float(r.get("start_time", 0.0)) for r in rows]
    end_times = [_safe_float(r.get("end_time", 0.0)) for r in rows]
    statuses = [r.get("status") for r in rows]
    users = [r.get("user") or "unknown" for r in rows]

    latencies = [_safe_float(r.get("latency", 0.0)) for r in rows]
    prompt_tokens = [_safe_int(r.get("prompt_tokens", 0)) for r in rows]
    output_tokens = [_safe_int(r.get("output_tokens", 0)) for r in rows]
    total_tokens = [_safe_int(r.get("total_tokens", 0)) for r in rows]
    ttft = [r.get("ttft") for r in rows]
    tpot = [r.get("tpot") for r in rows]

    total_cnt = int(len(rows))
    success_cnt = int(sum(1 for s in statuses if s == "success"))
    error_cnt = int(sum(1 for s in statuses if s == "error"))
    error_rate = float(error_cnt / total_cnt) if total_cnt > 0 else 0.0

    duration_s = float(max(end_times) - min(start_times)) if end_times and start_times else 0.0
    total_tok_sum = float(np.sum(total_tokens))
    e2e_tok_per_s = float(total_tok_sum / duration_s) if duration_s > 0 else 0.0
    success_req_per_s = float(success_cnt / duration_s) if duration_s > 0 else 0.0

    # Per-user aggregation (success only)
    per_user_total = {}
    per_user_out = {}
    per_user_cnt = {}
    per_user_lat_sum = {}
    for r in rows:
        if r.get("status") != "success":
            continue
        u = r.get("user") or "unknown"
        per_user_total[u] = per_user_total.get(u, 0.0) + _safe_float(r.get("total_tokens", 0.0))
        per_user_out[u] = per_user_out.get(u, 0.0) + _safe_float(r.get("output_tokens", 0.0))
        per_user_cnt[u] = per_user_cnt.get(u, 0) + 1
        per_user_lat_sum[u] = per_user_lat_sum.get(u, 0.0) + _safe_float(r.get("latency", 0.0))

    users_seen = sorted(per_user_total.keys())
    fairness_tokens = None
    fairness_output = None
    fairness_success = None
    fairness_eff_out_tput = None
    if users_seen:
        tokens_values = [per_user_total[u] for u in users_seen]
        output_values = [per_user_out.get(u, 0.0) for u in users_seen]
        goodput_values = [per_user_cnt.get(u, 0) for u in users_seen]
        eff_tput_values = []
        for u in users_seen:
            lat_sum = per_user_lat_sum.get(u, 0.0)
            out_sum = per_user_out.get(u, 0.0)
            eff_tput_values.append(out_sum / lat_sum if lat_sum > 0 else 0.0)
        fairness_tokens = float(jain_fairness_index(tokens_values))
        fairness_output = float(jain_fairness_index(output_values))
        fairness_success = float(jain_fairness_index(goodput_values))
        fairness_eff_out_tput = float(jain_fairness_index(eff_tput_values))

    # Tail stats (prefer success-only latency for comparison)
    success_lat = [
        _safe_float(r.get("latency", 0.0))
        for r in rows
        if r.get("status") == "success" and r.get("latency") is not None
    ]
    success_ttft = [
        _safe_float(r.get("ttft", 0.0))
        for r in rows
        if r.get("status") == "success" and r.get("ttft") is not None
    ]
    success_tpot = [
        _safe_float(r.get("tpot", 0.0))
        for r in rows
        if r.get("status") == "success" and r.get("tpot") is not None
    ]

    summary_row = {
        "trace": os.path.abspath(input_file),
        "total_requests": total_cnt,
        "success_requests": success_cnt,
        "error_requests": error_cnt,
        "error_rate": error_rate,
        "duration_s": duration_s,
        "total_tokens": total_tok_sum,
        "e2e_tokens_per_s": e2e_tok_per_s,
        "success_req_per_s": success_req_per_s,
        "e2e_latency_p50_s": _percentile(success_lat, 50),
        "e2e_latency_p99_s": _percentile(success_lat, 99),
        "ttft_p50_s": _percentile(success_ttft, 50),
        "ttft_p99_s": _percentile(success_ttft, 99),
        "tpot_p50_s": _percentile(success_tpot, 50),
        "tpot_p99_s": _percentile(success_tpot, 99),
        "users_seen": int(len(users_seen)),
        "jain_total_tokens": fairness_tokens,
        "jain_output_tokens": fairness_output,
        "jain_success_count": fairness_success,
        "jain_eff_output_tput": fairness_eff_out_tput,
    }
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    if users_seen:
        per_user_df = pd.DataFrame(
            {
                "user": users_seen,
                "success_requests": [per_user_cnt.get(u, 0) for u in users_seen],
                "total_tokens": [per_user_total.get(u, 0.0) for u in users_seen],
                "output_tokens": [per_user_out.get(u, 0.0) for u in users_seen],
                "latency_sum_s": [per_user_lat_sum.get(u, 0.0) for u in users_seen],
                "eff_output_tput": [
                    (per_user_out.get(u, 0.0) / per_user_lat_sum.get(u, 0.0)) if per_user_lat_sum.get(u, 0.0) > 0 else 0.0
                    for u in users_seen
                ],
            }
        )
    else:
        per_user_df = pd.DataFrame([])
    per_user_df.to_csv(per_user_path, index=False)


if __name__ == "__main__":
    main()

