import logging
import json
import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def jain_fairness_index(values):
    """Jain's fairness index.

    公式：J(x) = (sum(x))^2 / (n * sum(x^2))
    - values 中允许 0，但不允许全 0（全 0 时返回 0）
    """
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

def parse_goodput_target(goodput_target):
    pattern = r'^(e2e|tpot|ttft):(-?\d+(\.\d+)?)$'
    match = re.match(pattern, goodput_target)
    
    if match:
        metric = match.group(1)
        threshold = float(match.group(2))  # Convert to float
    else:
        raise ValueError(f"Invalid goodput spec: {goodput_target}")
    return metric, threshold
    
def main(args):
    input_file = args.trace
    output_arg = args.output
    # 兼容两种用法：
    # 1) --output <dir>：输出目录（保持原行为，生成 pdf + summary/per_user csv）
    # 2) --output <file.csv>：输出单个 summary csv，并在同目录生成 per_user csv 与 pdf
    is_file_output = bool(re.search(r"\.(csv|tsv)$", str(output_arg), flags=re.IGNORECASE))
    if is_file_output:
        output_dir = os.path.dirname(os.path.abspath(output_arg)) or os.getcwd()
        summary_csv_path = os.path.abspath(output_arg)
        base, _ = os.path.splitext(summary_csv_path)
        per_user_csv_path = base + "_per_user.csv"
    else:
        output_dir = os.path.abspath(output_arg)
        summary_csv_path = os.path.join(output_dir, "summary.csv")
        per_user_csv_path = os.path.join(output_dir, "per_user.csv")
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    # Extract metrics
    prompt_tokens = []
    output_tokens = []
    total_tokens = []
    latencies = []
    throughputs = []
    tokens_per_second = []
    ttft = []
    tpot = []
    total_errors = []
    timestamps = []
    end_times = []
    users = []
    per_user_total_tokens = {}
    per_user_output_tokens = {}
    per_user_success_cnt = {}
    per_user_latency_sum = {}
    for i, item in enumerate(data):
        user = item.get("user", "")
        if not user:
            user = "unknown"
        users.append(user)

        total_errors.append(1 if item["status"] == "error" else 0)
        timestamps.append(item.get("start_time", 0))
        end_times.append(item.get("end_time", 0))
        prompt_tokens.append(item["prompt_tokens"]) # Prompt tokens
        output_tokens.append(item["output_tokens"]) 
        total_tokens.append(item["total_tokens"]) 
        latencies.append(item["latency"])
        throughputs.append(item["throughput"])
        tokens_per_second.append(item["total_tokens"] / item["latency"])
        ttft.append(item["ttft"] if "ttft" in item else 0.0) # Time to First Token
        tpot.append(item["tpot"] if "tpot" in item else 0.0) # Time per Output Token

        # 仅用成功请求统计公平性（失败请求 total_tokens=0 会强烈干扰）
        if item.get("status") == "success":
            per_user_total_tokens[user] = per_user_total_tokens.get(user, 0.0) + float(item.get("total_tokens", 0))
            per_user_output_tokens[user] = per_user_output_tokens.get(user, 0.0) + float(item.get("output_tokens", 0))
            per_user_success_cnt[user] = per_user_success_cnt.get(user, 0) + 1
            per_user_latency_sum[user] = per_user_latency_sum.get(user, 0.0) + float(item.get("latency", 0.0))
    goodput = 0.0
    if args.goodput_target is not None:
        metric, threshold = parse_goodput_target(args.goodput_target)
        if metric == "e2e":
            if len(latencies) > 0:
                goodput = len([item for item in latencies if (item is not None and item <= threshold)]) / float(len(latencies))
        elif metric == "ttft":
            if len(ttft) > 0:
                goodput = len([item for item in ttft if (item is not None  and item <= threshold)]) / float(len(ttft))
        elif metric == "tpot":
            if len(tpot) > 0:
                goodput = len([item for item in tpot if (item is not None and item <= threshold)]) / float(len(tpot))
        else:
            raise ValueError(f"Invalid goodput target: {args.goodput_target}")

    # Sort data by start_time
    sorted_indices = np.argsort(timestamps)
    timestamps = [timestamps[i] for i in sorted_indices]
    prompt_tokens = [prompt_tokens[i] for i in sorted_indices]
    output_tokens = [output_tokens[i] for i in sorted_indices]
    total_tokens = [total_tokens[i] for i in sorted_indices]
    latencies = [latencies[i] for i in sorted_indices]
    throughputs = [throughputs[i] for i in sorted_indices]
    tokens_per_second = [tokens_per_second[i] for i in sorted_indices]
    ttft = [ttft[i] for i in sorted_indices]
    tpot = [tpot[i] for i in sorted_indices]
    start_times = timestamps
    

    # Convert timestamps to pandas datetime (if timestamps are actual time values)
    try:
        timestamps = pd.to_datetime(timestamps, unit='s')
    except Exception:
        timestamps = pd.Series(timestamps)

    # Helper function to calculate statistics
    def calculate_statistics(values):
        values = [value for value in values if value is not None]
        if len(values) == 0:
            return 0.0, 0.0, 0.0, 0.0
        total = sum(values)
        values = sorted(values)
        avg = sum(values) / len(values)
        median = np.median(values)
        percentile_99 = np.percentile(values, 99)
        return total, avg, median, percentile_99

    # Calculate statistics for each metric
    stats = {
        "End-to-End Latency (s)": calculate_statistics(latencies),
        "Throughput (per request, toks/s)": calculate_statistics(throughputs),
        "Tokens per Second": calculate_statistics(tokens_per_second),
        "Request Prompt Tokens": calculate_statistics(prompt_tokens),
        "Request Output Tokens": calculate_statistics(output_tokens),
        "Request Total Tokens": calculate_statistics(total_tokens),
        "Time to First Token (TTFT)": calculate_statistics(ttft),
        "Time per Output Token (TPOT)": calculate_statistics(tpot),
        "Errors": calculate_statistics(total_errors),
    }

    # Print statistics
    for metric, (total, avg, median, p99) in stats.items():
        logging.warning(f"{metric} Statistics: Total = {total:.4f} Average = {avg:.4f}, Median = {median:.4f}, 99th Percentile = {p99:.4f}")
    if goodput != None:
        logging.warning(f"Goodput (reqs/s) {goodput:.4f}")
    logging.warning(f"Total requests : {len(data)}")
    logging.warning(f"Total Duration (s): {np.max(end_times) - np.min(start_times)}")
    logging.warning(f"Total tokens generated (toks): {np.sum(total_tokens)}")
    logging.warning(f"Throughput (end-to-end, toks/s): {np.sum(total_tokens)/(np.max(end_times) - np.min(start_times))}")

    # Jain fairness（按用户聚合）
    # 1) tokens served fairness：看资源分配是否均衡
    # 2) goodput fairness：看成功请求数是否均衡
    # 3) effective throughput fairness：每用户 output_tokens / latency_sum
    users_seen = sorted(per_user_total_tokens.keys())
    fairness_tokens = None
    fairness_output = None
    fairness_success = None
    fairness_eff_out_tput = None
    if users_seen:
        tokens_values = [per_user_total_tokens[u] for u in users_seen]
        output_values = [per_user_output_tokens.get(u, 0.0) for u in users_seen]
        goodput_values = [per_user_success_cnt.get(u, 0) for u in users_seen]
        eff_tput_values = []
        for u in users_seen:
            lat_sum = per_user_latency_sum.get(u, 0.0)
            out_sum = per_user_output_tokens.get(u, 0.0)
            eff_tput_values.append(out_sum / lat_sum if lat_sum > 0 else 0.0)

        fairness_tokens = float(jain_fairness_index(tokens_values))
        fairness_output = float(jain_fairness_index(output_values))
        fairness_success = float(jain_fairness_index(goodput_values))
        fairness_eff_out_tput = float(jain_fairness_index(eff_tput_values))

        logging.warning(f"Users seen: {len(users_seen)}")
        logging.warning(f"Jain fairness (total_tokens per user): {fairness_tokens:.4f}")
        logging.warning(f"Jain fairness (output_tokens per user): {fairness_output:.4f}")
        logging.warning(f"Jain fairness (success_count per user): {fairness_success:.4f}")
        logging.warning(f"Jain fairness (effective_output_tput per user): {fairness_eff_out_tput:.4f}")
    # logging.warning(f"Failure Rate (%) {(total_errors / len(data)) * 100 if len(data) > 0 else 0}")

    # Create a DataFrame for plotting
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "User": users,
        "Prompt Tokens": prompt_tokens,
        "Output Tokens": output_tokens,
        "Total Tokens": total_tokens,
        "End-to-End Latency (s)": latencies,
        "Throughput": throughputs,
        "Tokens per Second": tokens_per_second,
        "Time to First Token (TTFT)": ttft,
        "Time per Output Token (TPOT)": tpot,
        "Errors": total_errors,
    }).set_index("Timestamp")

    # 写出 summary/per-user 聚合结果（便于 vtc-basic vs vtc-pred 对比）
    os.makedirs(output_dir, exist_ok=True)
    duration_s = float(np.max(end_times) - np.min(start_times)) if len(end_times) > 0 else 0.0
    success_cnt = int(sum(1 for x in data if x.get("status") == "success"))
    error_cnt = int(sum(1 for x in data if x.get("status") == "error"))
    total_cnt = int(len(data))
    error_rate = float(error_cnt / total_cnt) if total_cnt > 0 else 0.0
    e2e_tok_per_s = float(np.sum(total_tokens) / duration_s) if duration_s > 0 else 0.0
    req_per_s = float(success_cnt / duration_s) if duration_s > 0 else 0.0

    # 选取核心 tail 指标（p50/p99）
    e2e_latency_p50 = float(stats["End-to-End Latency (s)"][2])
    e2e_latency_p99 = float(stats["End-to-End Latency (s)"][3])
    ttft_p50 = float(stats["Time to First Token (TTFT)"][2])
    ttft_p99 = float(stats["Time to First Token (TTFT)"][3])
    tpot_p50 = float(stats["Time per Output Token (TPOT)"][2])
    tpot_p99 = float(stats["Time per Output Token (TPOT)"][3])

    summary_row = {
        "trace": os.path.abspath(input_file),
        "total_requests": total_cnt,
        "success_requests": success_cnt,
        "error_requests": error_cnt,
        "error_rate": error_rate,
        "duration_s": duration_s,
        "total_tokens": float(np.sum(total_tokens)),
        "e2e_tokens_per_s": e2e_tok_per_s,
        "success_req_per_s": req_per_s,
        "e2e_latency_p50_s": e2e_latency_p50,
        "e2e_latency_p99_s": e2e_latency_p99,
        "ttft_p50_s": ttft_p50,
        "ttft_p99_s": ttft_p99,
        "tpot_p50_s": tpot_p50,
        "tpot_p99_s": tpot_p99,
        "users_seen": int(len(users_seen)),
        "jain_total_tokens": fairness_tokens,
        "jain_output_tokens": fairness_output,
        "jain_success_count": fairness_success,
        "jain_eff_output_tput": fairness_eff_out_tput,
    }
    pd.DataFrame([summary_row]).to_csv(summary_csv_path, index=False)

    if users_seen:
        per_user_df = pd.DataFrame({
            "user": users_seen,
            "total_tokens": [per_user_total_tokens.get(u, 0.0) for u in users_seen],
            "output_tokens": [per_user_output_tokens.get(u, 0.0) for u in users_seen],
            "success_count": [per_user_success_cnt.get(u, 0) for u in users_seen],
            "latency_sum_s": [per_user_latency_sum.get(u, 0.0) for u in users_seen],
        })
        per_user_df["eff_output_tput"] = per_user_df.apply(
            lambda r: (float(r["output_tokens"]) / float(r["latency_sum_s"])) if float(r["latency_sum_s"]) > 0 else 0.0,
            axis=1,
        )
        per_user_df.to_csv(per_user_csv_path, index=False)

    # Plot each metric in a separate subplot
    num_metrics = len(df.columns)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 4 * num_metrics), sharex=True)

    for ax, (column, values) in zip(axes, df.items()):
        ax.plot(df.index, values, marker='o', linestyle='-', label=column)
        ax.set_ylabel(column)
        ax.legend()
        ax.grid()

    axes[-1].set_xlabel("Time")  # Only set x-axis label for the last subplot
    plt.suptitle("Time Series Analysis of LLM Performance Metrics")
    plt.xticks(rotation=45)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
    plt.savefig(os.path.join(output_dir, "performance_metrics_time_series.pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract and plot performance metrics from a JSONL file')
    parser.add_argument('--trace', type=str, required=True, help='Input trace containing collected metrics.')
    parser.add_argument('--output', type=str, required=True, default="output", help='Output path.')
    parser.add_argument('--goodput-target', type=str, required=False, default=None, help='Goodput target should be in the format of latency_metrics:threshold_in_seconds, choose latency metrics from one of the e2e, ttft, tpot.')
    
    args = parser.parse_args()
    main(args)
    
