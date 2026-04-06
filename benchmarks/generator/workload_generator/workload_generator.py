import logging
import argparse
import time
import csv
import os
import sys
import glob
import hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm

from pandas import Timedelta
from typing import List, Dict, Any, Optional
try:
    from transformers import PreTrainedTokenizerBase
except ImportError:  # Optional dependency (e.g., CPU-only / burstgpt trace generation)
    class PreTrainedTokenizerBase:  # type: ignore
        pass
from datetime import timedelta

# Allow running this file directly from the repo root (without setting PYTHONPATH).
# The `generator` package lives under `benchmarks/`, so we add that directory.
_BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)

from generator.workload_generator.sample_request import (load_requests,  
                            RequestFinder,
                        )
from generator.workload_generator.distribution import (generate_poisson_dist,
                          generate_token_len_from_percentiles,
                          to_fluctuate_pattern_config,
                          user_to_synthetic_config,
                          sine_fluctuation,
                          )
                          
from generator.workload_generator.utils import (if_sessioned_dataset,
                   convert_to_stat_df,
                   read_distribution_stats,
                   get_tokenizer, 
                   plot_workload, 
                   make_serializable, 
                   load_json,
                   load_jsonl,
                   save_workload, 
                   )

from generator.dataset_generator.synthetic_prompt import (generate_synthetic_prompt,
                                                          adjust_prompt_length,
                                                        )

# Set up logging to print only warning and above level messages
logging.basicConfig(level=logging.INFO)


def generate_from_stat_csv(prompt_file_path: str, 
                            duration_ms: int,
                            tokenizer: PreTrainedTokenizerBase,       
                            qps_stat: str = None,
                            input_stat: str = None,
                            output_stat: str = None,
                            qps_scale: float = 1.0,
                            input_scale: float = 1.0,
                            output_scale: float = 1.0,
                            stat_trace_type: str = 'maas',
                            max_concurrent_sessions: int = None,
                            output_file: str = 'output/output',
                            to_jsonl: bool = False,
                            ) -> Dict[str, Any]:
    merged_df = convert_to_stat_df(qps_stat, input_stat, output_stat, stat_trace_type)
    input_len_configs, output_len_configs, rps_configs = read_distribution_stats(merged_df)
    input_len_dist = []
    output_len_dist = []
    rps_dist = []
    for rps_config in rps_configs:
        rps_segment = generate_poisson_dist(target = rps_config['mean_rps'], sample_size = rps_config['total_seconds'], smooth_window_size = 10)
        rps_dist.extend(rps_segment)
    if stat_trace_type == "maas":
        for config in input_len_configs:
            config['scale'] = input_scale
            input_segment = generate_token_len_from_percentiles(**config)
            input_len_dist.extend(input_segment)
        for config in output_len_configs:
            config['scale'] = output_scale
            output_segment = generate_token_len_from_percentiles(**config)
            output_len_dist.extend(output_segment)
    elif stat_trace_type == "cloudide":
        for config in input_len_configs:
            config['scale'] = input_scale
            input_segment = generate_token_len_from_percentiles(**config)
            input_len_dist.extend(input_segment)
            output_segment = generate_token_len_from_percentiles(**config)
            output_len_dist.extend(output_segment)
    
    workload = generate_synthetic_from_dist(
        prompt_file_path = prompt_file_path,
        tokenizer = tokenizer,
        duration_ms =  duration_ms,
        rps_dist = rps_dist,
        input_token_len_dist = input_len_dist,
        output_token_len_dist = output_len_dist,
        qps_scale = qps_scale,
        input_scale = input_scale,
        output_scale = output_scale,
        max_concurrent_sessions = max_concurrent_sessions,
    )
    
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload
    
def generate_synthetic_from_dist(
        prompt_file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        duration_ms: int,
        rps_dist: List[int],
        input_token_len_dist: List[int],
        output_token_len_dist: List[int],
        qps_scale: float,
        input_scale: float,
        output_scale: float,
        max_concurrent_sessions: int,
    ) -> List[Dict[str, Any]]:
    
    if input_token_len_dist is not None and output_token_len_dist is not None: 
        if not (len(rps_dist) == len(input_token_len_dist) == len(output_token_len_dist)):
            raise ValueError(f"All distributions must have the same length, len(rps_dist): {len(rps_dist)}, len(input_token_len_dist): {len(input_token_len_dist)}, len(output_token_len_dist): {len(output_token_len_dist)}")
    workload = []
    current_time = 0
    total_seconds = len(rps_dist)
    logging.debug(f"total_seconds {total_seconds} rps_dist {rps_dist}")
    ts = time.time()
    prompt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)
    logging.info(f"Load requests took {int(time.time() - ts)}s")
    request_finder = RequestFinder(df=prompt_df)
    while current_time < total_seconds * 1000:
        time_idx = int(current_time / 1000)
        if time_idx >= total_seconds:
            time_idx = total_seconds - 1
        current_rate = rps_dist[time_idx] / qps_scale
        current_input_len = None
        current_output_len = None
        if input_token_len_dist is not None:
            current_input_len = input_token_len_dist[time_idx] / input_scale if input_token_len_dist[time_idx] else None 
        if output_token_len_dist is not None: 
            current_output_len = output_token_len_dist[time_idx] / output_scale if output_token_len_dist[time_idx] else None
        inter_arrival_time = 1000 if current_rate == 0 else np.random.exponential(scale=1000/current_rate) 
        current_time += inter_arrival_time
        if current_time < total_seconds * 1000:
            if current_rate != 0:
                if max_concurrent_sessions and if_sessioned_dataset(prompt_df):
                    request = request_finder.find_requests_max_session(
                        num_requests=1,
                        max_concurrent_session = max_concurrent_sessions,
                    )
                else:
                    request = request_finder.find_requests_len_range(
                        num_requests=1,
                        input_lens=[current_input_len],
                        output_lens=[current_output_len],
                        initial_err_perc=0.5,
                        err_step=0.05,
                    )
            else:
                request = []
            workload.append({"timestamp": int(current_time), "requests": request})  
            if current_time > duration_ms:
                break
        
    return workload

def generate_constant(prompt_file_path: str,
                    tokenizer: PreTrainedTokenizerBase,
                    qps: int, 
                    input_len: int = None,
                    output_len: int = None,
                    duration_ms: int = None,
                    interval_ms: int = None,
                    max_concurrent_sessions: int = None,
                    output_file: str = 'output/output',
                    to_jsonl: bool = False,
                    ) -> List[List[Any]]:
    workload = []
    ts = 0
    
    # if input_len != None and output_len != None:
    rps_dist = []
    input_token_len_dist = None
    if input_len != None:
        input_token_len_dist = []
    if output_len != None:
        output_token_len_dist = []
    output_token_len_dist = None
    while ts < duration_ms:
        rps_dist.append(qps)
        if input_len != None:
            input_token_len_dist.append(input_len)
        if output_len != None:
            output_token_len_dist.append(output_len)
        ts += interval_ms
    workload = generate_synthetic_from_dist(
        prompt_file_path = prompt_file_path,
        tokenizer = tokenizer,
        duration_ms =  duration_ms,
        rps_dist = rps_dist,
        input_token_len_dist = input_token_len_dist,
        output_token_len_dist = output_token_len_dist,
        qps_scale = 1.0,
        input_scale = 1.0,
        output_scale = 1.0,
        max_concurrent_sessions = max_concurrent_sessions
    )
        
    
    ### Generate constant load for all requests
    # idx = 0
    # request_finder = RequestFinder(df=sharegpt_df)
    # while idx < len(sharegpt_df):
    #     concurrent_reqs = request_finder.sample_requests_all(start_idx=idx, qps=qps)
    #     workload.append({"timestamp": ts, "requests": concurrent_reqs})  
    #     idx += qps
    #     ts += interval_ms
   
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload

def generate_synthetic(prompt_file_path: str,
                       tokenizer: PreTrainedTokenizerBase,
                       qps_pattern_config: Dict[str, Any],
                       input_pattern_config: Dict[str, Any],
                       output_pattern_config: Dict[str, Any],
                       duration_ms: int = None,
                       interval_ms: int = None,
                       max_concurrent_sessions: int = None,
                       output_file: str = 'output/output',
                       to_jsonl: bool = False,
                       ) -> List[List[Any]]:
    """
    Generates a workload based on a given list of input requests and a concurrency function.

    The concurrency function is defined as:
        concurrency(t) = trend(t) + noise
        trend(t) = A * sin(omega * t) + B
        noise ~ N(0, sigma^2)

    Args:
        input_requests (list): The list of all requests to be sent.
        A (float, optional): The amplitude of the sine wave in the concurrency function. Defaults to 1.
        B (float, optional): The vertical shift of the sine wave in the concurrency function. Defaults to 1.
        sigma (float, optional): The standard deviation of the normal distribution for the noise. Defaults to 0.1.
        omega (float, optional): if None, omega = pi / (2 * length / period)
        period (float, optional): See omega. Defaults to 0.25.
        only_rise: if True, the concurrency will monotonically increase
        length (int, optional): if None, length = duration_ms / interval_ms
        duration_ms (int, optional): See param: length
        interval_ms (int, optional): See param: length

    Returns:
        list: A list of items, where each item is a list of requests to be sent concurrently.
    """


    assert duration_ms is not None and interval_ms is not None, \
        "duration_ms and interval_ms must be specified."
    num_intervals = int(duration_ms // interval_ms) + 1
    workload = []
    interval = 0
    previous_rate = -1
    previous_input_len = None
    previous_output_len = None
    ts = 0
    
    rps_dist = []
    input_token_len_dist = []
    output_token_len_dist = []
    while interval < num_intervals:
        current_rate, previous_rate = sine_fluctuation(interval, qps_pattern_config, num_intervals, previous_rate)
        current_input_len = None
        current_output_len = None
        if input_pattern_config:
            current_input_len, previous_input_len = sine_fluctuation(interval, input_pattern_config, num_intervals, previous_input_len) 
            current_input_len = current_input_len if current_input_len > 0 else 1
        if output_pattern_config:
            current_output_len, previous_output_len = sine_fluctuation(interval, output_pattern_config, num_intervals, previous_output_len)
            current_output_len = current_output_len if current_output_len > 0 else 1
        rps_dist.append(current_rate)
        input_token_len_dist.append(current_input_len)
        output_token_len_dist.append(current_output_len)
        ts += interval_ms
        interval += 1
        
    workload = generate_synthetic_from_dist(
        prompt_file_path = prompt_file_path,
        tokenizer = tokenizer,
        duration_ms =  duration_ms,
        rps_dist = rps_dist,
        input_token_len_dist = input_token_len_dist,
        output_token_len_dist = output_token_len_dist,
        qps_scale = 1.0,
        input_scale = 1.0,
        output_scale = 1.0,
        max_concurrent_sessions = max_concurrent_sessions,
    )
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload


def generate_from_azure_csv(file_path: str,
                            prompt_file_path: str,
                            duration_ms: int,
                            tokenizer: PreTrainedTokenizerBase,
                            interval_ms: int,
                            output_file: str = 'output/output',
                            to_jsonl: bool = False,
                            ) -> List[List[Any]]:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Ensure TIMESTAMP is a datetime object
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])

    # Define the grouping time range (e.g., 1 second)
    time_range = timedelta(milliseconds=interval_ms)

    # Initialize a list to hold the grouped requests
    grouped_requests = []

    # Group requests by the time range
    df.set_index('TIMESTAMP', inplace=True)
    current_time = df.index.min()
    tracing_file_end_time = df.index.max()
    end_time = current_time + Timedelta(milliseconds=duration_ms)
    if tracing_file_end_time < end_time:
        logging.warning(f"{tracing_file_end_time} can not cover duration {duration_ms}, cap to end time of tracing file")
        end_time = tracing_file_end_time

    logging.info(f"Start generation from time {current_time} to {end_time}")
    sharegpt_df = load_requests(dataset_path=prompt_file_path, tokenizer=tokenizer)

    ts = 0
    request_finder = RequestFinder(df=sharegpt_df)
    while current_time <= end_time:
        # Select requests within the current time range
        mask = (df.index >= current_time) & (df.index < current_time + time_range)
        group = df.loc[mask]
        input_lens = []
        output_lens = []
        for _, row in group.iterrows():
            input_lens.append(int(row['ContextTokens']))
            output_lens.append(int(row['GeneratedTokens']))
        sampled_requests = request_finder.find_requests_len_range(
            num_requests=len(input_lens),
            input_lens=input_lens,
            output_lens=output_lens,
            initial_err_perc=0.1,
            err_step=0.05,
        )

        if sampled_requests:  # Only add non-empty groups
            grouped_requests.append({"timestamp": ts, "requests": sampled_requests})
        ts += interval_ms
        if ts > duration_ms:
            break
        # Move to the next time range
        current_time += time_range

    # Save to file
    grouped_requests = make_serializable(grouped_requests)
    save_workload(grouped_requests, output_file, use_jsonl=to_jsonl)

    return grouped_requests

def generate_from_mooncake_jsonl(file_path: str,
                            prompt_file_path: str,
                            duration_ms: int,
                            tokenizer: PreTrainedTokenizerBase,
                            output_file: str = 'output/output',
                            to_jsonl: bool = False,
                            ) -> List[List[Any]]:
    if prompt_file_path:
        raise ValueError(f"prompt_file_path can only be None")
    
    chunk_size = 512
    trace = load_jsonl(file_path)
    end_time = duration_ms
    id_to_chunks = {}
    grouped_requests = []
    prev_timestamp = -1
    current_group = None
    
    for entry in tqdm(trace, desc=f"Preparing prompts based on {file_path}"):
        timestamp = entry["timestamp"]
        if timestamp > end_time:
            break
            
        # Start a new group if timestamp changes
        if timestamp != prev_timestamp:
            if current_group is not None:
                grouped_requests.append(current_group)
            current_group = {
                "timestamp": timestamp,
                "requests": []
            }
            prev_timestamp = timestamp
            
        input_length = entry["input_length"]
        output_length = entry["output_length"]
        hash_ids = entry["hash_ids"]
        prompt_concat = ""
        
        for id in hash_ids:
            if id in id_to_chunks:
                prompt_concat += id_to_chunks[id]
            else:
                prompt_unique, _ = generate_synthetic_prompt(tokenizer=tokenizer, target_token_length=chunk_size, unique_prefix=str(id))
                id_to_chunks[id] = prompt_unique
                prompt_concat += prompt_unique
        prompt = adjust_prompt_length(tokenizer = tokenizer, prompt = prompt_concat, target_token_length=input_length)   
        current_group["requests"].append({
            "prompt": prompt,
            "prompt_length": len(tokenizer.encode(prompt_concat)),
            "output_length": output_length
        })
    
    # Add the last group if it exists
    if current_group is not None:
        grouped_requests.append(current_group)
        
    grouped_requests = make_serializable(grouped_requests)
    save_workload(grouped_requests, output_file, use_jsonl=to_jsonl)

    return grouped_requests


def generate_from_burstgpt_csv(file_path: str,
                              duration_ms: int,
                              output_file: str = 'output/output',
                              to_jsonl: bool = False,
                              user_count: Optional[int] = None,
                              user_mapping: str = "round_robin",
                              user_bucket_ms: int = 1000,
                              ) -> List[Dict[str, Any]]:
    """将 BurstGPT csv 直接转为 aibrix workload。

    - 不依赖 tokenizer（避免 CPU-only 场景下载/初始化大模型 tokenizer）
    - prompt 使用重复 token 近似（cl100k_base 下 `hello` 基本接近 1 token）
    - output_length 用于 client 侧 per-request max_tokens
    """

    def _safe_int(v):
        try:
            return int(float(v))
        except Exception:
            return 0

    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ts_ms = int(float(r.get("Timestamp", 0)) * 1000)
            in_tok = _safe_int(r.get("Request tokens", 0))
            out_tok = _safe_int(r.get("Response tokens", 0))
            if ts_ms <= 0 or in_tok <= 0 or out_tok <= 0:
                continue
            rows.append((ts_ms, in_tok, out_tok))

    if not rows:
        save_workload([], output_file, use_jsonl=to_jsonl)
        return []

    rows.sort(key=lambda x: x[0])
    base_ts = rows[0][0]

    def _assign_user(rel_ts: int, in_tok: int, out_tok: int, idx: int) -> Optional[str]:
        if user_count is None:
            return None
        try:
            n = int(user_count)
        except Exception:
            return None
        if n <= 0:
            return None

        m = (user_mapping or "round_robin").lower()
        if m == "round_robin":
            return f"user-{idx % n}"
        if m == "time_bucket":
            b = int(rel_ts // max(int(user_bucket_ms), 1))
            return f"user-{b % n}"
        if m == "hash":
            # 稳定 hash：同一份 trace/过滤规则下可复现
            import hashlib
            h = hashlib.md5(f"{rel_ts}:{in_tok}:{out_tok}:{idx}".encode("utf-8")).hexdigest()
            return f"user-{int(h[:8], 16) % n}"
        # fallback
        return f"user-{idx % n}"

    grouped = {}
    idx = 0
    for ts_ms, in_tok, out_tok in rows:
        rel_ts = ts_ms - base_ts
        if rel_ts < 0:
            continue
        if duration_ms is not None and rel_ts > duration_ms:
            break

        # 生成近似 token 数的 prompt（注意：过长会增加 HTTP 传输开销；但 1-2k token 级别是可接受的）
        prompt = ("hello " * in_tok).strip()
        req = {
            "prompt": prompt,
            "prompt_length": in_tok,
            "completion_length": out_tok,
            "output_length": out_tok,
        }
        u = _assign_user(rel_ts=rel_ts, in_tok=in_tok, out_tok=out_tok, idx=idx)
        if u is not None:
            req["user"] = u
        grouped.setdefault(rel_ts, []).append(req)
        idx += 1

    workload = [{"timestamp": ts, "requests": grouped[ts]} for ts in sorted(grouped.keys())]
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload


def generate_from_session_parquet(
    parquet_path: str,
    duration_ms: int,
    output_file: str = 'output/output',
    to_jsonl: bool = False,
    user_count: Optional[int] = None,
    user_mapping: str = "hash",
    user_bucket_ms: int = 1000,
    session_col: str = "conversation_hash",
    timestamp_col: str = "timestamp",
    conversation_col: str = "conversation",
    turn_col: Optional[str] = "turn",
    max_rows: Optional[int] = None,
    max_prompt_tokens: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """将带 session + timestamp 的 parquet 数据集转为 aibrix workload。

    - session_id: 使用 session_col（默认 conversation_hash）
    - timestamp: 使用 timestamp_col（默认 timestamp），并归一化为从 0ms 开始
    - prompt/output_length: 从 conversation 中抽取最后一轮 user/assistant，并估算 token 数

    说明：该转换不依赖 transformers tokenizer（CPU-only 可用）。token 估算优先用 tiktoken，
    否则退化到简单的 whitespace/字符近似。
    """

    def _list_parquet_files(p: str) -> List[str]:
        if os.path.isdir(p):
            return sorted(glob.glob(os.path.join(p, "*.parquet")))
        if any(ch in p for ch in ["*", "?", "["]):
            return sorted(glob.glob(p))
        return [p]

    def _estimate_tokens(text: Optional[str]) -> int:
        if not text:
            return 0
        try:
            import tiktoken  # type: ignore

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            s = str(text).strip()
            if not s:
                return 0
            ws = s.split()
            if len(ws) >= 2:
                return len(ws)
            # 对中文/无空格文本，按字符粗略估算
            return max(1, int(len(s) / 2))

    def _truncate_by_token_estimate(text: str, target_tokens: int) -> str:
        cur = _estimate_tokens(text)
        if cur <= 0 or cur <= target_tokens:
            return text
        ratio = float(target_tokens) / float(cur)
        new_chars = max(1, int(len(text) * ratio))
        return text[:new_chars]

    def _extract_last_turn(conv_obj) -> (Optional[str], Optional[str]):
        if conv_obj is None:
            return None, None
        try:
            conv = list(conv_obj)
        except Exception:
            return None, None

        last_user = None
        last_assistant = None

        # 优先找最后一个 assistant，并取其前最近的 user
        for i in range(len(conv) - 1, -1, -1):
            msg = conv[i]
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant = str(msg.get("content"))
                for j in range(i - 1, -1, -1):
                    m2 = conv[j]
                    if isinstance(m2, dict) and m2.get("role") == "user" and m2.get("content"):
                        last_user = str(m2.get("content"))
                        break
                break

        if last_user is None:
            for i in range(len(conv) - 1, -1, -1):
                msg = conv[i]
                if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                    last_user = str(msg.get("content"))
                    break

        return last_user, last_assistant

    def _assign_user(idx: int, rel_ts: int, session_id: str) -> Optional[str]:
        if user_count is None:
            return None
        try:
            n = int(user_count)
        except Exception:
            return None
        if n <= 0:
            return None

        m = (user_mapping or "hash").lower()
        if m == "round_robin":
            return f"user-{idx % n}"
        if m == "time_bucket":
            b = int(rel_ts // max(int(user_bucket_ms), 1))
            return f"user-{b % n}"

        sid = session_id or str(idx)
        h = hashlib.md5(sid.encode("utf-8")).hexdigest()
        return f"user-{int(h[:8], 16) % n}"

    files = _list_parquet_files(parquet_path)
    if not files:
        raise ValueError(f"No parquet files found under: {parquet_path}")

    frames = []
    for fpath in files:
        frames.append(pd.read_parquet(fpath))
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        save_workload([], output_file, use_jsonl=to_jsonl)
        return []

    required = [session_col, timestamp_col, conversation_col]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in parquet. columns={list(df.columns)}")

    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col, ascending=True)
    if max_rows is not None:
        try:
            df = df.head(int(max_rows))
        except Exception:
            pass

    if df.empty:
        save_workload([], output_file, use_jsonl=to_jsonl)
        return []

    base_ts = df[timestamp_col].iloc[0]
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    idx = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing session_parquet workload"):
        ts = row.get(timestamp_col)
        if ts is None or pd.isna(ts):
            continue
        try:
            rel_ms = int((ts - base_ts).total_seconds() * 1000)
        except Exception:
            continue
        if rel_ms < 0:
            continue
        if duration_ms is not None and rel_ms > duration_ms:
            break

        session_id = row.get(session_col)
        session_id = str(session_id) if session_id is not None else ""
        prompt_text, assistant_text = _extract_last_turn(row.get(conversation_col))
        if not prompt_text:
            continue

        ptxt = str(prompt_text)
        if max_prompt_tokens is not None:
            ptxt = _truncate_by_token_estimate(ptxt, int(max_prompt_tokens))
        prompt_len = _estimate_tokens(ptxt)

        out_len = _estimate_tokens(str(assistant_text) if assistant_text is not None else "")
        if out_len <= 0:
            out_len = 1
        if max_output_tokens is not None:
            try:
                out_len = min(int(out_len), int(max_output_tokens))
            except Exception:
                pass

        req: Dict[str, Any] = {
            "prompt": ptxt,
            "prompt_length": int(prompt_len),
            "completion_length": int(out_len),
            "output_length": int(out_len),
        }
        if session_id:
            req["session_id"] = session_id
            if turn_col and turn_col in df.columns:
                try:
                    req["turn"] = int(row.get(turn_col))
                except Exception:
                    pass

        u = _assign_user(idx=idx, rel_ts=rel_ms, session_id=session_id)
        if u is not None:
            req["user"] = u

        grouped.setdefault(rel_ms, []).append(req)
        idx += 1

    workload = [{"timestamp": ts, "requests": grouped[ts]} for ts in sorted(grouped.keys())]
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload

def generate_vtc_longtail_workload(
    duration_ms: int,
    output_file: str,
    to_jsonl: bool,
    mode: str,
    seed: int,
    user_count: int,
    heavy_user: int,
    heavy_req_ratio: float,
    burst_qps: int,
    interval_ms: int,
    short_prompt: int,
    short_output: int,
    long_prompt: int,
    long_output: int,
    super_long_prompt: int,
    super_long_output: int,
) -> List[Dict[str, Any]]:
    """专门为对比 vtc-basic vs vtc-pred 设计的 workload。

    设计目标：
    - 用户维度：一个 heavy user + 多个 normal user
    - token 长度：output length 强长尾（short/long/super-long），并与 prompt length 强相关（方便 vtc-pred 的 OutputPredictor 学习）
    - 到达过程：burst（同一秒内多个请求）以制造排队/HoL，放大路由差异

    注意：这里构造的是 workload（请求侧 trace），不是服务端真实 trace。
    """

    rng = np.random.default_rng(seed)
    mode = (mode or "main").lower()
    if mode not in ("warmup", "main"):
        raise ValueError("--vtc-mode must be one of: warmup, main")

    if duration_ms <= 0:
        save_workload([], output_file, use_jsonl=to_jsonl)
        return []

    if user_count <= 0:
        raise ValueError("--vtc-user-count must be > 0")

    def _mk_prompt(tok: int) -> str:
        # 用 whitespace token 近似，配合 mock 服务端的 token fallback：len(split())
        return ("hi " * int(tok)).strip()

    def _user(i: int) -> str:
        return f"user-{i}"

    heavy_user = int(max(0, min(user_count - 1, heavy_user)))

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    req_idx = 0

    # Warmup：覆盖多个 prompt bucket，训练 output predictor（output ≈ prompt）
    if mode == "warmup":
        warmup_total = min(400, max(50, int(duration_ms / 50)))  # duration 越长 warmup 越多，但上限 400
        prompt_candidates = [
            max(8, short_prompt),
            max(16, short_prompt * 2),
            max(32, long_prompt // 4),
            max(64, long_prompt // 2),
            max(128, long_prompt),
            max(256, super_long_prompt),
        ]
        prompt_candidates = sorted(set(int(x) for x in prompt_candidates))

        for k in range(warmup_total):
            ts = int((k / max(warmup_total - 1, 1)) * (duration_ms - 1))
            u = _user(k % user_count)
            p = int(rng.choice(prompt_candidates))
            # output 与 prompt 强相关（带少量抖动）
            out = int(max(1, p + int(rng.normal(0, max(1, p * 0.05)))))
            grouped.setdefault(ts, []).append({
                "prompt": _mk_prompt(p),
                "prompt_length": p,
                "completion_length": out,
                "output_length": out,
                "user": u,
            })
            req_idx += 1

    # Main：一个 heavy user 产生长尾输出；其他 user 主要是短请求。
    if mode == "main":
        if interval_ms <= 0:
            interval_ms = 1000
        ticks = int(np.ceil(duration_ms / interval_ms))
        heavy_prob = float(np.clip(heavy_req_ratio, 0.0, 1.0))
        for t in range(ticks):
            ts = t * interval_ms
            # 每个 tick 发 burst_qps 个请求
            for _ in range(max(1, burst_qps)):
                is_heavy = (rng.random() < heavy_prob)
                if is_heavy:
                    u = _user(heavy_user)
                    # heavy user 的 output 长尾：short/long/super-long
                    r = rng.random()
                    if r < 0.70:
                        p, out = long_prompt, long_output
                    elif r < 0.90:
                        p, out = super_long_prompt, super_long_output
                    else:
                        p, out = max(64, long_prompt // 2), max(64, long_output // 2)
                else:
                    # normal user：均匀分布在其余用户
                    candidates = [i for i in range(user_count) if i != heavy_user]
                    u = _user(int(rng.choice(candidates)) if candidates else heavy_user)
                    r = rng.random()
                    if r < 0.92:
                        p, out = short_prompt, short_output
                    else:
                        p, out = max(32, short_prompt * 4), max(32, short_output * 4)

                grouped.setdefault(ts, []).append({
                    "prompt": _mk_prompt(p),
                    "prompt_length": int(p),
                    "completion_length": int(out),
                    "output_length": int(out),
                    "user": u,
                })
                req_idx += 1

    workload = [{"timestamp": ts, "requests": grouped[ts]} for ts in sorted(grouped.keys())]
    workload = make_serializable(workload)
    save_workload(workload, output_file, use_jsonl=to_jsonl)
    return workload


def main(args):
    # Generate workloads and pair with prompts
    workload_dict = {}

    # burstgpt 不依赖 tokenizer（避免 CPU-only 场景下载/初始化大模型 tokenizer）
    tokenizer = None
    if args.trace_type not in ("burstgpt", "vtc_longtail", "session_parquet"):
        tokenizer = get_tokenizer(pretrained_model_name_or_path=args.tokenizer, trust_remote_code=True)

    if args.trace_type == "synthetic":
        qps_pattern_config = None
        input_pattern_config = None
        output_pattern_config = None
        comp_pattern_type = f"synthetic_manual_config"
        if args.traffic_pattern:
            qps_pattern_config = to_fluctuate_pattern_config(config_type = args.traffic_pattern, mean = 6)
        elif args.traffic_pattern_config:
            qps_pattern_config = user_to_synthetic_config(user_config = load_json(args.traffic_pattern_config), duration_ms = args.duration_ms)
            
        if args.prompt_len_pattern:
            input_pattern_config = to_fluctuate_pattern_config(config_type = args.prompt_len_pattern, mean = 1024)
        elif args.prompt_len_pattern_config:
            input_pattern_config = user_to_synthetic_config(user_config = load_json(args.prompt_len_pattern_config), duration_ms = args.duration_ms)
            
        if args.completion_len_pattern:
            output_pattern_config = to_fluctuate_pattern_config(config_type = args.completion_len_pattern, mean = 1024)
        elif args.completion_len_pattern_config:
            output_pattern_config = user_to_synthetic_config(user_config = load_json(args.completion_len_pattern_config), duration_ms = args.duration_ms)
        
        if qps_pattern_config is None:
            raise ValueError(f"qps_pattern_config cannot be None")
        
        generated_workload = generate_synthetic(prompt_file_path = args.prompt_file,
                                                tokenizer=tokenizer,
                                                qps_pattern_config = qps_pattern_config,
                                                input_pattern_config = input_pattern_config,
                                                output_pattern_config = output_pattern_config,
                                                duration_ms=args.duration_ms,
                                                interval_ms=args.interval_ms,
                                                max_concurrent_sessions=args.max_concurrent_sessions,
                                                output_file=f"{args.output_dir}/workload",
                                                to_jsonl=(args.output_format == "jsonl"),
                                            )
        workload_dict[comp_pattern_type] = generated_workload
    else:
        # Process for 'stat' and 'azure'
        if args.trace_type == "constant":
            generated_workload = generate_constant(prompt_file_path=args.prompt_file, 
                                                   tokenizer=tokenizer,
                                                    qps=args.target_qps,
                                                    input_len=args.target_prompt_len,
                                                    output_len=args.target_completion_len,
                                                    duration_ms=args.duration_ms, 
                                                    interval_ms=args.interval_ms,
                                                    max_concurrent_sessions=args.max_concurrent_sessions,
                                                    output_file=f"{args.output_dir}/workload",
                                                    to_jsonl=(args.output_format == "jsonl"),
                                                )
        elif args.trace_type == "stat":
            generated_workload = generate_from_stat_csv(prompt_file_path=args.prompt_file, 
                                                            duration_ms=args.duration_ms, 
                                                            tokenizer=tokenizer,
                                                            qps_stat=args.traffic_file, 
                                                            input_stat=args.prompt_len_file, 
                                                            output_stat=args.completion_len_file,
                                                            qps_scale=args.qps_scale,
                                                            input_scale=args.input_scale,
                                                            output_scale=args.output_scale,
                                                            stat_trace_type=args.stat_trace_type,
                                                            max_concurrent_sessions=args.max_concurrent_sessions,
                                                            output_file=f"{args.output_dir}/workload",
                                                            to_jsonl=(args.output_format == "jsonl"),
                                                            )

        elif args.trace_type == "azure":
            generated_workload = generate_from_azure_csv(file_path=args.traffic_file, 
                                                         prompt_file_path=args.prompt_file,
                                                         duration_ms=args.duration_ms, 
                                                         tokenizer=tokenizer,
                                                         interval_ms=args.interval_ms, 
                                                         output_file=f"{args.output_dir}/workload",
                                                         to_jsonl=(args.output_format == "jsonl"),
                                                         )

        elif args.trace_type == "mooncake":
            generated_workload = generate_from_mooncake_jsonl(file_path=args.traffic_file, 
                                                              prompt_file_path=args.prompt_file,
                                                              duration_ms=args.duration_ms, 
                                                              tokenizer=tokenizer,
                                                              output_file=f"{args.output_dir}/workload",
                                                              to_jsonl=(args.output_format == "jsonl"),
                                                              )

        elif args.trace_type == "burstgpt":
            generated_workload = generate_from_burstgpt_csv(
                file_path=args.traffic_file,
                duration_ms=args.duration_ms,
                output_file=f"{args.output_dir}/workload",
                to_jsonl=(args.output_format == "jsonl"),
                user_count=args.user_count,
                user_mapping=args.user_mapping,
                user_bucket_ms=args.user_bucket_ms,
            )

        elif args.trace_type == "session_parquet":
            generated_workload = generate_from_session_parquet(
                parquet_path=args.traffic_file,
                duration_ms=args.duration_ms,
                output_file=f"{args.output_dir}/workload",
                to_jsonl=(args.output_format == "jsonl"),
                user_count=args.user_count,
                user_mapping=args.user_mapping,
                user_bucket_ms=args.user_bucket_ms,
                session_col=args.session_col,
                timestamp_col=args.timestamp_col,
                conversation_col=args.conversation_col,
                turn_col=args.turn_col,
                max_rows=args.max_rows,
                max_prompt_tokens=args.max_prompt_tokens,
                max_output_tokens=args.max_output_tokens,
            )

        elif args.trace_type == "vtc_longtail":
            generated_workload = generate_vtc_longtail_workload(
                duration_ms=args.duration_ms,
                output_file=f"{args.output_dir}/workload",
                to_jsonl=(args.output_format == "jsonl"),
                mode=args.vtc_mode,
                seed=args.vtc_seed,
                user_count=args.vtc_user_count,
                heavy_user=args.vtc_heavy_user,
                heavy_req_ratio=args.vtc_heavy_req_ratio,
                burst_qps=args.vtc_burst_qps,
                interval_ms=args.interval_ms,
                short_prompt=args.vtc_short_prompt,
                short_output=args.vtc_short_output,
                long_prompt=args.vtc_long_prompt,
                long_output=args.vtc_long_output,
                super_long_prompt=args.vtc_super_long_prompt,
                super_long_output=args.vtc_super_long_output,
            )
        
        workload_dict[args.trace_type] = generated_workload

    if workload_dict:
        # Plot the workloads
        for workload_name, workload in workload_dict.items():
            plot_workload(
                workload = workload, 
                bin_size_sec = 1, 
                output_dir = f"{args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Workload Generator')
    parser.add_argument('--trace-type', type=str, required=True, choices=['constant','synthetic', 'stat', 'azure', 'mooncake', 'burstgpt', 'vtc_longtail', 'session_parquet'],
                        help='Type of trace consumed. Choose among: synthetic, stat, azure.')
    parser.add_argument('--tokenizer', type=str, required=False, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help='Target model for the workload.')
    parser.add_argument('--prompt-file', type=str, required=False, default = None, help='File containing sampling prompts.')
    parser.add_argument('--interval-ms', type=int, required=False, default=1000,
                        help='Granularity of request injection interval in milliseconds.')
    parser.add_argument('--duration-ms', type=int, default=60000, help='Duration of the trace generated.')
    parser.add_argument('--group-interval-seconds', type=int, default=1, help='Grouping interval seconds.')
    parser.add_argument('--stat-trace-type', type=str, choices=['maas', 'cloudide'], default="maas", help='Type of stat traces.')
    parser.add_argument('--output-dir', type=str, required=False, default="output", help='Output directory to save.'
                                                                                         'the workload.')
    parser.add_argument('--output-format', type=str, choices=['json', 'jsonl'], default='jsonl',
                        help='Set output data format to either .json or .jsonl (default is .json).')
    
    ###### Synthetic and constant workload
    parser.add_argument('--target-qps', type=int, required=False, default=1, help='Target QPS for the workload.')
    parser.add_argument('--target-prompt-len', type=int, required=False, default=None, help='Target prompt length for the workload.')
    parser.add_argument('--target-completion-len', type=int, required=False, default=None, help='Target completion length for the workload.')
    parser.add_argument('--traffic-pattern', type=str, required=False, choices=['quick_rising', 'slow_rising', 'slight_fluctuation', 'severe_fluctuation'], default=None,
                        help='Traffic patterns used for synthetic workload type.')
    parser.add_argument('--prompt-len-pattern', type=str, required=False, choices=['quick_rising', 'slow_rising', 'slight_fluctuation', 'severe_fluctuation'], default=None,
                        help='Prompt lengths patterns used for synthetic workload type.')
    parser.add_argument('--completion-len-pattern', type=str, required=False, choices=['quick_rising', 'slow_rising', 'slight_fluctuation', 'severe_fluctuation'], default=None,
                        help='Prompt lengths patterns used for synthetic workload type.')
    parser.add_argument('--traffic-pattern-config', type=str, required=False, default=None,
                        help='Traffic configuration file used for synthetic workload type.')
    parser.add_argument('--prompt-len-pattern-config', type=str, required=False, default=None,
                        help='Prompt lengths configuration file used for synthetic workload type.')
    parser.add_argument('--completion-len-pattern-config', type=str, required=False, default=None,
                        help='Completion lengths configuration file used for synthetic workload type.')
    
    ##### Trace and stats-driven workload
    parser.add_argument('--traffic-file', type=str, required=False, default=None,
                        help='Traffic file containing times of arrival, which workload generator depends upon to'
                             'convert to traffic used in workload. This is only needed for for stat and azure trace type.')
    parser.add_argument('--prompt-len-file', type=str, required=False, default=None,
                        help='File containing request input lengths varied by time, which workload generator depends upon to '
                             'select input prompt. This is only needed for for stat trace type. ')
    parser.add_argument('--completion-len-file', type=str, required=False, default=None,
                        help='File containing request output lengths varied by time, which workload generator depends upon to '
                             'select input prompt. This is only needed for for stat trace type. ')
    parser.add_argument('--qps-scale', type=float, required=False, default=1.0, help='QPS scaling factor.')
    parser.add_argument('--input-scale', type=float, required=False, default=1.0, help='Input length scaling factor.')
    parser.add_argument('--output-scale', type=float, required=False, default=1.0, help='Output length scaling factor.')
    parser.add_argument('--max-concurrent-sessions', type=int, required=False, default=1, help='Maximum number of overlapping sessions.')

    ##### BurstGPT user mapping (for fairness analysis)
    parser.add_argument('--user-count', type=int, required=False, default=None,
                        help='给 BurstGPT workload 注入 user 字段的用户数；为空则不写入 user。')
    parser.add_argument('--user-mapping', type=str, required=False, default='round_robin',
                        choices=['round_robin', 'hash', 'time_bucket'],
                        help='BurstGPT workload 的 user 划分规则：round_robin(默认)/hash/time_bucket。')
    parser.add_argument('--user-bucket-ms', type=int, required=False, default=1000,
                        help='user-mapping=time_bucket 时的 bucket 大小（ms），同 bucket 内请求映射到同一 user。')

    ##### Sessioned parquet workload
    parser.add_argument('--session-col', type=str, required=False, default='conversation_hash',
                        help='session_parquet: session id column name')
    parser.add_argument('--timestamp-col', type=str, required=False, default='timestamp',
                        help='session_parquet: timestamp column name')
    parser.add_argument('--conversation-col', type=str, required=False, default='conversation',
                        help='session_parquet: conversation column name')
    parser.add_argument('--turn-col', type=str, required=False, default='turn',
                        help='session_parquet: turn column name (optional)')
    parser.add_argument('--max-rows', type=int, required=False, default=None,
                        help='session_parquet: limit rows after sorting by timestamp')
    parser.add_argument('--max-prompt-tokens', type=int, required=False, default=None,
                        help='session_parquet: truncate prompt by estimated token count (omit to disable truncation)')
    parser.add_argument('--max-output-tokens', type=int, required=False, default=None,
                        help='session_parquet: cap output_length per request (omit to disable cap)')

    ##### VTC longtail workload (for vtc-basic vs vtc-pred comparison)
    parser.add_argument('--vtc-mode', type=str, required=False, default='main', choices=['warmup', 'main'],
                        help='vtc_longtail 的生成模式：warmup(训练 predictor)/main(长尾+用户偏斜)。')
    parser.add_argument('--vtc-seed', type=int, required=False, default=42, help='vtc_longtail 随机种子。')
    parser.add_argument('--vtc-user-count', type=int, required=False, default=10, help='vtc_longtail 用户数。')
    parser.add_argument('--vtc-heavy-user', type=int, required=False, default=0, help='heavy user 的编号（0..user_count-1）。')
    parser.add_argument('--vtc-heavy-req-ratio', type=float, required=False, default=0.35,
                        help='main 阶段 heavy user 占请求比例（0~1）。')
    parser.add_argument('--vtc-burst-qps', type=int, required=False, default=8,
                        help='main 阶段每个 interval_ms 内注入请求数（越大越容易排队）。')
    parser.add_argument('--vtc-short-prompt', type=int, required=False, default=64)
    parser.add_argument('--vtc-short-output', type=int, required=False, default=64)
    parser.add_argument('--vtc-long-prompt', type=int, required=False, default=1024)
    parser.add_argument('--vtc-long-output', type=int, required=False, default=1024)
    parser.add_argument('--vtc-super-long-prompt', type=int, required=False, default=2048)
    parser.add_argument('--vtc-super-long-output', type=int, required=False, default=2048)
    
    args = parser.parse_args()
    main(args)
