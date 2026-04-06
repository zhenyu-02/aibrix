import argparse
import logging
import time
import asyncio
import openai
import json
import io
import traceback
import threading
import os
import sys
import csv
import math


from typing import List, Dict, Callable, Optional, Set, Tuple

# Allow running this file directly from the repo root (without setting PYTHONPATH).
# The `client` package lives under `benchmarks/`, so we add that directory.
_BENCHMARKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BENCHMARKS_DIR not in sys.path:
    sys.path.insert(0, _BENCHMARKS_DIR)

from client.utils import (
    load_workload,
    prepare_prompt,
    update_response,
    create_client,
    pick_user_for_request,
)

logging.basicConfig(level=logging.INFO)
# Reduce third-party noisy logs (per-request HTTP lines).
logging.getLogger("httpx").setLevel(logging.WARNING)
session_history: Dict[str, List[Dict]] = {}
session_history_lock = threading.Lock()  # Use threading lock for thread safety
pending_sessioned_requests: Dict[str, List[Tuple[Dict, float]]] = {}
# 注意：asyncio.Queue 需要绑定到运行中的 event loop。
# 这里先占位，在 benchmark_launch() 内创建，避免 "attached to a different loop"。
completed_sessions = None

async def send_request_streaming(client: openai.AsyncOpenAI,
                             model: str,
                             max_output: int, 
                             request: Dict,
                             output_file: str,
                             request_id: int,
                             session_id: int,
                             target_time: int,
                             user_count: int,
                             ):
    session_id = request.get("session_id", None)
    prompt = prepare_prompt(
        prompt = request["prompt"], 
        session_id = request.get("session_id", None), 
        history = None if session_id is None else session_history,
        history_lock = None if session_id is None else session_history_lock) 
    start_time = time.time()
    first_response_time = None
    target_pod = ""
    target_request_id = ""
    user = pick_user_for_request(request, request_id, user_count)
    try:
        # This is extremely verbose for large workloads; keep it at debug level.
        logging.debug(
            f"send_request_streaming: Prepare to launch task after {target_time - start_time} target_time {target_time} start_time {start_time}"
        )
        if target_time > start_time:
            await asyncio.sleep(target_time - start_time)
        dispatch_time = asyncio.get_event_loop().time()
        # 支持 per-request output_length（例如 BurstGPT trace 驱动）
        req_max_tokens = max_output
        if request is not None and request.get("output_length") is not None:
            try:
                req_max_tokens = int(request.get("output_length"))
                if max_output is not None:
                    req_max_tokens = min(req_max_tokens, int(max_output))
            except Exception:
                req_max_tokens = max_output

        # Do NOT send `model` as a request header.
        # The gateway-plugin may pre-route in HandleRequestHeaders when `model` header is present,
        # which happens before the request body is parsed. That makes VTC variants see an empty
        # `RoutingContext.Message` and thus token features become 0, hiding vtc-pred benefits.
        request_client = client.with_options(default_headers={"user": user})
        response_stream = await request_client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=req_max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        if hasattr(response_stream, 'response') and hasattr(response_stream.response, 'headers'):
            target_pod = response_stream.response.headers.get('target-pod')
            target_request_id = response_stream.response.headers.get('request-id')

        text_chunks = []
        prompt_tokens = 0
        output_tokens = 0
        total_tokens = 0

        try:
            async for chunk in response_stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    output_text = delta.content
                    if output_text is None:
                        # Use getattr for safety as reasoning_content is not a standard field
                        output_text = getattr(delta, 'reasoning_content', None)
                    if output_text is not None:
                        if not first_response_time:
                            first_response_time = asyncio.get_event_loop().time()
                        text_chunks.append(output_text)
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    # For OpenAI, we expect to get complete usage stats, not partial ones to accumulate
                    # So we can safely overwrite previous values if they exist
                    if chunk.usage.prompt_tokens is not None:
                        prompt_tokens = chunk.usage.prompt_tokens
                    if chunk.usage.completion_tokens is not None:
                        output_tokens = chunk.usage.completion_tokens
                    if chunk.usage.total_tokens is not None:
                        total_tokens = chunk.usage.total_tokens
        except Exception as stream_error:
            # Handle errors during streaming
            logging.error(f"Request {request_id}: Stream interrupted: {type(stream_error).__name__}: {str(stream_error)}")

        response_text = "".join(text_chunks)
        response_time = asyncio.get_event_loop().time()
        latency = response_time - dispatch_time
        throughput = output_tokens / latency if output_tokens > 0 else 0
        ttft = first_response_time - dispatch_time if first_response_time else None
        tpot = (response_time - first_response_time) / output_tokens if first_response_time and output_tokens > 0 else None

        if session_id is not None:
            update_response(
                response = response_text, 
                session_id = session_id, 
                history = session_history,
                history_lock = session_history_lock,
            )
            try:
                await completed_sessions.put(session_id)
            except Exception as e:
                logging.error(f"Failed to signal session completion: {e}")
        
        result = {
            "request_id": request_id,
            "status": "success",
            "user": user,
            "input": prompt,
            "output": response_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput,
            "start_time": dispatch_time,
            "end_time": response_time,
            "ttft": ttft,
            "tpot": tpot,
            "target_pod": target_pod,
            "target_request_id": target_request_id,
            "session_id": session_id,
        }

        # Write result to JSONL file
        logging.debug(f"Request {request_id}: Completed successfully. Tokens: {total_tokens}, Latency: {latency:.2f}s")
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "user": user,
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "output": "",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency": error_time - dispatch_time,
            "throughput": 0,
            "start_time": dispatch_time,
            "end_time": error_time,
            "ttft": None,
            "tpot": None,
            "target_pod": target_pod,
            "target_request_id": target_request_id,
            "session_id": session_id,
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        if session_id is not None:
            await completed_sessions.put(session_id)
        return error_result

# Asynchronous request handler
async def send_request_batch(client: openai.AsyncOpenAI,
                             model: str,
                             max_output: int, 
                             request: Dict,
                             output_file: str,
                             request_id: int,
                             session_id: int, 
                             target_time: int,
                             user_count: int,
                             ):
    session_id = request.get("session_id", None)
    prompt = prepare_prompt(
        prompt = request["prompt"], 
        session_id = request.get("session_id", None), 
        history = None if session_id is None else session_history,
        history_lock = None if session_id is None else session_history_lock) 
    start_time = time.time()
    target_pod = ""
    user = pick_user_for_request(request, request_id, user_count)
    try:
        logging.debug(
            f"send_request_batch: Prepare to launch task after {target_time - start_time} target_time {target_time} start_time {start_time}"
        )
        if target_time > start_time:
            await asyncio.sleep(target_time - start_time)
        dispatch_time = asyncio.get_event_loop().time()
        # 支持 per-request output_length（例如 BurstGPT trace 驱动）
        req_max_tokens = max_output
        if request is not None and request.get("output_length") is not None:
            try:
                req_max_tokens = int(request.get("output_length"))
                if max_output is not None:
                    req_max_tokens = min(req_max_tokens, int(max_output))
            except Exception:
                req_max_tokens = max_output

        # Keep behavior consistent with streaming path (see comment above).
        request_client = client.with_options(default_headers={"user": user})
        response = await request_client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=0,
            max_tokens=req_max_tokens,
        )
        if hasattr(response, 'response') and hasattr(response.response, 'headers'):
            target_pod = response.response.headers.get('target-pod')

        response_time = asyncio.get_event_loop().time()
        latency = response_time - dispatch_time
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        throughput = output_tokens / latency
        output_text = response.choices[0].message.content

        if session_id is not None:
            update_response(
                response = output_text, 
                session_id = session_id, 
                history = session_history,
                history_lock = session_history_lock,
            )
            await completed_sessions.put(session_id)
        
        result = {
            "request_id": request_id,
            "status": "success",
            "user": user,
            "input": prompt,
            "output": output_text,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": latency,
            "throughput": throughput,
            "start_time": dispatch_time,
            "end_time": response_time,
            "ttft": None,
            "tpot": None,
            "target_pod": target_pod,
            "session_id": session_id,
        }
        logging.info(result)
        # Write result to JSONL file
        output_file.write(json.dumps(result) + "\n")
        output_file.flush()  # Ensure data is written immediately to the file
        return result

    except Exception as e:
        error_time = asyncio.get_event_loop().time()
        error_type = type(e).__name__
        error_result = {
            "request_id": request_id,
            "status": "error",
            "user": user,
            "error_type": error_type,
            "error_message": str(e),
            "error_traceback": traceback.format_exc(),
            "input": prompt,
            "output": "",
            "prompt_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency": error_time - dispatch_time,
            "throughput": 0,
            "start_time": dispatch_time,
            "end_time": error_time,
            "ttft": None,
            "tpot": None,
            "target_pod": target_pod,
            "session_id": session_id,
        }
        logging.error(f"Request {request_id}: Error ({error_type}): {str(e)}")
        output_file.write(json.dumps(error_result) + "\n")
        output_file.flush()
        if session_id is not None:
            await completed_sessions.put(session_id)
        return error_result

async def benchmark_launch(
    api_key: str,
    endpoint: str,
    max_retries: int,
    scale_factor: float,
    timeout: float,
    routing_strategy: str,
    load_struct: List[Dict],
    output_file: io.TextIOWrapper,
    model: str,
    max_output: int,
    send_request_func: Callable,
    duration_limit: Optional[float] = None,
    max_concurrent_sessions: Optional[int] = None,
    user_count: int = 10,
) -> None:

    global completed_sessions, pending_sessioned_requests, session_history
    # Reset per-run global states (important for sessioned workloads)
    pending_sessioned_requests.clear()
    session_history.clear()
    completed_sessions = asyncio.Queue()

    request_id = 0
    base_time = time.time()
    num_requests = 0
    tasks: List[asyncio.Task] = []
    client = create_client(api_key, endpoint, max_retries, timeout, routing_strategy)

    try:
        # Track active sessions for max_concurrent_sessions limit
        active_sessions = set()

        if max_concurrent_sessions is not None:
            logging.info(f"Max concurrent sessions limit: {max_concurrent_sessions}")

        # Set workload duration based on duration_limit parameter
        # If duration_limit is None, don't set a time limit (wait for all tasks)
        if duration_limit is None:
            workload_duration = None
            logging.info("No duration limit set. Benchmark will wait for all tasks to complete.")
        else:
            workload_duration = duration_limit
            logging.info(f"Duration limit set to {workload_duration:.1f}s")

        def send(request):
            nonlocal request_id, num_requests
            task = asyncio.create_task(
                send_request_func(
                    client=client,
                    model=model,
                    max_output=max_output,
                    request=request,
                    output_file=output_file,
                    request_id=request_id,
                    session_id=request.get("session_id", None) if "session_id" in request else None,
                    target_time=target_time,
                    user_count=user_count,
                )
            )
            request_id += 1
            num_requests += 1
            tasks.append(task)
        initiated_sessions = set()
        pending_new_sessions = []  # Queue for sessions that couldn't start due to capacity limit

        async def start_session_if_capacity_available(request, session_id):
            """Start a new session if capacity is available."""
            if session_id is not None:
                active_sessions.add(session_id)
                logging.info(f"Starting session {session_id}. Active sessions: {len(active_sessions)}/{max_concurrent_sessions}")
            send(request)
            initiated_sessions.add(session_id)

        for requests_dict in load_struct:
            ts = int(requests_dict["timestamp"] * scale_factor)
            requests = requests_dict["requests"]
            target_time = base_time + ts / 1000.0
            for i in range(len(requests)):
                session_id = requests[i].get("session_id", None) if "session_id" in requests[0] else None
                if session_id is None or session_id not in initiated_sessions:
                    # Check if we can start a new session without blocking
                    if max_concurrent_sessions is None or len(active_sessions) < max_concurrent_sessions:
                        await start_session_if_capacity_available(requests[i], session_id)
                    else:
                        # Can't start now, add to pending new sessions queue with timing info
                        logging.info(f"Session {session_id} cannot start yet (capacity full). Adding first request to pending.")
                        pending_new_sessions.append((requests[i], target_time))
                        initiated_sessions.add(session_id)  # Mark as initiated so future requests go to pending_sessioned_requests
                else:
                    logging.info(f"Adding request for session {session_id} to pending queue. Pending count: {len(pending_sessioned_requests.get(session_id, [])) + 1}")
                    pending_sessioned_requests.setdefault(session_id, []).append((requests[i], target_time))

        # Merge pending_new_sessions into pending_sessioned_requests
        for req, ttime in pending_new_sessions:
            sid = req.get("session_id")
            if sid is not None:
                pending_sessioned_requests.setdefault(sid, []).insert(0, (req, ttime))  # Insert at beginning since it's the first request

        logging.info(f"Finished processing all workload entries. Pending sessions: {len(pending_sessioned_requests)}, Pending requests: {sum(len(v) for v in pending_sessioned_requests.values())}")

        while len(pending_sessioned_requests) != 0:
            # Check if duration limit has been exceeded
            if workload_duration is not None:
                elapsed = time.time() - base_time
                if elapsed >= workload_duration:
                    logging.warning(f"Duration limit ({workload_duration:.1f}s) reached. Stopping session processing. {len(pending_sessioned_requests)} sessions remain pending.")
                    break

            logging.info(f"Waiting for session to complete. Pending sessions: {len(pending_sessioned_requests)}")
            done_session_id = await completed_sessions.get()
            logging.info(f"Session {done_session_id} signaled completion")

            if done_session_id in pending_sessioned_requests:
                next_request, target_time = pending_sessioned_requests[done_session_id].pop(0)
                send(next_request)
                if len(pending_sessioned_requests[done_session_id]) == 0:
                    pending_sessioned_requests.pop(done_session_id, None)
            else:
                # Session has no more pending requests, so it's truly complete
                if done_session_id in active_sessions:
                    active_sessions.remove(done_session_id)
                    logging.info(f"Session {done_session_id} completed. Active sessions: {len(active_sessions)}/{max_concurrent_sessions}")

                    # Start new sessions from pending to maintain capacity
                    while len(pending_sessioned_requests) > 0 and (max_concurrent_sessions is None or len(active_sessions) < max_concurrent_sessions):
                        # Find the first pending session and start it
                        next_session_id = next(iter(pending_sessioned_requests))
                        first_request, target_time = pending_sessioned_requests[next_session_id].pop(0)
                        if len(pending_sessioned_requests[next_session_id]) == 0:
                            pending_sessioned_requests.pop(next_session_id, None)

                        # Start the new session
                        active_sessions.add(next_session_id)
                        logging.info(f"Starting session {next_session_id}. Active sessions: {len(active_sessions)}/{max_concurrent_sessions}")
                        send(first_request)

        # Wait for tasks with duration limit
        logging.info(f"All {num_requests} tasks created. Waiting for completion...")

        if workload_duration is not None:
            elapsed = time.time() - base_time
            remaining = workload_duration - elapsed

            if remaining > 0:
                logging.info(f"Waiting up to {remaining:.1f}s more for tasks to complete (total duration: {workload_duration:.1f}s)")
                try:
                    await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=remaining)
                    logging.info("All tasks completed within workload duration.")
                except asyncio.TimeoutError:
                    pending_count = sum(1 for task in tasks if not task.done())
                    logging.warning(f"Workload duration ({workload_duration:.1f}s) reached. Cancelling {pending_count} pending requests...")
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
            else:
                logging.warning(f"Workload duration already exceeded by {-remaining:.1f}s. Cancelling pending tasks...")
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

            completed_count = sum(1 for task in tasks if task.done() and not task.cancelled())
            cancelled_count = sum(1 for task in tasks if task.cancelled())
            logging.warning(f"Benchmark complete. Total: {num_requests}, Completed: {completed_count}, Cancelled: {cancelled_count}")
        else:
            await asyncio.gather(*tasks)
            logging.warning(f"All {num_requests} requests completed for deployment.")
    finally:
        # Ensure OpenAI client is properly closed to prevent connection leaks
        # This runs regardless of whether the benchmark completed successfully,
        # was cancelled due to timeout, or encountered an error
        logging.info("Closing OpenAI client...")
        try:
            await client.close()
            logging.info("OpenAI client closed successfully.")
        except Exception as e:
            # Log but don't raise - cleanup failures shouldn't crash the program
            logging.warning(f"Error while closing OpenAI client: {e}")


def main(args):
    def _read_trace_jsonl(path: str):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows

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

    def _percentile(values, q: float) -> float:
        vals = [v for v in values if v is not None]
        if not vals:
            return 0.0
        vals.sort()
        if len(vals) == 1:
            return float(vals[0])
        # Linear interpolation (like numpy default)
        pos = (len(vals) - 1) * (q / 100.0)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return float(vals[lo])
        w = pos - lo
        return float(vals[lo] * (1.0 - w) + vals[hi] * w)

    def _jain(values) -> float:
        arr = [float(v) for v in values if v is not None]
        if not arr:
            return 0.0
        s = float(sum(arr))
        if s == 0.0:
            return 0.0
        denom = float(len(arr) * sum(x * x for x in arr))
        if denom == 0.0:
            return 0.0
        return (s * s) / denom

    def _write_summary_csv(trace_path: str, summary_csv_path: str):
        rows = _read_trace_jsonl(trace_path)
        if not rows:
            os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)) or os.getcwd(), exist_ok=True)
            with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["trace", "total_requests"])
                writer.writeheader()
                writer.writerow({"trace": os.path.abspath(trace_path), "total_requests": 0})
            return

        start_times = [_safe_float(r.get("start_time", 0.0)) for r in rows]
        end_times = [_safe_float(r.get("end_time", 0.0)) for r in rows]
        statuses = [r.get("status") for r in rows]
        users = [r.get("user") or "unknown" for r in rows]
        target_pods = [r.get("target_pod") or "unknown" for r in rows]

        latencies = [_safe_float(r.get("latency", 0.0)) for r in rows]
        prompt_tokens = [_safe_int(r.get("prompt_tokens", 0)) for r in rows]
        output_tokens = [_safe_int(r.get("output_tokens", 0)) for r in rows]
        total_tokens = [_safe_int(r.get("total_tokens", 0)) for r in rows]
        ttft = [r.get("ttft") for r in rows]
        tpot = [r.get("tpot") for r in rows]

        total_cnt = int(len(rows))
        success_mask = [s == "success" for s in statuses]
        success_cnt = int(sum(1 for x in success_mask if x))
        error_cnt = int(sum(1 for s in statuses if s == "error"))
        error_rate = float(error_cnt / total_cnt) if total_cnt > 0 else 0.0

        duration_s = float(max(end_times) - min(start_times)) if end_times and start_times else 0.0
        total_tok_sum = float(sum(total_tokens))
        e2e_tok_per_s = float(total_tok_sum / duration_s) if duration_s > 0 else 0.0
        success_req_per_s = float(success_cnt / duration_s) if duration_s > 0 else 0.0

        success_lat = [latencies[i] for i in range(total_cnt) if success_mask[i]]
        success_ttft = [_safe_float(ttft[i], 0.0) for i in range(total_cnt) if success_mask[i] and ttft[i] is not None]
        success_tpot = [_safe_float(tpot[i], 0.0) for i in range(total_cnt) if success_mask[i] and tpot[i] is not None]

        # Per-user aggregation (success only)
        per_user_total = {}
        per_user_out = {}
        per_user_cnt = {}
        per_user_lat_sum = {}
        for i, r in enumerate(rows):
            if r.get("status") != "success":
                continue
            u = users[i]
            per_user_total[u] = per_user_total.get(u, 0.0) + _safe_float(r.get("total_tokens", 0.0))
            per_user_out[u] = per_user_out.get(u, 0.0) + _safe_float(r.get("output_tokens", 0.0))
            per_user_cnt[u] = per_user_cnt.get(u, 0) + 1
            per_user_lat_sum[u] = per_user_lat_sum.get(u, 0.0) + _safe_float(r.get("latency", 0.0))

        users_seen = sorted(per_user_total.keys())
        fairness_tokens = _jain([per_user_total[u] for u in users_seen]) if users_seen else 0.0
        fairness_output = _jain([per_user_out.get(u, 0.0) for u in users_seen]) if users_seen else 0.0
        fairness_success = _jain([per_user_cnt.get(u, 0) for u in users_seen]) if users_seen else 0.0
        fairness_eff_out_tput = 0.0
        if users_seen:
            eff_vals = []
            for u in users_seen:
                lat_sum = per_user_lat_sum.get(u, 0.0)
                out_sum = per_user_out.get(u, 0.0)
                eff_vals.append(out_sum / lat_sum if lat_sum > 0 else 0.0)
            fairness_eff_out_tput = _jain(eff_vals)

        # Per-pod distribution (success only) to quantify hotspots
        per_pod_success = {}
        for i in range(total_cnt):
            if not success_mask[i]:
                continue
            p = target_pods[i]
            per_pod_success[p] = per_pod_success.get(p, 0) + 1
        pod_success_counts = list(per_pod_success.values())
        max_pod_success = int(max(pod_success_counts)) if pod_success_counts else 0
        max_pod_success_share = float(max_pod_success / success_cnt) if success_cnt > 0 else 0.0
        jain_pod_success = _jain(pod_success_counts) if pod_success_counts else 0.0

        summary_row = {
            "trace": os.path.abspath(trace_path),
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
            "jain_total_tokens": float(fairness_tokens),
            "jain_output_tokens": float(fairness_output),
            "jain_success_count": float(fairness_success),
            "jain_eff_output_tput": float(fairness_eff_out_tput),
            "pods_seen": int(len(per_pod_success)),
            "max_pod_success_requests": max_pod_success,
            "max_pod_success_share": max_pod_success_share,
            "jain_pod_success": float(jain_pod_success),
        }

        os.makedirs(os.path.dirname(os.path.abspath(summary_csv_path)) or os.getcwd(), exist_ok=True)
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
            writer.writeheader()
            writer.writerow(summary_row)

    def _derive_paths(output_file_path: str, routing_strategy: str):
        # If output_file_path is a directory, create a default base name.
        if output_file_path.endswith(os.sep) or os.path.isdir(output_file_path):
            out_dir = output_file_path
            base = "trace"
        else:
            out_dir = os.path.dirname(os.path.abspath(output_file_path)) or os.getcwd()
            base = os.path.basename(output_file_path)
            if base.endswith(".jsonl"):
                base = base[:-5]
        os.makedirs(out_dir, exist_ok=True)
        trace_path = os.path.join(out_dir, f"{base}.{routing_strategy}.jsonl")
        summary_path = os.path.join(out_dir, f"{base}.{routing_strategy}.summary.csv")
        return trace_path, summary_path, os.path.join(out_dir, f"{base}.compare.csv")

    logging.info(f"Starting benchmark on endpoint {args.endpoint}")

    # Compare mode: run multiple routing strategies sequentially with the same workload.
    compare_strategies = []
    if getattr(args, "compare_routing_strategies", None):
        compare_strategies = [s.strip() for s in str(args.compare_routing_strategies).split(",") if s.strip()]

    strategies = compare_strategies if compare_strategies else [args.routing_strategy]

    load_struct = load_workload(args.workload_path)
    send_request_func = send_request_streaming if args.streaming else send_request_batch

    duration_limit = args.duration_limit if hasattr(args, 'duration_limit') else None
    if duration_limit:
        logging.info(f"Duration limit set to {duration_limit:.1f}s (from command line)")

    max_concurrent_sessions = args.max_concurrent_sessions if hasattr(args, 'max_concurrent_sessions') else None

    compare_rows = []
    compare_csv_path = None
    for strat in strategies:
        if compare_strategies:
            trace_path, summary_path, compare_csv_path = _derive_paths(args.output_file_path, strat)
        else:
            trace_path = os.path.abspath(args.output_file_path)
            # Derive a default summary path next to the trace
            if trace_path.endswith(".jsonl"):
                summary_path = trace_path[:-5] + ".summary.csv"
            else:
                summary_path = trace_path + ".summary.csv"
            compare_csv_path = None
        logging.info(f"Running routing-strategy={strat}, trace={trace_path}")

        start_time = time.time()
        with open(trace_path, 'w', encoding='utf-8') as output_file:
            asyncio.run(benchmark_launch(
                api_key=args.api_key,
                endpoint=args.endpoint,
                max_retries=args.max_retries,
                scale_factor=args.time_scale,
                timeout=args.timeout_second,
                routing_strategy=strat,
                load_struct=load_struct,
                output_file=output_file,
                model=args.model,
                max_output=args.output_token_limit,
                send_request_func=send_request_func,
                duration_limit=duration_limit,
                max_concurrent_sessions=max_concurrent_sessions,
                user_count=args.user_count,
            ))
        end_time = time.time()
        logging.info(f"Benchmark completed (routing-strategy={strat}) in {end_time - start_time:.2f} seconds")

        if getattr(args, "emit_summary", False) or compare_strategies:
            _write_summary_csv(trace_path, summary_path)
            # Read back summary row for compare output
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row["routing_strategy"] = strat
                        compare_rows.append(row)
                        break
            except Exception as e:
                logging.warning(f"Failed to read summary csv {summary_path}: {e}")

    if compare_rows and compare_csv_path:
        # Write a compact compare table (one row per routing strategy)
        # Keep the field order stable for diffs.
        prefer_cols = [
            "routing_strategy",
            "success_requests",
            "error_rate",
            "duration_s",
            "e2e_tokens_per_s",
            "e2e_latency_p50_s",
            "e2e_latency_p99_s",
            "ttft_p50_s",
            "ttft_p99_s",
            "pods_seen",
            "max_pod_success_share",
            "jain_pod_success",
            "jain_eff_output_tput",
            "trace",
        ]
        # Union all keys
        all_keys = set()
        for r in compare_rows:
            all_keys.update(r.keys())
        fieldnames = [c for c in prefer_cols if c in all_keys] + [k for k in sorted(all_keys) if k not in prefer_cols]
        with open(compare_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in compare_rows:
                writer.writerow(r)
        logging.info(f"Wrote compare summary: {compare_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Workload Generator Client')
    parser.add_argument("--workload-path", type=str, default=None, help="File path to the workload file.")
    parser.add_argument("--model", type=str, default=None, help="Default target model (if workload does not contains target model).")
    parser.add_argument('--endpoint', type=str, required=True)
    parser.add_argument("--api-key", type=str, default=None, help="API key to the service. ")
    parser.add_argument('--output-file-path', type=str, default="output.jsonl")
    parser.add_argument("--streaming", action="store_true", help="Use streaming client.")
    parser.add_argument("--routing-strategy", type=str, required=False, default="random", help="Routing strategy to use.")
    parser.add_argument(
        "--compare-routing-strategies",
        type=str,
        default=None,
        help="逗号分隔的 routing-strategy 列表；若设置则按顺序重复跑同一 workload 并输出 compare.csv（例如 'vtc-basic,vtc-pred'）。",
    )
    parser.add_argument(
        "--emit-summary",
        action="store_true",
        help="为本次 run 额外写出 summary.csv（默认只在 compare 模式写）。",
    )
    parser.add_argument("--output-token-limit", type=int, required=False, default=None, help="Limit the maximum number of output tokens.")
    parser.add_argument('--time-scale', type=float, default=1.0, help="Scaling factor for workload's logical time.")
    parser.add_argument('--timeout-second', type=float, default=60.0, help="Timeout for each request in seconds.")
    parser.add_argument('--max-retries', type=int, default=0, help="Number of maximum retries for each request.")
    parser.add_argument('--duration-limit', type=float, default=None, help="Duration limit in seconds. Benchmark stops after this time, cancelling pending requests. If not set, uses workload's last timestamp.")
    parser.add_argument('--max-concurrent-sessions', type=int, default=None, help="Maximum number of sessions that can run concurrently. Only applies to sessioned workloads.")
    parser.add_argument('--user-count', type=int, default=10, help="用户数量（用于发送 user header，并在 analyze 阶段计算 Jain fairness）。")

    args = parser.parse_args()
    main(args)
