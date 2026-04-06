import json
import openai
import threading
import os
from typing import List, Any, Dict

def load_workload(input_path: str) -> List[Any]:
    load_struct = None
    if input_path.endswith(".jsonl"):
        with open(input_path, "r") as file:
            load_struct = [json.loads(line) for line in file]
    else:
        with open(input_path, "r") as file:
            load_struct = json.load(file)
    return load_struct

# Function to wrap the prompt into OpenAI's chat completion message format.
def prepare_prompt(prompt: str, 
                   session_id: str = None, 
                   history: Dict = None,
                   history_lock: threading.Lock = None) -> List[Dict]:
    """
    Wrap the prompt into OpenAI's chat completion message format.

    :param prompt: The user prompt to be converted.
    :param session_id: Optional session ID for conversation history.
    :param history: Optional history dictionary to store conversation.
    :param history_lock: Optional threading lock for thread safety.
    :return: A list containing chat completion messages.
    """
    if session_id is not None and history is not None:
        if history_lock:
            with history_lock:
                past_history = history.get(session_id, [])
                user_message = {"role": "user", "content": f"{prompt}"}
                past_history.append(user_message) 
                history[session_id] = past_history
                return list(past_history)
        else:
            # Fallback for when no lock is provided (not thread-safe)
            past_history = history.get(session_id, [])
            user_message = {"role": "user", "content": f"{prompt}"}
            past_history.append(user_message) 
            history[session_id] = past_history
            return list(past_history)
    else:    
        user_message = {"role": "user", "content": prompt}
        return [user_message]
    
def update_response(response: str, 
                    session_id: str = None, 
                    history: Dict = None,
                    history_lock: threading.Lock = None):
    """
    Update the conversation history with the assistant's response.

    :param response: The assistant's response to add to history.
    :param session_id: Optional session ID for conversation history.
    :param history: Optional history dictionary to store conversation.
    :param history_lock: Optional threading lock for thread safety.
    """
    if session_id is not None and history is not None:
        if history_lock:
            with history_lock:
                past_history = history.get(session_id, [])
                assistant_message = {"role": "assistant", "content": f"{response}"}
                past_history.append(assistant_message) 
        else:
            # Fallback for when no lock is provided (not thread-safe)
            past_history = history.get(session_id, [])
            assistant_message = {"role": "assistant", "content": f"{response}"}
            past_history.append(assistant_message)

def create_client(api_key: str,
                  endpoint: str,
                  max_retries: int,
                  timeout: float,
                  routing_strategy: str,
                  ):
    # openai>=1.x 强制要求 api_key 非空；本地/CPU-only bench 走的是兼容 OpenAI 的 mock 服务，
    # 因此默认填一个占位 key，避免调用方必须配置环境变量。
    effective_key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client = openai.AsyncOpenAI(
        api_key=effective_key,
        base_url=endpoint + "/v1",
        max_retries=max_retries,
        timeout=timeout,
    )
    if routing_strategy is not None:
        client = client.with_options(
            default_headers={"routing-strategy": routing_strategy}
        )
    return client


def pick_user_for_request(request: Dict, request_id: int, user_count: int) -> str:
    """为请求挑选一个 user（用于网关侧按用户统计/路由）。

    优先级：
    1) workload/request 里显式带了 user 字段
    2) sessioned workload：用 session_id 做稳定映射
    3) 否则按 request_id 取模
    """
    if request is None:
        return f"user-{request_id % user_count}"

    explicit = request.get("user")
    if explicit:
        return str(explicit)

    session_id = request.get("session_id")
    if session_id is not None:
        try:
            return f"user-{int(session_id) % user_count}"
        except Exception:
            return f"user-{request_id % user_count}"

    return f"user-{request_id % user_count}"
