#!/bin/bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
MODEL_NAME="${MODEL_NAME:-your-model-name}"
USER_NAME="${USER_NAME:-vtc-user}"
ROUTING_STRATEGY="${ROUTING_STRATEGY:-vtc-pred}"
DEBUG="${DEBUG:-0}"
SEND_MODEL_HEADER="${SEND_MODEL_HEADER:-0}"

# Optional: initialize Redis users for standalone gateway (when request header `user` is set).
INIT_REDIS_USERS="${INIT_REDIS_USERS:-0}"
USER_COUNT="${USER_COUNT:-10}"
USER_PREFIX="${USER_PREFIX:-user-}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_CONTAINER="${REDIS_CONTAINER:-aibrix-redis}"

if ! command -v curl >/dev/null 2>&1; then
  echo "[ERROR] missing dependency: curl" >&2
  exit 2
fi

init_redis_user() {
  local name="$1"
  local key="aibrix-users/${name}"
  local val
  val=$(printf '{"name":"%s","rpm":0,"tpm":0}' "$name")

  if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" SET "$key" "$val" >/dev/null
    return 0
  fi

  if command -v docker >/dev/null 2>&1; then
    if docker ps --format '{{.Names}}' | grep -qx "$REDIS_CONTAINER"; then
      docker exec "$REDIS_CONTAINER" redis-cli SET "$key" "$val" >/dev/null
      return 0
    fi
  fi

  echo "[ERROR] 无法初始化 Redis 用户：缺少 redis-cli，且未发现运行中的 redis 容器 ($REDIS_CONTAINER)。" >&2
  echo "        解决方式之一：安装 redis-cli 或启动 redis 容器后重试。" >&2
  return 1
}

if [ "$INIT_REDIS_USERS" = "1" ]; then
  echo "[INFO] Initializing Redis users... (USER_COUNT=$USER_COUNT, USER_PREFIX=$USER_PREFIX, REDIS=$REDIS_HOST:$REDIS_PORT)" >&2
  init_redis_user "$USER_NAME"
  i=0
  while [ "$i" -lt "$USER_COUNT" ]; do
    init_redis_user "${USER_PREFIX}${i}"
    i=$((i+1))
  done
fi

REQ_BODY=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32,"stream":false}' "$MODEL_NAME")

declare -a MODEL_HEADER=()
if [ "$SEND_MODEL_HEADER" = "1" ]; then
  MODEL_HEADER=(-H "model: $MODEL_NAME")
fi

HEADERS_FILE="$(mktemp)"
BODY_FILE="$(mktemp)"
cleanup() {
  rm -f "$HEADERS_FILE" "$BODY_FILE"
}
trap cleanup EXIT

set +e
HTTP_CODE=$(curl -sS --connect-timeout 2 --max-time 15 -w "%{http_code}" -D "$HEADERS_FILE" -o "$BODY_FILE" \
  -H "Content-Type: application/json" \
  ${MODEL_HEADER[@]+"${MODEL_HEADER[@]}"} \
  -H "routing-strategy: $ROUTING_STRATEGY" \
  -H "user: $USER_NAME" \
  -X POST "$BASE_URL/v1/chat/completions" \
  -d "$REQ_BODY")
CURL_EXIT=$?
set -e

if [ $CURL_EXIT -ne 0 ]; then
  echo "[ERROR] curl request failed (exit=$CURL_EXIT)." >&2
  echo "        请确认网关已启动，并且 BASE_URL 可访问：BASE_URL=$BASE_URL" >&2
  exit 1
fi

if [ "$HTTP_CODE" != "200" ]; then
  echo "[ERROR] unexpected http status: $HTTP_CODE" >&2

  # 友好提示：默认 8888 在本机经常被 Jupyter 占用（会返回 403 + TornadoServer）
  if grep -qi "tornadoserver" "$HEADERS_FILE" || grep -qi "jupyter" "$BODY_FILE"; then
    echo "        当前 BASE_URL 似乎不是 AIBrix Gateway（更像 Jupyter）。" >&2
    echo "        请显式设置 BASE_URL 指向你的网关，例如：" >&2
    echo "          BASE_URL=http://127.0.0.1:<gateway-port> ./scripts/test-vtc-pred-smoke.sh" >&2
  fi

  if [ "$DEBUG" = "1" ]; then
    cat "$HEADERS_FILE" >&2
    cat "$BODY_FILE" >&2
  else
    echo "        (如需完整响应，使用 DEBUG=1 重试)" >&2
    sed -n '1,80p' "$HEADERS_FILE" >&2 || true
    sed -n '1,80p' "$BODY_FILE" >&2 || true
  fi
  exit 1
fi

if ! grep -qi "^target-pod:" "$HEADERS_FILE"; then
  echo "[ERROR] missing response header: target-pod" >&2
  echo "        这通常表示 ext-proc/gateway 没有把目标 pod 回传到 header，或路由插件未生效。" >&2
  cat "$HEADERS_FILE" >&2
  cat "$BODY_FILE" >&2
  exit 1
fi

cat "$HEADERS_FILE"
cat "$BODY_FILE"
