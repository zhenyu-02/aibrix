# AIBrix vtc-pred 项目报告：输出长度感知的公平路由（含 VTC “clamp 饱和热点”修复与真实排队/异构验证）

## 摘要

本项目围绕 AIBrix Gateway 的路由算法 VTC（Virtual Token Counter）开展：

1. **验证并引入输出长度预测器**（`SimpleOutputPredictor`）到 VTC 的成本估计中，形成 `vtc-pred`；
2. **补齐“真实现象可观测”条件**（排队、并发上限、长输出更慢、异构节点）使路由差异能转化为可观测的 TTFT/E2E 收益；
3. 修复了 VTC 在高累计 token 场景下的**映射病态：clamp 导致永久热点**，将映射改为 **wrap（mod）+ 环形距离**；
4. 在同一套 workload（`output/session_parquet_adversarial_00012/workload.jsonl`，来源于 e2e 测试数据集 `allenai/WildChat-1M`）与更真实的 mock 条件下，给出可复现的对比证据：修复后 `vtc-pred` 在吞吐与尾延迟上显著优于 `vtc-basic`，并能更稳定地抑制热点 Pod 的拥塞。

## 1. 背景知识

### 1.1 AIBrix Router 在做什么

当用户请求进入 AIBrix Gateway 后，需要从“当前可用的一组后端推理 Pod”中选一个作为转发目标。这个“选哪个 Pod”的策略就是 routing algorithm（router）。

- 路由上下文结构体在 `pkg/types/router_context.go:49`，包含 `Model`、`Message`、`User` 等字段。
- router 需要通过 `ctx.SetTargetPod(...)`（`pkg/types/router_context.go:171`）写入目标 Pod，并返回目标地址。

### 1.2 LLM Serving 的资源消耗为何与输出 token 强相关

在常见的推理引擎（例如 vLLM）中，一次请求通常包含：

- **Prefill**：处理 prompt，成本主要与输入 token 数相关；
- **Decode**：逐 token 生成输出，成本主要与输出 token 数相关。

当输出长度分布是重尾（少量请求输出非常长）时，如果路由算法低估这些长输出请求的成本，会导致：

- 长输出请求过早挤入同一条执行/排队路径，放大排队；
- 用户公平性与尾延迟变差；
- 在异构节点（慢卡/拥塞）条件下更容易形成热点。

### 1.3 VTC 的核心思想（Virtual Token Counter）

VTC 用一个“虚拟 token 账本”来近似用户近期对系统资源的消耗：

- 每个用户维护一个滑动窗口累计值 `userTokens`；
- 每次请求路由结束后，把本次请求的“估算成本”累加进 `userTokens`。

成本更新公式在 `pkg/plugins/gateway/algorithms/vtc/token_tracker.go:269` 到 `pkg/plugins/gateway/algorithms/vtc/token_tracker.go:316`：

```
newTokens = inputTokens * InputTokenWeight + outputTokens * OutputTokenWeight
```

其中默认权重在 `pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:32` 到 `pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:37`，默认 `output` 权重更高（2.0），意味着**输出长度估计的准确性对路由行为影响更大**。

## 2. 动机：为什么要做 vtc-pred

### 2.1 vtc-basic 的输出估计过于粗糙

vtc-basic 默认用规则估计输出 token：

- 输入 token：`len(message)/4`
- 输出 token：`inputTokens * 1.5`

对应实现为 `SimpleTokenEstimator`（`pkg/plugins/gateway/algorithms/vtc/token_estimator.go:29`）。

该估计不利用任何历史，不区分模型与场景，在重尾输出条件下容易系统性偏差。

### 2.2 AIBrix 已经有 OutputPredictor，但 VTC 没用

AIBrix cache 层已经维护了 per-model 的输出长度预测器：

- 预测接口：`SimpleOutputPredictor.Predict`（`pkg/cache/output_predictor.go:196`）
- 分桶逻辑：`token2bucket`（`pkg/cache/output_predictor.go:233`）
- 历史回填：请求完成时 `meta.OutputPredictor.AddTrace(...)`（`pkg/cache/cache_impl.go:249` 到 `pkg/cache/cache_impl.go:273`）

这意味着引入 predictor 到 VTC 的工程成本较低，且 predictor 会随着真实请求不断更新。

### 2.3 仅“预测更准”不够：必须让“真实现象可观测”

如果后端执行时间与 token 长度无关、并发不受限、不会排队，那么不同路由策略的差异难以体现在 TTFT/E2E 上。

因此项目同时引入了更真实的 mock 条件（详见 §5.2）：

- 并发上限（导致等待/排队）
- token 耗时模型（输出长 -> decode 更慢）
- 异构 profile（某个 pod 更慢/更容易拥塞）

## 3. Proposal 计划（从验证到落地）

项目按“先验证 predictor 必要性，再集成路由，再做端到端证据”的路径推进：

1. **离线/在线评估 predictor（必要性与有效性）**：在真实轨迹数据 BurstGPT 上比较 `SimpleOutputPredictor` vs `ceil(input*1.5)`，验证 predictor 不是“随机波动”。
2. **把 predictor 接入 VTC，形成 vtc-pred**：在不改变 VTC 主结构的前提下，仅替换输出 token 估计来源，并保留 fallback。
3. **构建可观测的端到端实验环境**：用同一 workload 驱动压测，mock 后端具备排队、并发上限、token 耗时、异构配置。
4. **定位并修复 VTC 映射病态（clamp 饱和热点）**：避免机制上系统性惩罚 vtc-pred。
5. **给出“同一 workload + 更真实条件”下的硬证据**：吞吐、TTFT/E2E 尾延迟、热点抑制。

## 4. 初步验证实验：predictor 的有效性（BurstGPT）

验证结论与复现实验记录在：`docs/proposals/vtc-pred/vtc-pred-proposal-burstgpt-eval.md`。

### 4.1 评估方法（核心要点）

- 数据：BurstGPT v1.2（清洗后约 140 万条样本）。
- 评估方式：严格在线（对每个模型按时间顺序预测，再写入历史，避免数据泄漏）。
- 指标重点：由于 predictor 是分桶输出（`2^k`），以 **bucket accuracy** 为主要指标，同时给出 MAE/MAPE 作为补充。

### 4.2 总体结果（window_seconds=60，overall）

来自 `docs/proposals/vtc-pred/vtc-pred-proposal-burstgpt-eval.md:154` 到 `:189`：

| 指标 | SimpleOutputPredictor | 基线：`ceil(input*1.5)` | 改善 |
|---|---:|---:|---:|
| bucket_acc (↑) | 0.6076 | 0.0753 | +0.5323 |
| bucket_acc±1 (↑) | 0.8346 | 0.1740 | +0.6606 |
| bucket_acc±2 (↑) | 0.8851 | 0.2803 | +0.6048 |
| MAE (↓) | 66.385 | 831.388 | 765.003 ↓ |

结论：在“输出长度等级识别”维度（bucket）上，`SimpleOutputPredictor` 相比 `ceil(input*1.5)` 有显著提升，因此把该 predictor 引入 VTC 的成本估计是有必要且合理的。

## 5. 方案落地：vtc-pred 与真实现象建模

### 5.1 vtc-pred 如何接入 VTC（代码路径）

VTC 路由主流程在 `pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:86`。

当 `Variant == vtc-pred` 时：

- 输入 token：使用真实 tokenizer 的 prompt 长度 `ctx.PromptLength()`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:103`，实现见 `pkg/types/router_context.go:132`）
- 输出 token：从 cache 获取 predictor 并预测 `predictor.Predict(promptLen)`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:277` 到 `:287`）
- fallback：若 predictor 不可用，则继续使用 `SimpleTokenEstimator` 的输出估计

### 5.2 “更真实”的 mock 后端：排队 + token 耗时 + 异构

为了让路由差异能转化为 TTFT/E2E 差异，mock 后端引入如下机制：

1. **并发上限 + 排队**：`AIBRIX_MOCK_MAX_CONCURRENCY` 限制每个 backend 同时处理的请求数，超过则等待（`development/app/app.py:77` 到 `development/app/app.py:129`）。
2. **token 耗时模型**：开启后端耗时与 token 数相关（prefill 与 decode TPS 可配置），让长输出请求更慢（`development/app/app.py:30` 到 `development/app/app.py:140`）。
3. **可复现异构 profile**：通过 `AIBRIX_MOCK_PROFILE=normal/slow/fast` 控制 TPS 与并发，复现实验中的“慢卡/拥塞节点”（`development/app/app.py:35` 到 `development/app/app.py:63`）。
4. **让排队在 /metrics 可见**：mock 会维护 `_mock_waiting/_mock_inflight` 并写入 `overrides`（`development/app/app.py:93` 到 `development/app/app.py:101`）。

### 5.3 VTC 映射病态：clamp 饱和热点问题与修复

#### 5.3.1 问题是什么

VTC 需要把 `userTokens` 映射到 pod index 空间，并以此计算 fairnessScore。

如果映射是 clamped-linear（“超过上限就永远卡在最后一个 index”），则当 `userTokens/bucket` 足够大时：

- `normalizedTokens` 永远等于 `npods-1`
- `fairnessScore` 对最后一个 pod 永远最小
- 结果是大量请求长期打到最后一个 pod，形成永久热点，并在存在并发上限/排队时放大尾延迟

#### 5.3.2 修复做了什么（wrap + 环形距离）

当前实现改为：

1. **wrap 映射**：`normalizedTokens = mod(userTokens/adaptiveBucketSize, npods)`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:162` 到 `:173`）。
2. **环形距离**：`fairnessScore = min(|i-x|, n-|i-x|)`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:175` 到 `:181`）。

这保证了 `userTokens` 增长不会把系统推入“永久偏好某个 pod”的吸收态。

### 5.4 可选增强：让 VTC utilization 感知排队

默认 VTC 的 podLoad 使用 `NumRequestsRunning`（运行中请求数）。为了在有排队时更敏感，可选把 waiting 纳入 podLoad：

- 环境变量：`AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_INCLUDE_WAITING`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:40` 到 `:58`）
- 实现：当开启时 podLoad = running + waiting（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:193` 到 `:211`）
- 同步订阅等待指标：`SubscribedMetrics()` 会额外订阅 `NumRequestsWaiting`（`pkg/plugins/gateway/algorithms/vtc/vtc_basic.go:263` 到 `:270`）

## 6. 端到端实验设计（同一 workload，对比 vtc-basic vs vtc-pred）

### 6.1 固定的 workload

- `output/session_parquet_adversarial_00012/workload.jsonl`
- 数据来源：该 workload 由 AIBrix e2e 测试所使用的数据集 `allenai/WildChat-1M` 生成/抽样并落盘到 `output/` 目录（用于保证对比实验的输入流量一致、可复现）。
- 目标：在相同输入流量下，只比较 router 行为差异。

### 6.2 对比策略

- `vtc-basic`
- `vtc-pred`

### 6.3 评估指标

- `e2e_tokens_per_s`：端到端吞吐（tokens/s）
- `e2e_latency_p50/p99`：端到端延迟分位
- `ttft_p50/p99`：首 token 延迟分位（更能体现排队与拥塞）
- 热点抑制：从 trace 统计 `target_pod` 分布，关注“最热点 pod 占比”

## 7. 硬证据：实验结果与收益

### 7.1 主证据：bucket=5k（wrap 修复版）compare_run2

数据来源：

- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run2/basic/analysis/summary.csv`
- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run2/pred/analysis/summary.csv`

对比表（400 req，全成功，总 token 相同）：

| 指标 | vtc-basic | vtc-pred | 改善 |
|---|---:|---:|---:|
| e2e_tokens_per_s (↑) | 2464.14 | 3172.91 | +28.7% |
| e2e_latency_p50 (↓) | 23.12s | 5.80s | 3.99× ↓ |
| e2e_latency_p99 (↓) | 59.11s | 18.73s | 3.16× ↓ |
| ttft_p50 (↓) | 20.90s | 3.71s | 5.63× ↓ |
| ttft_p99 (↓) | 56.80s | 15.66s | 3.63× ↓ |

热点抑制（从 trace 统计 `target_pod`）：

- vtc-basic：最热点 pod `host.docker.internal:8004` 占比 **76.00%**
- vtc-pred：最热点 pod 占比降到 **62.50%**（下降 13.5 个百分点）

trace 位置：

- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run2/basic/vtc-basic.trace.jsonl`
- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run2/pred/vtc-pred.trace.jsonl`

### 7.2 更贴近生产现象：slow8004（compare_run3_slow8004）

数据来源：

- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run3_slow8004/basic/analysis/summary.csv`
- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run3_slow8004/pred/analysis/summary.csv`

对比表（400 req，全成功）：

| 指标 | vtc-basic | vtc-pred | 改善 |
|---|---:|---:|---:|
| e2e_tokens_per_s (↑) | 2418.87 | 2939.00 | +21.5% |
| e2e_latency_p50 (↓) | 23.38s | 10.80s | 2.16× ↓ |
| e2e_latency_p99 (↓) | 61.35s | 30.48s | 2.01× ↓ |
| ttft_p50 (↓) | 21.06s | 9.40s | 2.24× ↓ |
| ttft_p99 (↓) | 58.13s | 27.21s | 2.14× ↓ |

热点抑制（trace 统计）：

- vtc-basic：最热点 pod 占比 **76.00%**
- vtc-pred：最热点 pod 占比 **66.50%**（下降 9.5 个百分点）

trace 位置：

- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run3_slow8004/basic/vtc-basic.trace.jsonl`
- `output/session_parquet_adversarial_00012_bucket5k_wrap/compare_run3_slow8004/pred/vtc-pred.trace.jsonl`

## 8. 结论（我们证明了什么）

1. predictor 有效性已经在 BurstGPT 上被独立验证：相对 `ceil(input*1.5)`，在 bucket 维度显著更准（§4）。
2. 早期“vtc-pred 更差”的根因不必然是 predictor 不准，而是 VTC 的 clamped-linear 映射存在**饱和吸收态**，会在高累计 token 时把大量请求压到同一 pod，排队被放大（§5.3）。
3. 修复映射为 wrap + 环形距离后，在“排队可发生、长输出更慢、节点异构可复现”的条件下：

   - `vtc-pred` 能更早/更强地把长输出成本反映到 `userTokens` 增长中；
   - 进而在打分中更倾向把请求从热点/排队点分散出去；
   - 最终表现为吞吐更高、TTFT/E2E 尾延迟更低，并且热点 pod 占比下降。

## 9. 如何复现（最短路径）

### 9.1 启动 gateway-plugin（standalone）

```bash
go run ./cmd/plugins \
  --standalone \
  --endpoints-config=deployment/standalone/configs/endpoints.yaml \
  --grpc-bind-address=:15052 \
  --metrics-bind-address=:18081
```

### 9.2 启动 4 个 mock 后端（示例：让 8004 更慢）

```bash
STANDALONE_MODE=true PORT=8001 AIBRIX_MOCK_ENABLE_LATENCY_MODEL=1 AIBRIX_MOCK_PROFILE=normal python3 development/app/app.py
STANDALONE_MODE=true PORT=8002 AIBRIX_MOCK_ENABLE_LATENCY_MODEL=1 AIBRIX_MOCK_PROFILE=normal python3 development/app/app.py
STANDALONE_MODE=true PORT=8003 AIBRIX_MOCK_ENABLE_LATENCY_MODEL=1 AIBRIX_MOCK_PROFILE=normal python3 development/app/app.py
STANDALONE_MODE=true PORT=8004 AIBRIX_MOCK_ENABLE_LATENCY_MODEL=1 AIBRIX_MOCK_PROFILE=slow   python3 development/app/app.py
```

### 9.3 跑 benchmark（同一 workload，一键对比）

```bash
python3 benchmarks/client/client.py \
  --endpoint http://127.0.0.1:18080 \
  --workload-path output/session_parquet_adversarial_00012/workload.jsonl \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --streaming \
  --compare-routing-strategies vtc-basic,vtc-pred \
  --output-file-path output/my_compare/trace.jsonl \
  --user-count 8
```

说明：benchmark 客户端刻意**不发送 `model` header**，避免 gateway 在 body 未解析前提前路由导致 `RoutingContext.Message` 为空，从而隐藏 vtc-pred 的 token 特征差异（`benchmarks/client/client.py:81` 到 `:85`）。

## 10. 限制与后续工作

- utilization 信号目前默认只用 `NumRequestsRunning`，对“队列等待”不一定敏感；可按 §5.4 开启 waiting 纳入 podLoad，并结合更真实的指标/权重继续优化。
- bucket/minTokens 等参数是敏感度旋钮：越小越敏感，但抖动可能更大；后续可系统性扫一组区间并给出默认推荐参数。
