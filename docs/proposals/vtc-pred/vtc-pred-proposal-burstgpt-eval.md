# vtc-pred 调研实验：BurstGPT 上的输出长度预测器评估

> 本文档是 [Aibrix vtc-pred 公平路由 Proposal](https://docs.qq.com/markdown/DZGhTRHVwcnJZWGZ3?)  的调研实验记录，用于验证：在 VTC 成本估计中引入 `SimpleOutputPredictor`（历史分布 + 分桶随机采样）相较于 `output = ceil(input * 1.5)` 的规则估计是否更准确。

## 1. 实验目标

- 证明在真实/准真实的请求轨迹上，`SimpleOutputPredictor` 的输出长度估计**显著优于**基线规则 `output = ceil(input * 1.5)`。
- 同时考察不同滑动时间窗口 `window_seconds` 对预测效果的影响，为后续 vtc-pred 参数选择提供依据。

## 2. 评估对象

### 2.1 Predictor：SimpleOutputPredictor（Aibrix 现有实现）

实现位置：`/Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go`。

核心机制（概念化描述）：

1. **输入/输出 token 分桶**：
   - `bucket(tokens) = round(log2(tokens))`，并在 bucket 数上限处截断。
   - 这意味着 predictor 的“输出值空间”是粗粒度的幂次：`1, 2, 4, 8, ...`。

2. **按输入桶维护输出桶直方图**：
   - 对每个输入桶 `b_in`，维护该桶下历史输出桶 `b_out` 的计数分布（moving histogram）。

3. **预测时加权随机采样输出桶**：
   - 给定新请求输入长度 `x`，先得到 `b_in = bucket(x)`。
   - 在该 `b_in` 对应的输出桶分布上按权重随机抽样一个 `b_out`。
   - 返回预测 token 数为 `2^b_out`（桶代表值）。

4. **滑动窗口**：
   - 只保留最近 `window_seconds` 的历史（内部以 10s 为时间片滚动）。

5. **冷启动策略（OptimisticColdPrediction）**：
   - 当某输入桶没有历史时，返回 `1`（偏乐观、profile 友好），而不是追求“绝对准确”。

对应代码要点：
- 分桶：`/Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go:233`
- 预测（按直方图权重随机采样）：`/Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go:196`
- 冷启动：`/Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go:215`
- 窗口滚动：`/Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go:244`

### 2.2 Baseline：规则估计

- 规则：`pred = ceil(input_tokens * 1.5)`
- 含义：与 `vtc-basic` 的默认输出估计一致（参见 proposal 的背景部分）。

### 2.3 Baseline：随机点验（sanity check）

为了验证 predictor 是否可能“还不如随机猜”，额外引入两类随机基线：

1. **随机均匀（uniform）**
   - 对每个模型，统计全量样本的 `min_output` 与 `max_output`。
   - 每次预测在 `[min_output, max_output]` 之间均匀采样一个 token 作为预测。

2. **随机正态（normal）**
   - 对每个模型，基于全量样本统计 `mean_output` 与 `std_output`。
   - 每次预测从 `N(mean_output, std_output)` 采样，再截断到 `[min_output, max_output]` 并四舍五入为整数。

3. **输入输出联合分布随机（joint）**
   - 对每个模型，基于全量样本统计 `input_len` 与 `output_len` 的联合分布（以 log2 bucket 表示）。
   - 给定输入长度，按该输入桶对应的输出桶分布随机采样一个输出桶，返回桶代表值 `2^k`。

## 3. 数据集与清洗

### 3.1 数据集

- BurstGPT v1.2：`/Users/bytedance/cuhk/aibrix/len-pred-exp/BurstGPT-1.2/data/BurstGPT_1.csv`
- 字段：`Timestamp, Model, Request tokens, Response tokens, ...`

### 3.2 清洗规则

为了避免失败样本/异常样本对评估指标造成不可解释的扭曲，本次评估只保留：

- `Model` 非空
- `Request tokens > 0`
- `Response tokens > 0`
- （防御性）同一 `Model` 内如果出现时间戳回退（`ts < last_ts[model]`）则丢弃该行

本次全量数据统计：

- `total_rows = 1,429,737`
- `kept_rows = 1,404,294`
- `dropped_invalid = 25,443`
- `dropped_out_of_order = 0`

实现位置：`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py:198`、`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py:245`。

## 4. 实验方法（在线评估）

### 4.1 评估方式：严格在线（online）

对每个模型独立维护 predictor：`predictor[model]`。

对每条样本 `(model, ts, req, res)`，按时间顺序执行：

1. `pred = predictor[model].predict(req)`（仅依赖过去窗口内历史）
2. 记录该条样本的误差指标（对比 `pred` vs `res`）
3. `predictor[model].add_trace_with_timestamp(req, res, ts)`（将当前样本写入历史）

这一流程确保不会“用未来的 label 预测过去”（避免数据泄漏）。

实现位置：`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py:251` 到 `:275`。

### 4.2 点验实验设计：随机基线

随机基线属于“sanity check”，用于回答“predictor 是否比随机好”这一问题，因此采用全量数据统计得到分布参数：

- 随机均匀：每个模型独立统计 `min_output` 与 `max_output`，在线评估时对每条样本均匀抽样。
- 随机正态：每个模型独立统计 `mean_output` 与 `std_output`，在线评估时从正态分布采样并截断。
- 联合分布随机：每个模型独立统计 `input_len`/`output_len` 的联合分桶分布，在线评估时按输入桶的输出分布随机采样。

该点验不影响对 predictor 的在线评估结论，仅作为对比基线补充。

### 4.3 指标口径（重点）

由于 `SimpleOutputPredictor` 输出本质是“桶代表值”（`2^k`），它不是一个连续回归器；因此仅用 token 级 MAPE/MAE 会在解释上偏“苛刻/不贴合”。本次同时给两类指标：

1. **桶命中率（bucket accuracy，推荐用作“准确率”）**
   - `b_true = bucket(actual_output)`
   - `b_pred = bucket(pred_output)`
   - `bucket_acc = P(b_pred == b_true)`
   - `bucket_acc±1 = P(|b_pred - b_true| <= 1)`
   - `bucket_acc±2 = P(|b_pred - b_true| <= 2)`

2. **token 级误差（补充）**
   - MAE、RMSE
   - MAPE（注意：当预测系统性偏离很大时，MAPE 可以远大于 1；`1 - MAPE` 不是通常意义的准确率）

实现位置：`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py:49`、`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py:269`。

## 5. 复现实验

脚本：`/Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py`

示例：

```bash
python3 /Users/bytedance/cuhk/aibrix/len-pred-exp/burstgpt_eval.py \
  --data_path /Users/bytedance/cuhk/aibrix/len-pred-exp/BurstGPT-1.2/data/BurstGPT_1.csv \
  --window_seconds 60 \
  --seed 42 \
  --baseline_multiplier 1.5 \
  --random_seed_uniform 43 \
  --random_seed_normal 44 \
  --random_seed_joint 45
```

说明：`seed` 固定用于保证 predictor 加权随机采样的可复现性；`random_seed_uniform`、`random_seed_normal`、`random_seed_joint` 固定用于随机点验基线的可复现性。

## 6. 结果与分析

### 6.1 总体结论（是否“更准”）

在 BurstGPT（清洗后 140 万条样本）上，`SimpleOutputPredictor` 相对 `ceil(input * 1.5)` 的规则估计，**桶级准确率提升约 +0.51 ~ +0.53（+51~53 个百分点）**，属于显著提升。

以 `window_seconds = 60` 为例（overall）：

- `SimpleOutputPredictor`：`bucket_acc = 0.6076`
- baseline：`bucket_acc = 0.0753`
- **提升**：`+0.5323`

同时在“允许偏差 1 桶/2 桶”的容忍度下，提升更大：

- `bucket_acc±1`：`0.8346 - 0.1740 = +0.6606`
- `bucket_acc±2`：`0.8851 - 0.2803 = +0.6048`

为了更直观看到 baseline 与 predictor 的差异，下表给出 `window_seconds=60` 的 overall 指标对比（`seed=42`）：

| 指标 | SimpleOutputPredictor | baseline(`ceil(input*1.5)`) | 改善 |
|---|---:|---:|---:|
| bucket_acc (↑) | 0.6076 | 0.0753 | +0.5323 |
| bucket_acc±1 (↑) | 0.8346 | 0.1740 | +0.6606 |
| bucket_acc±2 (↑) | 0.8851 | 0.2803 | +0.6048 |
| MAE (↓) | 66.385 | 831.388 | 765.003 ↓ |
| RMSE (↓) | 181.223 | 1398.373 | 1217.150 ↓ |
| MAPE (↓) | 1.4011 | 54.0629 | 52.6618 ↓ |

说明：这里的“改善”对 (↑) 指标表示 `predictor - baseline`，对 (↓) 指标表示 `baseline - predictor`。

补充随机点验基线（`window_seconds=60`，overall，`random_seed_uniform=43`，`random_seed_normal=44`，`random_seed_joint=45`）：

| 指标 | SimpleOutputPredictor | baseline(`ceil(input*1.5)`) | random_uniform | random_normal | random_joint |
|---|---:|---:|---:|---:|---:|
| bucket_acc (↑) | 0.6076 | 0.0753 | 0.0190 | 0.0773 | 0.2627 |
| bucket_acc±1 (↑) | 0.8346 | 0.1740 | 0.0629 | 0.2403 | 0.4748 |
| bucket_acc±2 (↑) | 0.8851 | 0.2803 | 0.1311 | 0.3787 | 0.6687 |
| MAE (↓) | 66.385 | 831.388 | 2190.896 | 207.216 | 142.824 |
| RMSE (↓) | 181.223 | 1398.373 | 2635.763 | 336.422 | 344.288 |
| MAPE (↓) | 1.4011 | 54.0629 | 274.2940 | 21.3867 | 7.4801 |

### 6.2 不同窗口长度对比

下表为不同 `window_seconds` 的 overall 指标（`seed=42`）。baseline 不依赖窗口，因此 baseline 列在各行相同；为了对比直观，仍在表格中展开。

| window_seconds | pred bucket_acc | base bucket_acc | Δ | pred bucket_acc±1 | base bucket_acc±1 | Δ | pred bucket_acc±2 | base bucket_acc±2 | Δ | pred MAE | base MAE | MAE↓ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 0.6062 | 0.0753 | +0.5309 | 0.8267 | 0.1740 | +0.6527 | 0.8742 | 0.2803 | +0.5939 | 67.041 | 831.388 | 764.347 ↓ |
| 60 | 0.6076 | 0.0753 | +0.5323 | 0.8346 | 0.1740 | +0.6606 | 0.8851 | 0.2803 | +0.6048 | 66.385 | 831.388 | 765.003 ↓ |
| 120 | 0.6071 | 0.0753 | +0.5318 | 0.8410 | 0.1740 | +0.6670 | 0.8949 | 0.2803 | +0.6146 | 65.760 | 831.388 | 765.628 ↓ |
| 300 | 0.6041 | 0.0753 | +0.5288 | 0.8453 | 0.1740 | +0.6713 | 0.9047 | 0.2803 | +0.6244 | 65.930 | 831.388 | 765.458 ↓ |
| 600 | 0.5998 | 0.0753 | +0.5245 | 0.8456 | 0.1740 | +0.6716 | 0.9094 | 0.2803 | +0.6291 | 66.546 | 831.388 | 764.842 ↓ |
| 1800 | 0.5858 | 0.0753 | +0.5105 | 0.8352 | 0.1740 | +0.6612 | 0.9065 | 0.2803 | +0.6262 | 69.440 | 831.388 | 761.948 ↓ |
| 3600 | 0.5723 | 0.0753 | +0.4970 | 0.8213 | 0.1740 | +0.6473 | 0.8974 | 0.2803 | +0.6171 | 72.394 | 831.388 | 758.994 ↓ |

观察：

- `bucket_acc` 在 30~120 秒附近最好（约 0.606~0.608）；窗口继续增大时，exact bucket 命中率逐步下降。
- 但 `bucket_acc±2` 在 300~1800 秒仍较高（约 0.905~0.909），说明更长窗口下预测更“稳定/保守”，但更难完全命中同一桶。
- baseline 的 bucket 命中率极低（0.0753），说明 `ceil(input * 1.5)` 在该数据上**系统性失配**。

### 6.3 结论（用于 vtc-pred proposal）

- `SimpleOutputPredictor` 并非“高精度连续预测器”，其输出是桶代表值且存在冷启动偏乐观。
- 即便如此，在 BurstGPT 这样包含重尾输出的轨迹数据上，它仍然在桶级别显著优于 `ceil(input * 1.5)`。
- 在随机点验基线上（uniform/normal/joint），predictor 仍明显更优，说明其性能提升不是“随机波动”造成。
- 因此，在 VTC 成本估计中引入该 predictor（vtc-pred）是有意义的：至少在“识别不同输出长度等级”的维度上，它能显著减少规则估计带来的系统性偏差。

## 7. 后续工作建议

- 补充多随机种子（例如 5~10 个 seed）统计均值/方差，量化随机采样带来的波动。
- 结合真实线上 tokenizer（如有）重新评估 `promptLen` 的估计误差对结果的影响。
- 将 bucket 指标映射到 VTC 的实际收益：例如用该预测替换 vtc-basic 的输出估计后，对 fairness / tail latency 的影响。
