# Aibrix vtc-pred 公平路由方案开题 Proposal

## 摘要

本项目面向 Aibrix 网关路由中的 VTC（Virtual Token Counter）策略，提出一种基于输出长度预测的公平路由变体 vtc-pred。当前 vtc-basic 使用简单的字符长度估计与固定输出比例来近似请求成本，在输出长度分布重尾（LLM 常见）场景下会系统性低估长回答请求，进而导致公平性波动、长尾排队放大、资源争抢不稳定。Aibrix 内部已有 OutputPredictor 组件与完整的历史回填数据路径，但尚未用于 VTC。本文提出将 OutputPredictor 接入 VTC 的成本估计，使输出 token 估计由“规则估计”升级为“基于历史分布的预测”，在不改变 VTC 核心策略的前提下提升公平性与鲁棒性。

## 1. 背景与动机

### 1.1 LLM Serving 的核心资源消耗

LLM 请求通常拆分为两个阶段：

- **Prefill**：处理输入 prompt，计算 KV cache。
- **Decode**：逐 token 生成输出，时间和显存压力与输出 token 数直接相关。

在实际线上流量中，输出长度往往呈现重尾分布：大多数请求输出较短，但少量请求输出非常长且占用大量 decode 时间。这类长尾请求如果被低估，会在公平调度中“账面成本”偏低，从而更容易再次获得调度机会，破坏公平性。

### 1.2 公平路由的目标

VTC 的目标是减少不同用户之间的资源占用不均衡，让“已经消耗大量 token 的用户”在调度中得到更低的优先级，从而提高公平性。该目标的关键依赖**每个请求的成本估计是否准确**。

## 2. 现有系统与代码现状

### 2.1 vtc-basic 的实现逻辑

vtc-basic 的核心逻辑在 [vtc_basic.go](file:///Users/bytedance/cuhk/aibrix/pkg/plugins/gateway/algorithms/vtc/vtc_basic.go)：

- 从 TokenEstimator 获取输入与输出 token 估计值；
- 从 TokenTracker 获取用户历史 token 消耗；
- 将公平性得分与 Pod 利用率得分组合，选择最小分数的 Pod。

当前 TokenEstimator 是 SimpleTokenEstimator（[token_estimator.go](file:///Users/bytedance/cuhk/aibrix/pkg/plugins/gateway/algorithms/vtc/token_estimator.go)），规则非常简单：

- **输入 token** ≈ `len(message) / 4`
- **输出 token** ≈ `输入 token * 1.5`

这是一种静态估计，忽略了模型输出长度的动态分布与场景差异。

### 2.2 OutputPredictor 已存在但未被 VTC 使用

Aibrix 已经实现了 OutputPredictor，并在请求完成时持续回填真实 token：

- OutputPredictor 定义：[output_predictor.go](file:///Users/bytedance/cuhk/aibrix/pkg/cache/output_predictor.go)
- 在模型缓存初始化时创建 predictor：[informers.go](file:///Users/bytedance/cuhk/aibrix/pkg/cache/informers.go#L309-L326)
- 请求完成时回填历史数据：[cache_impl.go](file:///Users/bytedance/cuhk/aibrix/pkg/cache/cache_impl.go#L246-L273)

该预测器已经被 SLO 队列使用（[slo_queue.go](file:///Users/bytedance/cuhk/aibrix/pkg/plugins/gateway/queue/slo_queue.go#L111-L135)），说明它是稳定可用的能力，但 VTC 目前未接入。

调研实验也证明了 OutputPredictor 能带来预测精度的提升：
[vtc-pred调研 BurstGPT 输出长度预测](https://docs.qq.com/markdown/DZHZibnVqak1DRXpF?nlc=1) 

### 2.3 代码里明确指出，之后需要支持 vtc-pred

pkg/plugins/gateway/algorithms/vtc/vtc_router.go
![image](https://docimg3.docs.qq.com/image/AgAABW3Puj05Y5Tj6y1JxqO55sJc99qb.png?w=1914&h=1064)

## 3. 现有 vtc-basic 的不足

### 3.1 输出长度估计偏差

固定比例估计无法反映真实分布，尤其在以下场景容易失真：

- **长回答或多轮对话**：输出 token 可能远高于输入 token；
- **复杂指令或思维链**：输出长度与输入长度相关性弱；
- **重尾分布**：少数长回答显著拉高平均负载。

### 3.2 公平性波动

VTC 依赖估计的 token 成本来更新用户的“账面贡献”。当长尾请求被低估时：

- 用户实际占用更多 decode 时间，却显示“贡献较小”；
- 在公平排序中更容易继续获得资源；
- 对短请求用户产生不公平挤压。

### 3.3 对调度鲁棒性不利

在高负载情况下，错误的输出估计会放大排队效应和系统波动，导致 tail latency 更高、吞吐更不稳定。

## 4. 改进方案：vtc-pred

### 4.1 核心思想

将 VTC 的输出 token 估计从“规则估计”升级为“历史分布预测”，保持 VTC 算法结构不变，只替换成本估计来源。

目标公式：

```
cost = w_in * input_tokens + w_out * predicted_output_tokens
```

其中 `predicted_output_tokens` 使用 OutputPredictor 预测。

### 4.2 技术方案

- **输入 token**：使用已有的 PromptLength（真实 tokenizer 长度）；
- **输出 token**：通过 cache.GetOutputPredictor(model) 获取 predictor，调用 Predict(promptLen)；
- **回退机制**：若预测器不可用，则回退到 SimpleTokenEstimator。

这样可以在冷启动或异常情况下保持稳定性，同时逐步利用历史数据提升准确性。

### 4.3 方案优势

- **改动小、收益大**：只替换估计器，VTC 核心逻辑保持一致；
- **已有数据链路**：OutputPredictor 已在系统中维护历史；
- **公平性更稳**：在重尾场景下减少长尾请求的低估。

## 5. 方案示例

假设两个请求：

- 请求 A：输入 100 tokens，输出 120 tokens
- 请求 B：输入 100 tokens，输出 1200 tokens（长尾）

**vtc-basic 估计**：

- A：输出估计 ≈ 150
- B：输出估计 ≈ 150（被严重低估）

**vtc-pred 估计**：

- A：预测输出 ≈ 120
- B：预测输出 ≈ 1000+（靠近真实）

公平性上，vtc-pred 能正确识别 B 的高成本，避免其“账面贡献偏小”。

## 6. 实验设计

### 6.1 实验对比

对比以下策略：

- vtc-basic
- vtc-pred
- random 或 least-request（作为基线）

### 6.2 数据集与流量模式

- 使用 ShareGPT / Vicuna 类聊天数据集；
- 构造不同负载与分布模式：Balanced / Bursty / High Usage；
- 模拟重尾输出分布。

### 6.3 评估指标

- **公平性指标**：Jain’s fairness index（越接近 1 越公平）
- **Tail latency**：P95 / P99 延迟
- **吞吐**：tokens/s 或 requests/s
- **稳定性**：不同用户的调度波动性

## 7. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 冷启动时预测器缺乏历史 | 预测偏差 | 回退到规则估计 |
| 训练数据与线上分布不一致 | 预测失准 | 滑动窗口 + 动态更新 |
| 输出预测波动 | 可能引入不稳定 | 统计平滑 + 预测上限 |



## 10. 总结

vtc-pred 的核心价值在于利用 Aibrix 既有的 OutputPredictor 能力，修正 vtc-basic 在重尾场景下的成本低估问题，在公平性与性能稳定性上获得确定性收益。该方案改动小、可落地性强，适合作为课程项目的工程型研究方向。