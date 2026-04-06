/*
Copyright 2025 The Aibrix Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vtc

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/vllm-project/aibrix/pkg/cache"
	"github.com/vllm-project/aibrix/pkg/metrics"
	"github.com/vllm-project/aibrix/pkg/types"
	"github.com/vllm-project/aibrix/pkg/utils"
	v1 "k8s.io/api/core/v1"
	"k8s.io/klog/v2"
)

const (
	defaultMaxPodLoad        = 100.0
	defaultInputTokenWeight  = 1.0
	defaultOutputTokenWeight = 2.0
	defaultFairnessWeight    = 1.0
	defaultUtilizationWeight = 1.0
)

const (
	VTC_MAX_POD_LOAD        = "AIBRIX_ROUTER_VTC_BASIC_MAX_POD_LOAD"
	VTC_INPUT_TOKEN_WEIGHT  = "AIBRIX_ROUTER_VTC_BASIC_INPUT_TOKEN_WEIGHT"
	VTC_OUTPUT_TOKEN_WEIGHT = "AIBRIX_ROUTER_VTC_BASIC_OUTPUT_TOKEN_WEIGHT"
	VTC_FAIRNESS_WEIGHT     = "AIBRIX_ROUTER_VTC_BASIC_FAIRNESS_WEIGHT"
	VTC_UTILIZATION_WEIGHT  = "AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_WEIGHT"
	// Optional: use (running + waiting) as pod load to make queueing visible to VTC.
	// Default is false to keep backward compatibility.
	VTC_UTILIZATION_INCLUDE_WAITING = "AIBRIX_ROUTER_VTC_BASIC_UTILIZATION_INCLUDE_WAITING"
)

var (
	maxPodLoad        = utils.LoadEnvFloat(VTC_MAX_POD_LOAD, defaultMaxPodLoad)
	inputTokenWeight  = utils.LoadEnvFloat(VTC_INPUT_TOKEN_WEIGHT, defaultInputTokenWeight)
	outputTokenWeight = utils.LoadEnvFloat(VTC_OUTPUT_TOKEN_WEIGHT, defaultOutputTokenWeight)
	fairnessWeight    = utils.LoadEnvFloat(VTC_FAIRNESS_WEIGHT, defaultFairnessWeight)
	utilizationWeight = utils.LoadEnvFloat(VTC_UTILIZATION_WEIGHT, defaultUtilizationWeight)
	utilizationIncludeWaiting = utils.LoadEnvBool(VTC_UTILIZATION_INCLUDE_WAITING, false)
)

// BasicVTCRouter implements the VTC routing algorithm
type BasicVTCRouter struct {
	cache                   cache.MetricCache
	outputPredictorProvider types.OutputPredictorProvider
	tokenTracker            TokenTracker
	tokenEstimator          TokenEstimator
	config                  *VTCConfig
}

// NewBasicVTCRouter creates a new BasicVTCRouter with the provided token tracker and estimator
func NewBasicVTCRouter(tokenTracker TokenTracker, tokenEstimator TokenEstimator, config *VTCConfig) (*BasicVTCRouter, error) {
	c, err := cache.Get()
	if err != nil {
		klog.Error("fail to get cache store in basic-vtc router")
		return nil, err
	}

	return &BasicVTCRouter{
		cache:                   c,
		outputPredictorProvider: c,
		tokenTracker:            tokenTracker,
		tokenEstimator:          tokenEstimator,
		config:                  config,
	}, nil
}

// Route implements the VTC routing algorithm
func (r *BasicVTCRouter) Route(ctx *types.RoutingContext, readyPodList types.PodList) (string, error) {
	readyPods := readyPodList.All()
	user := ctx.User
	if user == nil {
		klog.Warningf("VTC routing not possible: user is nil, falling back to random pod selection")
		randomPod, err := utils.SelectRandomPod(readyPods, rand.Intn)
		if err != nil {
			return "", fmt.Errorf("fallback to random pod selection failed: %w", err)
		}
		ctx.SetTargetPod(randomPod)
		return ctx.TargetAddress(), nil
	}

	inputTokens := r.tokenEstimator.EstimateInputTokens(ctx.Message)
	outputTokens := r.tokenEstimator.EstimateOutputTokens(ctx.Message)
	if r.config != nil && r.config.Variant == RouterVTCPred {
		promptLen, err := ctx.PromptLength()
		if err != nil {
			klog.ErrorS(err, "failed to get prompt length for vtc-pred")
		} else {
			inputTokens = float64(promptLen)
			if predicted, ok := r.predictOutputTokens(ctx, promptLen); ok {
				outputTokens = predicted
			}
		}
	}

	userTokens, err := r.tokenTracker.GetTokenCount(ctx.Context, *user)
	if err != nil {
		klog.ErrorS(err, "failed to get user token count, falling back to zero", "user", *user)
		userTokens = 0
	}

	klog.InfoS("VTC tokens for user",
		"user", *user,
		"tokens", userTokens,
		"inputTokens", inputTokens,
		"outputTokens", outputTokens)

	var targetPod *v1.Pod
	var minScore float64 = math.MaxFloat64

	// Simple vtc-basic implementation
	// Using clamped-linear instead of modulo - offers good monotonicity, fairness based routing.
	// Pod utilization is used as a secondary metric to ensure good utilization.
	// By adapting bucket sizes and normalizing scores, the algorithm remains robust as system load and user activity fluctuate.

	// Get the min and max token counts for adaptive bucket sizing
	minTokens, err := r.tokenTracker.GetMinTokenCount(ctx.Context)
	if err != nil {
		klog.ErrorS(err, "failed to get minimum token count, using default value")
		minTokens = tokenTrackerMinTokens // Use the configured default minimum token count
	}

	maxTokens, err := r.tokenTracker.GetMaxTokenCount(ctx.Context)
	if err != nil {
		klog.ErrorS(err, "failed to get maximum token count, using default value")
		maxTokens = tokenTrackerMaxTokens // Use the configured default maximum token count
	}

	// Calculate scores for each pod
	for i, pod := range readyPods {

		// 1. Dynamically calculate a reasonable "step size" for mapping user tokens onto pod indices, ensuring the mapping is
		// relevant to the current system load while maintaining a minimum sensitivity
		adaptiveBucketSize := math.Max(tokenTrackerMinTokens, (minTokens+maxTokens)/2)

		metrics.SetGaugeMetric(
			metrics.VTCBucketSizeActive,
			metrics.GetMetricHelp(metrics.VTCBucketSizeActive),
			adaptiveBucketSize,
			[]string{"pod", "model"},
			pod.Name, ctx.Model,
		)

		// Apply wrapped linear mapping: tokens / bucket_size, wrapped to [0, npods)
		//
		// NOTE: The previous clamped mapping would saturate at (npods-1) once userTokens grows large,
		// causing many users to be routed to the last pod and amplifying queueing. Wrapping keeps the
		// monotonic progression while preventing permanent saturation.
		normalizedTokens := 0.0
		if adaptiveBucketSize > 0 && len(readyPods) > 0 {
			normalizedTokens = math.Mod(float64(userTokens)/adaptiveBucketSize, float64(len(readyPods)))
			if normalizedTokens < 0 {
				normalizedTokens += float64(len(readyPods))
			}
		}

		// Circular distance on the ring: min(|i-x|, n-|i-x|)
		n := float64(len(readyPods))
		diff := math.Abs(float64(i) - normalizedTokens)
		fairnessScore := diff
		if n > 0 {
			fairnessScore = math.Min(diff, n-diff)
		}

		klog.InfoS("VTC token normalization details",
			"user", *user,
			"userTokens", userTokens,
			"minTokens", minTokens,
			"maxTokens", maxTokens,
			"adaptiveBucketSize", adaptiveBucketSize,
			"normalizedTokens", normalizedTokens,
			"podIndex", i,
			"fairnessScore", fairnessScore)

		// 2. Get pod load for utilization score
		// Default: running requests. Optionally include waiting requests to reflect queueing.
		var podLoad float64
		if r.cache != nil {
			reqCount, err := r.cache.GetMetricValueByPodModel(pod.Name, pod.Namespace, ctx.Model, metrics.NumRequestsRunning)
			if err != nil {
				klog.ErrorS(err, "failed to get pod metrics, using default value", "pod", pod.Name)
				podLoad = 0
			} else {
				podLoad = reqCount.GetSimpleValue()
			}
			if utilizationIncludeWaiting {
				waitingCount, err := r.cache.GetMetricValueByPodModel(pod.Name, pod.Namespace, ctx.Model, metrics.NumRequestsWaiting)
				if err != nil {
					klog.ErrorS(err, "failed to get pod waiting metrics, ignoring", "pod", pod.Name)
				} else {
					podLoad += waitingCount.GetSimpleValue()
				}
			}
		} else {
			klog.Info("Cache is nil, using default pod load value")
			podLoad = 0
		}

		// 3. Calculate utilization score (normalized between 0-1)
		utilizationScore := min(podLoad/maxPodLoad, 1.0)

		// 4. Add a small random factor to break ties and improve distribution
		randomFactor := rand.Float64() * 0.1

		// 5. Calculate combined score (lower is better) - using configurable weights for fairness and utilization
		score := (fairnessWeight * fairnessScore) + (utilizationWeight * utilizationScore) + randomFactor

		klog.InfoS("VTC hybrid pod selection",
			"pod", pod.Name,
			"podIndex", i,
			"userTokens", userTokens,
			"podLoad", podLoad,
			"fairnessScore", fairnessScore,
			"fairnessWeight", fairnessWeight,
			"utilizationScore", utilizationScore,
			"utilizationWeight", utilizationWeight,
			"combinedScore", score)

		if score < minScore {
			minScore = score
			targetPod = pod
		}
	}

	if targetPod == nil {
		klog.Warning("No pods with valid metrics found or all pods scored equally; selecting a pod randomly as fallback")
		var err error
		targetPod, err = utils.SelectRandomPod(readyPods, rand.Intn)
		if err != nil {
			return "", fmt.Errorf("random fallback selection failed: %w", err)
		}
	}

	if *user != "" {
		err := r.tokenTracker.UpdateTokenCount(ctx.Context, *user, inputTokens, outputTokens)
		if err != nil {
			klog.ErrorS(err, "failed to update user token count", "user", *user)
		}
	}

	ctx.SetTargetPod(targetPod)
	return ctx.TargetAddress(), nil
}

func (r *BasicVTCRouter) SubscribedMetrics() []string {
	if utilizationIncludeWaiting {
		return []string{
			metrics.NumRequestsRunning,
			metrics.NumRequestsWaiting,
			metrics.VTCBucketSizeActive,
		}
	}
	return []string{
		metrics.NumRequestsRunning,
		metrics.VTCBucketSizeActive,
	}
}

func (r *BasicVTCRouter) predictOutputTokens(ctx *types.RoutingContext, promptLen int) (float64, bool) {
	if r.outputPredictorProvider == nil {
		return 0, false
	}
	predictor, err := r.outputPredictorProvider.GetOutputPredictor(ctx.Model)
	if err != nil {
		klog.ErrorS(err, "failed to get output predictor for vtc-pred", "model", ctx.Model)
		return 0, false
	}
	ctx.SetOutputPreditor(predictor)
	return float64(predictor.Predict(promptLen)), true
}
