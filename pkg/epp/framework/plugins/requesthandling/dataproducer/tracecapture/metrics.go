/*
Copyright 2026 The llm-d Authors.

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

package tracecapture

import (
	"errors"
	"fmt"

	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	eppmetrics "github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	dropReasonBufferFull = "buffer_full"
	dropReasonSinkClosed = "sink_closed"
	dropReasonWriteError = "write_error"

	skipReasonSampledOut   = "sampled_out"
	skipReasonNoTokens     = "no_tokens"
	skipReasonNonCompleted = "non_completed"
)

var (
	traceCaptureRecordsWritten = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "trace_capture_records_written_total",
			Help:      metricsutil.HelpMsgWithStability("Captured request trace records persisted to JSONL files.", compbasemetrics.ALPHA),
		},
		[]string{"producer_name"},
	)

	traceCaptureRecordsDropped = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "trace_capture_records_dropped_total",
			Help:      metricsutil.HelpMsgWithStability("Captured request trace records dropped before persistence, by reason.", compbasemetrics.ALPHA),
		},
		[]string{"producer_name", "reason"},
	)

	traceCaptureRequestsSkipped = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "trace_capture_requests_skipped_total",
			Help:      metricsutil.HelpMsgWithStability("Requests skipped by trace capture, by reason.", compbasemetrics.ALPHA),
		},
		[]string{"producer_name", "reason"},
	)

	traceCaptureRotations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem: eppmetrics.LLMDRouterEndpointPickerSubsystem,
			Name:      "trace_capture_rotations_total",
			Help:      metricsutil.HelpMsgWithStability("Output JSONL file rotations performed by trace capture.", compbasemetrics.ALPHA),
		},
		[]string{"producer_name"},
	)
)

func registerMetrics(registerer prometheus.Registerer) error {
	if registerer == nil {
		return nil
	}
	collectors := []prometheus.Collector{
		traceCaptureRecordsWritten,
		traceCaptureRecordsDropped,
		traceCaptureRequestsSkipped,
		traceCaptureRotations,
	}
	for _, collector := range collectors {
		if err := registerer.Register(collector); err != nil {
			var alreadyRegistered prometheus.AlreadyRegisteredError
			if errors.As(err, &alreadyRegistered) && alreadyRegistered.ExistingCollector == collector {
				continue
			}
			return fmt.Errorf("register trace capture metric: %w", err)
		}
	}
	return nil
}
