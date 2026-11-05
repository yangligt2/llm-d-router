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
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionid"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
)

func TestPluginFactory_ConfigValidation(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	cases := []struct {
		name    string
		rawJSON string
		wantErr string
	}{
		{
			name:    "missing parameters",
			rawJSON: `{}`,
			wantErr: "outputDir",
		},
		{
			name:    "negative blockSizeTokens",
			rawJSON: `{"outputDir": "` + dir + `", "blockSizeTokens": -16}`,
			wantErr: "blockSizeTokens",
		},
		{
			name:    "sampleRate above 1",
			rawJSON: `{"outputDir": "` + dir + `", "sampleRate": 1.5}`,
			wantErr: "sampleRate",
		},
		{
			name:    "sampleRate negative",
			rawJSON: `{"outputDir": "` + dir + `", "sampleRate": -0.1}`,
			wantErr: "sampleRate",
		},
		{
			name:    "invalid session source",
			rawJSON: `{"outputDir": "` + dir + `", "sessionSources": [{"header": "a", "attribute": "b"}]}`,
			wantErr: "sessionSources",
		},
		{
			name:    "unknown field rejected by strict decoder",
			rawJSON: `{"outputDir": "` + dir + `", "nonExistentField": true}`,
			wantErr: "nonExistentField",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := PluginFactory("capture", fwkplugin.StrictDecoder(json.RawMessage(tc.rawJSON)), nil)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.wantErr)
		})
	}
}

type stubTokenProducer struct {
	backend string
}

func (s *stubTokenProducer) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: tokenizer.PluginType, Name: tokenizer.PluginType}
}

func (s *stubTokenProducer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		tokenizer.TokenizedPromptDataKey: fwkplugin.ProducedData{
			Type:       fwkrequest.TokenizedRequest{},
			Attributes: map[string]string{"backend": s.backend},
		},
	}
}

func (s *stubTokenProducer) Produce(_ context.Context, _ *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	return nil
}

func TestConsumes_VLLMBackendEnforcementAndOptionalSessionSources(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	rawCfg, err := json.Marshal(map[string]any{"outputDir": dir})
	require.NoError(t, err)

	plg, err := PluginFactory("capture", fwkplugin.StrictDecoder(rawCfg), nil)
	require.NoError(t, err)
	producer := plg.(*Producer)
	t.Cleanup(func() { _ = producer.Close() })

	// 1. Estimate backend must be rejected by default via ValidateAndOrderDataDependencies.
	_, err = datalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{
		&stubTokenProducer{backend: tokenizer.BackendEstimate},
		producer,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), tokenizer.BackendVLLM)

	// 2. vLLM backend succeeds AND CreateMissingDataProducers must NOT attempt to auto-create
	// session-id-producer (which fails on nil parameters) because session/subagent attributes are Optional.
	defaultReg := map[string]string{
		fwkplugin.NewDataKey("SessionID", sessionid.SessionIDProducerType).String(): sessionid.SessionIDProducerType,
	}
	factoryReg := map[string]fwkplugin.FactoryFunc{
		sessionid.SessionIDProducerType: sessionid.Factory,
	}

	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	vllmTok := &stubTokenProducer{backend: tokenizer.BackendVLLM}
	handle.AddPlugin(vllmTok.TypedName().Name, vllmTok)
	handle.AddPlugin(producer.TypedName().Name, producer)

	require.NoError(t, datalayer.CreateMissingDataProducers(context.Background(), defaultReg, factoryReg, handle))
	ordered, err := datalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins())
	require.NoError(t, err)
	assert.Equal(t, []string{vllmTok.TypedName().String(), producer.TypedName().String()}, ordered)

	// 3. allowEstimateBackend: true permits estimate backend for offline testing.
	rawAllow, err := json.Marshal(map[string]any{"outputDir": t.TempDir(), "allowEstimateBackend": true})
	require.NoError(t, err)
	plgAllow, err := PluginFactory("capture-est", fwkplugin.StrictDecoder(rawAllow), nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = plgAllow.(*Producer).Close() })

	_, err = datalayer.ValidateAndOrderDataDependencies([]fwkplugin.Plugin{
		&stubTokenProducer{backend: tokenizer.BackendEstimate},
		plgAllow,
	})
	require.NoError(t, err)
}

func TestProducer_LifecycleStreamingNonStreamingErrorsAndRedaction(t *testing.T) {
	t.Parallel()

	outDir := t.TempDir()
	reg := prometheus.NewRegistry()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	handle := fwkplugin.NewEppHandle(ctx, nil, fwkplugin.WithMetricsRecorder(reg))
	rawCfg, err := json.Marshal(map[string]any{
		"outputDir":       outDir,
		"blockSizeTokens": 64,
	})
	require.NoError(t, err)

	plg, err := PluginFactory("capture", fwkplugin.StrictDecoder(rawCfg), handle)
	require.NoError(t, err)
	producer := plg.(*Producer)

	maxTok := int64(512)
	baseTime := time.Unix(1_700_000_000, 0)

	// Request 1: Streaming request with tools, x-dynamo-session-id, usage omitted -> falls back to StreamedEvents.
	req1 := &fwksched.InferenceRequest{
		RequestID:   "uuid-req-1",
		TargetModel: "Qwen/Qwen2.5-7B-Instruct",
		Headers: map[string]string{
			"x-dynamo-session-id": "secret-user-session-alpha",
		},
		Body: &fwkrequest.InferenceRequestBody{
			Model:           "Qwen/Qwen2.5-7B-Instruct",
			Stream:          true,
			MaxOutputTokens: &maxTok,
			ChatCompletions: &fwkrequest.ChatCompletionsRequest{
				Tools: []any{map[string]any{"type": "function"}},
			},
			TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{makeSeqTokens(1, 150)}),
		},
	}
	require.NoError(t, producer.Produce(ctx, req1, nil))

	// Check DumpState while req1 is in flight: zero raw secrets or tokens exposed.
	dumpRaw, err := producer.DumpState()
	require.NoError(t, err)
	assert.NotContains(t, string(dumpRaw), "secret-user-session-alpha")
	var ds debugState
	require.NoError(t, json.Unmarshal(dumpRaw, &ds))
	assert.Equal(t, int64(1), ds.InFlight)

	// Simulate handler releasing request body before response phase.
	req1.Body = req1.Body.ResponsePhaseView()

	producer.ResponseHeader(ctx, req1, &fwkrh.Response{
		RequestID: req1.RequestID,
		Headers:   map[string]string{reqcommon.PseudoHeaderStatus: "200"},
	}, nil)
	// Intermediate streaming chunk (EndOfStream=false) must not finalize state.
	producer.ResponseBody(ctx, req1, &fwkrh.Response{
		RequestID:      req1.RequestID,
		EndOfStream:    false,
		Streaming:      true,
		StreamedEvents: 10,
	}, nil)
	producer.ResponseBody(ctx, req1, &fwkrh.Response{
		RequestID:           req1.RequestID,
		EndOfStream:         true,
		Streaming:           true,
		StreamedEvents:      45,
		TerminationCause:    fwkrh.TerminationCauseNatural,
		ReceivedTimestamp:   baseTime,
		FirstTokenTimestamp: baseTime.Add(120 * time.Millisecond),
		CompleteTimestamp:   baseTime.Add(920 * time.Millisecond),
	}, nil)

	// Request 2: Non-streaming subagent request referencing parent session + agent-identity attribute.
	req2 := &fwksched.InferenceRequest{
		RequestID:   "uuid-req-2",
		TargetModel: "Qwen/Qwen2.5-7B-Instruct",
		Headers: map[string]string{
			"x-dynamo-parent-session-id": "secret-user-session-alpha",
			"x-claude-code-agent-id":     "explore-subagent-1",
		},
		Body: &fwkrequest.InferenceRequestBody{
			Model:            "Qwen/Qwen2.5-7B-Instruct",
			Stream:           false,
			TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{makeSeqTokens(500, 80)}),
		},
	}
	req2.PutAttribute(agentidentity.AgentIdentityKey, "subagent-own-session")
	require.NoError(t, producer.Produce(ctx, req2, nil))
	req2.Body = req2.Body.ResponsePhaseView()

	producer.ResponseHeader(ctx, req2, &fwkrh.Response{
		RequestID: req2.RequestID,
		Headers:   map[string]string{reqcommon.PseudoHeaderStatus: "200"},
	}, nil)
	producer.ResponseBody(ctx, req2, &fwkrh.Response{
		RequestID:         req2.RequestID,
		EndOfStream:       true,
		Streaming:         false,
		TerminationCause:  fwkrh.TerminationCauseNatural,
		ReceivedTimestamp: baseTime.Add(2 * time.Second),
		CompleteTimestamp: baseTime.Add(2500 * time.Millisecond),
		Usage: fwkrequest.Usage{
			PromptTokens:       80,
			CompletionTokens:   19,
			PromptTokenDetails: &fwkrequest.PromptTokenDetails{CachedTokens: 64},
		},
	}, nil)

	// Request 3: Upstream HTTP 500 error (Natural stream end, status 500) -> skipped by default.
	req3 := &fwksched.InferenceRequest{
		RequestID:   "uuid-req-3",
		TargetModel: "Qwen/Qwen2.5-7B-Instruct",
		Headers:     map[string]string{"x-session-id": "err-sess"},
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{makeSeqTokens(1, 128)}),
		},
	}
	require.NoError(t, producer.Produce(ctx, req3, nil))
	producer.ResponseHeader(ctx, req3, &fwkrh.Response{
		RequestID: req3.RequestID,
		Headers:   map[string]string{reqcommon.PseudoHeaderStatus: "500"},
	}, nil)
	producer.ResponseBody(ctx, req3, &fwkrh.Response{
		RequestID:        req3.RequestID,
		EndOfStream:      true,
		TerminationCause: fwkrh.TerminationCauseNatural,
	}, nil)

	// Request 4: Abandoned before PreRequest -> cleaned up immediately.
	req4 := &fwksched.InferenceRequest{
		RequestID:   "uuid-req-4",
		TargetModel: "Qwen/Qwen2.5-7B-Instruct",
		Body: &fwkrequest.InferenceRequestBody{
			TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{makeSeqTokens(1, 128)}),
		},
	}
	require.NoError(t, producer.Produce(ctx, req4, nil))
	producer.RequestAbandoned(ctx, req4)

	require.NoError(t, producer.Close())

	dumpAfter, err := producer.DumpState()
	require.NoError(t, err)
	require.NoError(t, json.Unmarshal(dumpAfter, &ds))
	assert.Equal(t, int64(0), ds.InFlight)
	assert.Equal(t, int64(2), ds.RecordsWritten)

	matches, err := filepath.Glob(filepath.Join(outDir, "capture-*.jsonl"))
	require.NoError(t, err)
	records, err := LoadCapturedRecords(matches)
	require.NoError(t, err)
	require.Len(t, records, 2)

	r1 := records[0]
	assert.Equal(t, "s", r1.StreamType)
	assert.Equal(t, 150, r1.TokenizedIn)
	assert.Equal(t, 45, r1.OutputTokens)
	assert.True(t, r1.HasTools)
	require.NotNil(t, r1.MaxOutputTokens)
	assert.Equal(t, int64(512), *r1.MaxOutputTokens)
	assert.InDelta(t, 0.12, r1.FirstTokenSeconds, 1e-6)
	assert.InDelta(t, 0.92, r1.APITimeSeconds, 1e-6)
	assert.Len(t, r1.BlockHashes, 2)
	assert.NotEmpty(t, r1.SessionID)
	assert.NotEqual(t, "secret-user-session-alpha", r1.SessionID)

	r2 := records[1]
	assert.Equal(t, "n", r2.StreamType)
	assert.Zero(t, r2.FirstTokenSeconds)
	assert.Equal(t, 80, r2.TokenizedIn)
	assert.Equal(t, 80, r2.UsageIn)
	assert.Equal(t, 64, r2.CachedTokens)
	assert.Equal(t, 19, r2.OutputTokens)
	// Subagent's ParentSessionID matches r1's SessionID via shared HMAC domain!
	assert.Equal(t, r1.SessionID, r2.ParentSessionID)
	assert.NotEmpty(t, r2.SubagentID)
}
