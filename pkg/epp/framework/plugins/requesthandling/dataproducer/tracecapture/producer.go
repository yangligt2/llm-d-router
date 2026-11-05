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
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
)

const (
	stateKey fwkplugin.StateKey = PluginType

	outcomeCompleted = "completed"
	outcomeEvicted   = "evicted"
	outcomeCancelled = "cancelled"
	outcomeFailed    = "failed"

	streamTypeStreaming      = "s"
	streamTypeNormal         = "n"
	subagentSessionDelimiter = "::sa:"
)

var (
	_ fwkrh.DataProducer            = (*Producer)(nil)
	_ fwkplugin.ConsumerPlugin      = (*Producer)(nil)
	_ fwkrh.RequestAbandoned        = (*Producer)(nil)
	_ fwkrh.ResponseHeaderProcessor = (*Producer)(nil)
	_ fwkrh.ResponseBodyProcessor   = (*Producer)(nil)
	_ fwkplugin.StateDumper         = (*Producer)(nil)
)

// Producer captures anonymized per-request KV-block hashes and response timing.
type Producer struct {
	typedName                 fwkplugin.TypedName
	cfg                       Config
	sampleRate                float64
	useDefaultSessionFallback bool

	hasher      *Hasher
	sink        *Sink
	pluginState *fwkplugin.PluginState
	inFlight    atomic.Int64
}

// PluginFactory instantiates a trace-capture-producer plugin from JSON configuration.
func PluginFactory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if name == "" {
		name = PluginType
	}
	var cfg Config
	if rawParameters != nil {
		if err := rawParameters.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of %q plugin %q: %w", PluginType, name, err)
		}
	}
	usedDefaultSession := cfg.ApplyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}

	ctx := context.Background()
	if handle != nil {
		if err := registerMetrics(handle.Metrics()); err != nil {
			return nil, err
		}
		if handle.Context() != nil {
			ctx = handle.Context()
		}
	}

	hasher, err := NewHasher()
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}

	sink, err := NewSink(ctx, SinkOptions{
		OutputDir:       cfg.OutputDir,
		RunID:           hasher.CaptureRunID(),
		BufferCapacity:  cfg.BufferCapacity,
		MaxFileBytes:    cfg.MaxFileBytes,
		MaxRotatedFiles: cfg.MaxRotatedFiles,
		FlushInterval:   time.Duration(cfg.FlushIntervalSeconds * float64(time.Second)),
		PluginName:      name,
	})
	if err != nil {
		return nil, fmt.Errorf("%s plugin %q: %w", PluginType, name, err)
	}

	return &Producer{
		typedName:                 fwkplugin.TypedName{Type: PluginType, Name: name},
		cfg:                       cfg,
		sampleRate:                *cfg.SampleRate,
		useDefaultSessionFallback: usedDefaultSession,
		hasher:                    hasher,
		sink:                      sink,
		pluginState:               fwkplugin.NewPluginState(ctx),
	}, nil
}

// TypedName returns the type and instance name of this plugin.
func (p *Producer) TypedName() fwkplugin.TypedName { return p.typedName }

// Produces returns nil because trace-capture-producer is a terminal observer.
func (p *Producer) Produces() map[fwkplugin.DataKey]any { return nil }

// Consumes requires TokenizedPrompt from token-producer (enforcing the vllm
// backend unless AllowEstimateBackend is set) and optionally consumes any
// request attributes referenced by session or subagent sources.
func (p *Producer) Consumes() fwkplugin.DataDependencies {
	var tokenDep any = fwkrequest.TokenizedRequest{}
	if !p.cfg.AllowEstimateBackend {
		tokenDep = fwkplugin.ConsumedData{
			Type:    fwkrequest.TokenizedRequest{},
			Matches: fwkplugin.MatchKeyValues(map[string]string{"backend": tokenizer.BackendVLLM}),
		}
	}

	optional := map[fwkplugin.DataKey]any{}
	mergeOptional := func(deps fwkplugin.DataDependencies) {
		for k, v := range deps.Optional {
			optional[k] = v
		}
	}
	mergeOptional(sessionaffinity.SourcesConsumes(p.cfg.SessionSources))
	mergeOptional(sessionaffinity.SourcesConsumes(p.cfg.ParentSessionSources))
	mergeOptional(sessionaffinity.SourcesConsumes(p.cfg.SubagentSources))

	deps := fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			tokenizer.TokenizedPromptDataKey: tokenDep,
		},
	}
	if len(optional) > 0 {
		deps.Optional = optional
	}
	return deps
}

// Produce hashes complete prompt blocks and stores anonymized request metadata
// in PluginState before the handler releases the parsed request body.
func (p *Producer) Produce(ctx context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil || request.Body == nil || request.RequestID == "" {
		return nil
	}

	rawSessionID := sessionaffinity.ResolveSessionID(ctx, request, p.cfg.SessionSources)
	if rawSessionID == "" && p.useDefaultSessionFallback &&
		request.FairnessID != "" && request.FairnessID != reqcommon.DefaultFairnessID {
		rawSessionID = request.FairnessID
	}
	rawParentSessionID := sessionaffinity.ResolveSessionID(ctx, request, p.cfg.ParentSessionSources)
	rawSubagentID := sessionaffinity.ResolveSessionID(ctx, request, p.cfg.SubagentSources)
	if rawParentSessionID == "" {
		if parent, sub := splitCompositeSubagentSessionID(rawSessionID); parent != "" {
			rawParentSessionID = parent
			if rawSubagentID == "" {
				rawSubagentID = sub
			}
		}
	}

	// Key sampling on root session when present so a sampled agent trace retains all subagents.
	sampleKey := rootSessionPrefix(rawParentSessionID)
	if sampleKey == "" {
		sampleKey = rootSessionPrefix(rawSessionID)
	}
	if sampleKey == "" {
		sampleKey = request.RequestID
	}
	if !p.hasher.ShouldSample(sampleKey, p.sampleRate) {
		traceCaptureRequestsSkipped.WithLabelValues(p.typedName.Name, skipReasonSampledOut).Inc()
		return nil
	}

	tr := request.Body.TokenizedRequest
	totalTokens := tr.TokenCount()
	if tr == nil || totalTokens == 0 {
		traceCaptureRequestsSkipped.WithLabelValues(p.typedName.Name, skipReasonNoTokens).Inc()
		return nil
	}

	blockHashes := p.hasher.HashTokenizedRequest(tr, p.cfg.BlockSizeTokens)

	var maxOutCopy *int64
	if request.Body.MaxOutputTokens != nil {
		v := *request.Body.MaxOutputTokens
		maxOutCopy = &v
	}

	model := request.TargetModel
	if model == "" {
		model = request.Body.Model
	}

	entry := &inFlightCapture{
		arrivalTime:     time.Now(),
		model:           model,
		tokenizedIn:     totalTokens,
		blockHashes:     blockHashes,
		sessionID:       p.hasher.AnonymizeSessionID(rawSessionID),
		parentSessionID: p.hasher.AnonymizeSessionID(rawParentSessionID),
		subagentID:      p.hasher.AnonymizeSubagentID(rawSubagentID),
		maxOutputTokens: maxOutCopy,
		hasTools:        hasRequestTools(request.Body),
		onEvict: func() {
			p.inFlight.Add(-1)
		},
	}

	if _, existed := p.pluginState.ReadOrWrite(request.RequestID, stateKey, entry); !existed {
		p.inFlight.Add(1)
	} else {
		p.pluginState.Write(request.RequestID, stateKey, entry)
	}
	p.pluginState.BindLiveness(ctx, request.RequestID)

	if v := log.FromContext(ctx).V(logging.TRACE); v.Enabled() {
		v.Info("Captured prompt block digests",
			"plugin", p.typedName.String(),
			"tokenizedIn", totalTokens,
			"blocks", len(blockHashes))
	}
	return nil
}

// RequestAbandoned releases per-request state when routing aborts before dispatch.
func (p *Producer) RequestAbandoned(_ context.Context, request *fwksched.InferenceRequest) {
	if request == nil || request.RequestID == "" {
		return
	}
	p.pluginState.Delete(request.RequestID)
}

// ResponseHeader records the upstream HTTP status code on the in-flight entry.
func (p *Producer) ResponseHeader(_ context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || request.RequestID == "" {
		return
	}
	entry, err := fwkplugin.ReadPluginStateKey[*inFlightCapture](p.pluginState, request.RequestID, stateKey)
	if err != nil || entry == nil {
		return
	}
	status := response.Headers[reqcommon.PseudoHeaderStatus]
	if status == "" {
		status = response.Headers[reqcommon.HeaderStatus]
	}
	entry.setStatus(status)
}

// ResponseBody finalizes the captured record at end of stream and submits it to the sink.
func (p *Producer) ResponseBody(_ context.Context, request *fwksched.InferenceRequest, response *fwkrh.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream || request.RequestID == "" {
		return
	}
	entry, err := fwkplugin.ReadPluginStateKey[*inFlightCapture](p.pluginState, request.RequestID, stateKey)
	p.pluginState.Delete(request.RequestID)
	if err != nil || entry == nil {
		return
	}

	outcome := classifyOutcome(entry.status(), response.TerminationCause)
	if outcome != outcomeCompleted && !p.cfg.IncludeNonCompleted {
		traceCaptureRequestsSkipped.WithLabelValues(p.typedName.Name, skipReasonNonCompleted).Inc()
		return
	}

	arrival := entry.arrivalTime
	if !response.ReceivedTimestamp.IsZero() {
		arrival = response.ReceivedTimestamp
	}
	completedAt := time.Now()
	if !response.CompleteTimestamp.IsZero() {
		completedAt = response.CompleteTimestamp
	}
	apiSeconds := max(0.0, completedAt.Sub(arrival).Seconds())

	streamType := streamTypeNormal
	var ttftSeconds float64
	if response.Streaming {
		streamType = streamTypeStreaming
		if !response.FirstTokenTimestamp.IsZero() {
			ttftSeconds = max(0.0, response.FirstTokenTimestamp.Sub(arrival).Seconds())
		}
	}

	outTokens := response.Usage.CompletionTokens
	if outTokens == 0 && response.Streaming && response.StreamedEvents > 0 {
		outTokens = response.StreamedEvents
	}

	var cachedTokens int
	if response.Usage.PromptTokenDetails != nil {
		cachedTokens = response.Usage.PromptTokenDetails.CachedTokens
	}

	model := entry.model
	if model == "" && request.TargetModel != "" {
		model = request.TargetModel
	}

	p.sink.Enqueue(CapturedRecord{
		SchemaVersion:     SchemaVersion,
		CaptureRunID:      p.hasher.CaptureRunID(),
		RequestID:         request.RequestID,
		SessionID:         entry.sessionID,
		ParentSessionID:   entry.parentSessionID,
		SubagentID:        entry.subagentID,
		ArrivalUnixNano:   arrival.UnixNano(),
		FirstTokenSeconds: ttftSeconds,
		APITimeSeconds:    apiSeconds,
		StreamType:        streamType,
		Model:             model,
		BlockSize:         p.cfg.BlockSizeTokens,
		TokenizedIn:       entry.tokenizedIn,
		UsageIn:           response.Usage.PromptTokens,
		CachedTokens:      cachedTokens,
		OutputTokens:      outTokens,
		MaxOutputTokens:   entry.maxOutputTokens,
		HasTools:          entry.hasTools,
		Outcome:           outcome,
		BlockHashes:       entry.blockHashes,
	})
}

// Close flushes and closes the underlying file sink.
func (p *Producer) Close() error {
	return p.sink.Close()
}

type debugState struct {
	CaptureRunID    string  `json:"captureRunId"`
	BlockSizeTokens int     `json:"blockSizeTokens"`
	SampleRate      float64 `json:"sampleRate"`
	ActiveFile      string  `json:"activeFile"`
	InFlight        int64   `json:"inFlight"`
	RecordsWritten  int64   `json:"recordsWritten"`
	RecordsDropped  int64   `json:"recordsDropped"`
	WriteErrors     int64   `json:"writeErrors"`
	FileRotations   int64   `json:"fileRotations"`
}

// DumpState implements fwkplugin.StateDumper with operational counters only.
// No HMAC key material, session identifiers, or block digests are exposed.
func (p *Producer) DumpState() (json.RawMessage, error) {
	return json.Marshal(debugState{
		CaptureRunID:    p.hasher.CaptureRunID(),
		BlockSizeTokens: p.cfg.BlockSizeTokens,
		SampleRate:      p.sampleRate,
		ActiveFile:      p.sink.ActiveFile(),
		InFlight:        p.inFlight.Load(),
		RecordsWritten:  p.sink.WrittenRecords(),
		RecordsDropped:  p.sink.DroppedRecords(),
		WriteErrors:     p.sink.WriteErrors(),
		FileRotations:   p.sink.Rotations(),
	})
}

type inFlightCapture struct {
	arrivalTime     time.Time
	model           string
	tokenizedIn     int
	blockHashes     []string
	sessionID       string
	parentSessionID string
	subagentID      string
	maxOutputTokens *int64
	hasTools        bool

	httpStatus atomic.Value // string
	evictOnce  sync.Once
	onEvict    func()
}

func (c *inFlightCapture) setStatus(status string) {
	c.httpStatus.Store(status)
}

func (c *inFlightCapture) status() string {
	v, _ := c.httpStatus.Load().(string)
	return v
}

func (c *inFlightCapture) Clone() fwkplugin.StateData {
	hashes := make([]string, len(c.blockHashes))
	copy(hashes, c.blockHashes)
	var maxOut *int64
	if c.maxOutputTokens != nil {
		v := *c.maxOutputTokens
		maxOut = &v
	}
	cloned := &inFlightCapture{
		arrivalTime:     c.arrivalTime,
		model:           c.model,
		tokenizedIn:     c.tokenizedIn,
		blockHashes:     hashes,
		sessionID:       c.sessionID,
		parentSessionID: c.parentSessionID,
		subagentID:      c.subagentID,
		maxOutputTokens: maxOut,
		hasTools:        c.hasTools,
	}
	if st := c.status(); st != "" {
		cloned.setStatus(st)
	}
	return cloned
}

func (c *inFlightCapture) OnEvicted(_ string, _ fwkplugin.StateKey) {
	c.evictOnce.Do(func() {
		if c.onEvict != nil {
			c.onEvict()
		}
	})
}

func classifyOutcome(status string, cause fwkrh.TerminationCause) string {
	switch {
	case cause == fwkrh.TerminationCauseEvicted:
		return outcomeEvicted
	case status != "" && status != "200":
		return outcomeFailed
	case cause == fwkrh.TerminationCauseNatural:
		return outcomeCompleted
	case cause == fwkrh.TerminationCauseClientDisconnect:
		return outcomeCancelled
	default:
		return outcomeFailed
	}
}

func hasRequestTools(body *fwkrequest.InferenceRequestBody) bool {
	if body == nil {
		return false
	}
	if body.ChatCompletions != nil && len(body.ChatCompletions.Tools) > 0 {
		return true
	}
	if body.Messages != nil && len(body.Messages.Tools) > 0 {
		return true
	}
	if body.Responses != nil && body.Responses.Tools != nil {
		rv := reflect.ValueOf(body.Responses.Tools)
		if rv.Kind() == reflect.Slice && rv.Len() > 0 {
			return true
		}
	}
	return false
}

// splitCompositeSubagentSessionID extracts the immediate parent session ID and
// normalized subagent key from "<parent>::sa:<agent_id>[:s<stream_index>]"
// composite session identifiers emitted by inference-perf WEKA trace replay.
func splitCompositeSubagentSessionID(sessionID string) (parentID, subagentKey string) {
	idx := strings.LastIndex(sessionID, subagentSessionDelimiter)
	if idx <= 0 {
		return "", ""
	}
	parent := sessionID[:idx]
	tail := sessionID[idx+len(subagentSessionDelimiter):]
	if tail == "" {
		return "", ""
	}
	agentBase := stripStreamIndexSuffix(tail)
	if agentBase == "" {
		return "", ""
	}
	return parent, parent + subagentSessionDelimiter + agentBase
}

func stripStreamIndexSuffix(s string) string {
	idx := strings.LastIndex(s, ":s")
	if idx <= 0 || idx+2 >= len(s) {
		return s
	}
	for i := idx + 2; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return s
		}
	}
	return s[:idx]
}

func rootSessionPrefix(sessionID string) string {
	if idx := strings.Index(sessionID, subagentSessionDelimiter); idx > 0 {
		return sessionID[:idx]
	}
	return sessionID
}
