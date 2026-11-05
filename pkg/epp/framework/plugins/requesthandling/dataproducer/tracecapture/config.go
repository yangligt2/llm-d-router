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

// Package tracecapture provides a DataProducer and response-lifecycle plugin
// that captures anonymized, KV-block-hashed inference request records and
// assembles them into WEKA-format multi-turn agentic traces.
package tracecapture

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// PluginType is the registered plugin type name.
	PluginType = "trace-capture-producer"

	// Client and harness session/subagent correlation headers not defined in
	// shared router packages.
	HeaderDynamoSessionID       = "x-dynamo-session-id"
	HeaderDynamoParentSessionID = "x-dynamo-parent-session-id"
	HeaderCorrelationID         = "x-correlation-id"
	HeaderSMGRoutingKey         = "x-smg-routing-key"
	HeaderClaudeCodeAgentID     = "x-claude-code-agent-id"

	DefaultBlockSizeTokens      = 64
	DefaultSampleRate           = 1.0
	DefaultBufferCapacity       = 4096
	DefaultMaxFileBytes         = 128 << 20 // 128 MiB
	DefaultMaxRotatedFiles      = 16
	DefaultFlushIntervalSeconds = 1.0

	maxBlockSizeTokens = 65536
	maxBufferCapacity  = 1_000_000
)

// Config configures the trace-capture-producer plugin.
type Config struct {
	// OutputDir is the local directory where rotated capture JSONL files are written.
	OutputDir string `json:"outputDir"`
	// BlockSizeTokens is the KV-cache block size in tokens. Trailing partial blocks are omitted.
	BlockSizeTokens int `json:"blockSizeTokens,omitempty"`
	// SampleRate is the fraction [0.0, 1.0] of sessions captured. Nil defaults to 1.0.
	SampleRate *float64 `json:"sampleRate,omitempty"`
	// BufferCapacity bounds the in-memory record channel feeding the background writer.
	BufferCapacity int `json:"bufferCapacity,omitempty"`
	// MaxFileBytes is the byte threshold at which the active JSONL file rotates.
	MaxFileBytes int64 `json:"maxFileBytes,omitempty"`
	// MaxRotatedFiles bounds how many closed JSONL files are retained alongside the active file.
	MaxRotatedFiles int `json:"maxRotatedFiles,omitempty"`
	// FlushIntervalSeconds sets the periodic flush cadence of the buffered file writer.
	FlushIntervalSeconds float64 `json:"flushIntervalSeconds,omitempty"`
	// AllowEstimateBackend permits running alongside token-producer's byte-packing estimate backend.
	// False requires token-producer with backend "vllm".
	AllowEstimateBackend bool `json:"allowEstimateBackend,omitempty"`
	// IncludeNonCompleted records evicted, cancelled, and failed requests instead of skipping them.
	IncludeNonCompleted bool `json:"includeNonCompleted,omitempty"`
	// SessionSources resolves the session identifier in priority order.
	SessionSources []sessionaffinity.SessionIDSource `json:"sessionSources,omitempty"`
	// ParentSessionSources resolves the parent session identifier for subagent requests.
	ParentSessionSources []sessionaffinity.SessionIDSource `json:"parentSessionSources,omitempty"`
	// SubagentSources resolves the subagent identifier within a parent session.
	SubagentSources []sessionaffinity.SessionIDSource `json:"subagentSources,omitempty"`
}

// DefaultSessionSources returns the priority-ordered session sources used when
// Config.SessionSources is unset.
func DefaultSessionSources() []sessionaffinity.SessionIDSource {
	return []sessionaffinity.SessionIDSource{
		{Attribute: agentidentity.AgentIdentityKey.DataType()},
		{Header: sessionaffinity.DefaultSessionIDHeader},
		{Header: HeaderDynamoSessionID},
		{Header: HeaderCorrelationID},
		{Header: HeaderSMGRoutingKey},
		{Header: agentidentity.ClaudeCodeSessionHeader},
		{Header: agentidentity.OpenCodeSessionHeader},
		{Header: agentidentity.CodexSessionHeader},
		{Header: agentidentity.CodexSessionHeaderLegacy},
		{Header: metadata.FlowFairnessIDKey},
	}
}

// ApplyDefaults populates unset fields with built-in defaults and reports
// whether default SessionSources were applied.
func (c *Config) ApplyDefaults() (usedDefaultSessionSources bool) {
	c.OutputDir = strings.TrimSpace(c.OutputDir)
	if c.BlockSizeTokens == 0 {
		c.BlockSizeTokens = DefaultBlockSizeTokens
	}
	if c.SampleRate == nil {
		rate := DefaultSampleRate
		c.SampleRate = &rate
	}
	if c.BufferCapacity == 0 {
		c.BufferCapacity = DefaultBufferCapacity
	}
	if c.MaxFileBytes == 0 {
		c.MaxFileBytes = DefaultMaxFileBytes
	}
	if c.MaxRotatedFiles == 0 {
		c.MaxRotatedFiles = DefaultMaxRotatedFiles
	}
	if c.FlushIntervalSeconds == 0 {
		c.FlushIntervalSeconds = DefaultFlushIntervalSeconds
	}
	if len(c.SessionSources) == 0 {
		c.SessionSources = DefaultSessionSources()
		usedDefaultSessionSources = true
	}
	if len(c.ParentSessionSources) == 0 {
		c.ParentSessionSources = []sessionaffinity.SessionIDSource{
			{Header: HeaderDynamoParentSessionID},
		}
	}
	if len(c.SubagentSources) == 0 {
		c.SubagentSources = []sessionaffinity.SessionIDSource{
			{Header: HeaderClaudeCodeAgentID},
		}
	}
	return usedDefaultSessionSources
}

// Validate checks a defaulted Config.
func (c *Config) Validate() error {
	if c.OutputDir == "" {
		return errors.New("outputDir is required")
	}
	if strings.ContainsRune(c.OutputDir, '\x00') {
		return errors.New("outputDir must not contain NUL bytes")
	}
	if c.BlockSizeTokens <= 0 || c.BlockSizeTokens > maxBlockSizeTokens {
		return fmt.Errorf("blockSizeTokens must be in [1, %d], got %d", maxBlockSizeTokens, c.BlockSizeTokens)
	}
	if c.SampleRate == nil || math.IsNaN(*c.SampleRate) || *c.SampleRate < 0 || *c.SampleRate > 1 {
		return fmt.Errorf("sampleRate must be in [0, 1], got %v", c.SampleRate)
	}
	if c.BufferCapacity <= 0 || c.BufferCapacity > maxBufferCapacity {
		return fmt.Errorf("bufferCapacity must be in [1, %d], got %d", maxBufferCapacity, c.BufferCapacity)
	}
	if c.MaxFileBytes <= 0 {
		return fmt.Errorf("maxFileBytes must be > 0, got %d", c.MaxFileBytes)
	}
	if c.MaxRotatedFiles <= 0 {
		return fmt.Errorf("maxRotatedFiles must be > 0, got %d", c.MaxRotatedFiles)
	}
	if math.IsNaN(c.FlushIntervalSeconds) || math.IsInf(c.FlushIntervalSeconds, 0) || c.FlushIntervalSeconds <= 0 {
		return fmt.Errorf("flushIntervalSeconds must be > 0, got %v", c.FlushIntervalSeconds)
	}
	if err := sessionaffinity.ValidateSources(c.SessionSources); err != nil {
		return fmt.Errorf("sessionSources: %w", err)
	}
	if err := sessionaffinity.ValidateSources(c.ParentSessionSources); err != nil {
		return fmt.Errorf("parentSessionSources: %w", err)
	}
	if err := sessionaffinity.ValidateSources(c.SubagentSources); err != nil {
		return fmt.Errorf("subagentSources: %w", err)
	}
	return nil
}
