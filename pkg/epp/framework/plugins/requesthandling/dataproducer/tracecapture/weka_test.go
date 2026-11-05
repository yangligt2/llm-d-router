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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestAssembleWekaTraces_SubagentsClassifierFilterAndLocalScope(t *testing.T) {
	t.Parallel()

	smallCap := int64(32)
	records := []CapturedRecord{
		// Root turn 1 at t=0s (duration 1.5s)
		{
			RequestID:         "r1",
			SessionID:         "sessA",
			ArrivalUnixNano:   10_000_000_000,
			APITimeSeconds:    1.5,
			FirstTokenSeconds: 0.1,
			StreamType:        "s",
			Model:             "qwen-7b",
			BlockSize:         64,
			TokenizedIn:       150,
			OutputTokens:      40,
			HasTools:          true,
			Outcome:           "completed",
			BlockHashes:       []string{"h1", "h2"},
		},
		// Classifier call inside sessA (max_output_tokens <= 64 && !has_tools) -> dropped when DropClassifierCalls=true
		{
			RequestID:       "r-classifier",
			SessionID:       "sessA",
			ArrivalUnixNano: 11_600_000_000,
			APITimeSeconds:  0.2,
			StreamType:      "n",
			Model:           "qwen-7b",
			BlockSize:       64,
			TokenizedIn:     80,
			OutputTokens:    5,
			MaxOutputTokens: &smallCap,
			HasTools:        false,
			Outcome:         "completed",
			BlockHashes:     []string{"hc1"},
		},
		// Subagent turn 1 at t=2.0s (duration 0.8s)
		{
			RequestID:         "r-sub-1",
			SessionID:         "subSess1",
			ParentSessionID:   "sessA",
			SubagentID:        "agent01",
			ArrivalUnixNano:   12_000_000_000,
			APITimeSeconds:    0.8,
			FirstTokenSeconds: 0.05,
			StreamType:        "s",
			Model:             "qwen-7b",
			BlockSize:         64,
			TokenizedIn:       130,
			OutputTokens:      25,
			HasTools:          true,
			Outcome:           "completed",
			BlockHashes:       []string{"h1", "sub_b2"},
		},
		// Subagent turn 2 at t=3.3s (0.5s think_time after subagent turn 1 ended at 2.8s)
		{
			RequestID:       "r-sub-2",
			SessionID:       "subSess1",
			ParentSessionID: "sessA",
			SubagentID:      "agent01",
			ArrivalUnixNano: 13_300_000_000,
			APITimeSeconds:  0.7,
			StreamType:      "n",
			Model:           "qwen-7b",
			BlockSize:       64,
			TokenizedIn:     200,
			OutputTokens:    30,
			HasTools:        true,
			Outcome:         "completed",
			BlockHashes:     []string{"h1", "sub_b2", "sub_b3"},
		},
		// Root turn 2 at t=5.0s (3.5s think_time after root turn 1 ended at 1.5s)
		{
			RequestID:       "r2",
			SessionID:       "sessA",
			ArrivalUnixNano: 15_000_000_000,
			APITimeSeconds:  2.0,
			StreamType:      "n",
			Model:           "qwen-7b",
			BlockSize:       64,
			TokenizedIn:     210,
			OutputTokens:    55,
			HasTools:        true,
			Outcome:         "completed",
			BlockHashes:     []string{"h1", "h2", "h3"},
		},
		// Second independent session sessB sharing digest "h1" -> under local scope gets hash_id 1 independently!
		{
			RequestID:       "b1",
			SessionID:       "sessB",
			ArrivalUnixNano: 20_000_000_000,
			APITimeSeconds:  1.0,
			StreamType:      "n",
			Model:           "qwen-7b",
			BlockSize:       64,
			TokenizedIn:     70,
			OutputTokens:    12,
			HasTools:        true,
			Outcome:         "completed",
			BlockHashes:     []string{"other_root_block"},
		},
	}

	traces, err := AssembleWekaTraces(records, AssembleOptions{
		HashIDScope:         HashIDScopeLocal,
		DropClassifierCalls: true,
	})
	require.NoError(t, err)
	require.Len(t, traces, 2)

	trA := traces[0]
	assert.Equal(t, "trace_0000", trA.ID)
	assert.Equal(t, 64, trA.BlockSize)
	assert.Equal(t, HashIDScopeLocal, trA.HashIDScope)
	assert.Equal(t, []string{"qwen-7b"}, trA.Models)
	require.Len(t, trA.Requests, 3) // root turn 1, subagent entry, root turn 2

	turn1 := trA.Requests[0].Turn
	require.NotNil(t, turn1)
	assert.Equal(t, "s", turn1.Type)
	assert.InDelta(t, 0.0, turn1.T, 1e-6)
	assert.Equal(t, []int{1, 2}, turn1.HashIDs)
	require.NotNil(t, turn1.TTFT)
	assert.InDelta(t, 0.1, *turn1.TTFT, 1e-6)
	require.NotNil(t, turn1.ThinkTime)
	assert.InDelta(t, 0.0, *turn1.ThinkTime, 1e-6)

	sub := trA.Requests[1].Subagent
	require.NotNil(t, sub)
	assert.Equal(t, "subagent", sub.Type)
	assert.Equal(t, "completed", sub.Status)
	assert.Equal(t, []string{"qwen-7b"}, sub.Models)
	require.NotNil(t, sub.TotalTokens)
	assert.Equal(t, 385, *sub.TotalTokens)
	assert.InDelta(t, 2.0, sub.T, 1e-6)
	require.NotNil(t, sub.DurationMs)
	assert.Equal(t, 2000, *sub.DurationMs) // 12.0s -> 14.0s = 2000ms
	require.Len(t, sub.Requests, 2)
	assert.Equal(t, "n", sub.Requests[0].Type)
	assert.Nil(t, sub.Requests[0].TTFT)
	assert.InDelta(t, 2.0, sub.Requests[0].T, 1e-6)
	assert.Equal(t, []int{1, 3}, sub.Requests[0].HashIDs) // shares block 1 ("h1") with parent!
	assert.Equal(t, "n", sub.Requests[1].Type)
	assert.Nil(t, sub.Requests[1].TTFT)
	assert.InDelta(t, 3.3, sub.Requests[1].T, 1e-6)
	require.NotNil(t, sub.Requests[1].ThinkTime)
	assert.InDelta(t, 0.5, *sub.Requests[1].ThinkTime, 1e-6)
	assert.Equal(t, []int{1, 3, 4}, sub.Requests[1].HashIDs)

	turn2 := trA.Requests[2].Turn
	require.NotNil(t, turn2)
	assert.Equal(t, "n", turn2.Type)
	assert.Nil(t, turn2.TTFT)
	assert.InDelta(t, 5.0, turn2.T, 1e-6)
	require.NotNil(t, turn2.ThinkTime)
	assert.InDelta(t, 3.5, *turn2.ThinkTime, 1e-6)
	assert.Equal(t, []int{1, 2, 5}, turn2.HashIDs)

	require.NotNil(t, trA.Totals)
	assert.Equal(t, 1, trA.Totals.SubagentCount)
	assert.Equal(t, 360, trA.Totals.ParentTokens.Input)
	assert.Equal(t, 330, trA.Totals.SubagentTokens.Input)
	assert.Equal(t, 690, trA.Totals.CombinedTokens.Input)

	// Trace B resets local hash_id numbering back to 1.
	trB := traces[1]
	require.Len(t, trB.Requests, 1)
	assert.Equal(t, []int{1}, trB.Requests[0].Turn.HashIDs)
}

func TestAssembleWekaTraces_LCPSessionRecoveryWithoutSessionHeaders(t *testing.T) {
	t.Parallel()

	// Two interleaved conversations with empty SessionID; LCP chaining (>= 0.90) must separate them cleanly.
	records := []CapturedRecord{
		{
			RequestID:       "a1",
			ArrivalUnixNano: 1_000_000_000,
			APITimeSeconds:  1.0,
			StreamType:      "n",
			Model:           "m",
			BlockSize:       64,
			TokenizedIn:     640,
			OutputTokens:    64,
			Outcome:         "completed",
			BlockHashes:     []string{"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10"},
		},
		{
			RequestID:       "b1",
			ArrivalUnixNano: 1_500_000_000,
			APITimeSeconds:  1.0,
			StreamType:      "n",
			Model:           "m",
			BlockSize:       64,
			TokenizedIn:     640,
			OutputTokens:    64,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10"},
		},
		{
			RequestID:       "a2",
			ArrivalUnixNano: 3_000_000_000,
			APITimeSeconds:  1.2,
			StreamType:      "n",
			Model:           "m",
			BlockSize:       64,
			TokenizedIn:     768,
			OutputTokens:    50,
			Outcome:         "completed",
			BlockHashes:     []string{"a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"},
		},
		{
			RequestID:       "b2",
			ArrivalUnixNano: 4_000_000_000,
			APITimeSeconds:  1.1,
			StreamType:      "n",
			Model:           "m",
			BlockSize:       64,
			TokenizedIn:     768,
			OutputTokens:    45,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10_edited", "b11", "b12"},
		},
	}

	traces, err := AssembleWekaTraces(records, AssembleOptions{
		LCPOverlapThreshold: 0.90,
	})
	require.NoError(t, err)
	require.Len(t, traces, 2)
	require.Len(t, traces[0].Requests, 2)
	require.Len(t, traces[1].Requests, 2)
	assert.Equal(t, 12, len(traces[0].Requests[1].Turn.HashIDs))
	assert.Equal(t, 12, len(traces[1].Requests[1].Turn.HashIDs))
}

func TestEndToEnd_AIPerfAgentXWekaTraceReplayFidelity(t *testing.T) {
	t.Parallel()

	// Simulate an original WEKA AgentX trace replayed by aiperf against llm-d-router with trace-capture-producer
	// enabled, then exported via AssembleWekaTraces, and verify structural isomorphism with the input WEKA trace:
	// exact turn counts, subagent structure, in/out token counts, hash_ids LCP prefix-sharing matrix, and timings.
	const blockSize = 64

	// Deterministic synthetic token bank mapping original WEKA block ID -> 64 token IDs (exactly what aiperf does).
	tokensForHashIDs := func(hashIDs []int, totalIn int) []uint32 {
		tokens := make([]uint32, 0, totalIn)
		for _, hid := range hashIDs {
			base := uint32(hid * 1000)
			for j := range blockSize {
				tokens = append(tokens, base+uint32(j))
			}
		}
		for i := len(tokens); i < totalIn; i++ {
			tokens = append(tokens, uint32(900_000+i))
		}
		return tokens
	}

	type simCall struct {
		reqID     string
		sessionID string
		parentID  string
		agentID   string
		tSec      float64
		ttftSec   float64
		apiSec    float64
		streaming bool
		inTokens  int
		outTokens int
		hashIDs   []int
	}

	originalCalls := []simCall{
		// AgentX session 0: 3 parent turns + 1 nested 2-turn subagent, including a prefix truncation on turn 3
		{reqID: "c1", sessionID: "agentx-0", tSec: 0.0, ttftSec: 0.08, apiSec: 1.2, streaming: true, inTokens: 270, outTokens: 42, hashIDs: []int{1, 2, 3, 4}},
		{reqID: "c2", sessionID: "agentx-0", tSec: 3.2, ttftSec: 0.09, apiSec: 1.4, streaming: true, inTokens: 400, outTokens: 58, hashIDs: []int{1, 2, 3, 4, 5, 6}},
		{reqID: "c3", sessionID: "agentx-sub-0", parentID: "agentx-0", agentID: "sub-a", tSec: 5.0, ttftSec: 0.05, apiSec: 0.6, streaming: true, inTokens: 200, outTokens: 30, hashIDs: []int{1, 2, 7}},
		{reqID: "c4", sessionID: "agentx-sub-0", parentID: "agentx-0", agentID: "sub-a", tSec: 6.1, ttftSec: 0.06, apiSec: 0.7, streaming: true, inTokens: 268, outTokens: 25, hashIDs: []int{1, 2, 7, 8}},
		{reqID: "c5", sessionID: "agentx-0", tSec: 8.0, ttftSec: 0.11, apiSec: 1.6, streaming: true, inTokens: 460, outTokens: 64, hashIDs: []int{1, 2, 3, 9, 10, 11, 12}}, // prefix modification at block index 3
	}

	outDir := t.TempDir()
	rawCfg, err := json.Marshal(map[string]any{
		"outputDir":       outDir,
		"blockSizeTokens": blockSize,
	})
	require.NoError(t, err)

	plg, err := PluginFactory("capture", fwkplugin.StrictDecoder(rawCfg), nil)
	require.NoError(t, err)
	producer := plg.(*Producer)

	t0 := time.Unix(1_750_000_000, 0)
	ctx := context.Background()

	for _, sc := range originalCalls {
		headers := map[string]string{
			HeaderDynamoSessionID: sc.sessionID,
		}
		if sc.parentID != "" {
			headers[HeaderDynamoParentSessionID] = sc.parentID
		}
		if sc.agentID != "" {
			headers[HeaderClaudeCodeAgentID] = sc.agentID
		}
		req := &fwksched.InferenceRequest{
			RequestID:   sc.reqID,
			TargetModel: "Qwen/Qwen2.5-0.5B-Instruct",
			Headers:     headers,
			Body: &fwkrequest.InferenceRequestBody{
				Model:            "Qwen/Qwen2.5-0.5B-Instruct",
				Stream:           sc.streaming,
				TokenizedRequest: fwkrequest.NewTokenizedRequest([][]uint32{tokensForHashIDs(sc.hashIDs, sc.inTokens)}),
			},
		}
		require.NoError(t, producer.Produce(ctx, req, nil))
		req.Body = req.Body.ResponsePhaseView()

		producer.ResponseHeader(ctx, req, &fwkrh.Response{
			RequestID: req.RequestID,
			Headers:   map[string]string{reqcommon.PseudoHeaderStatus: "200"},
		}, nil)

		arr := t0.Add(time.Duration(sc.tSec * float64(time.Second)))
		producer.ResponseBody(ctx, req, &fwkrh.Response{
			RequestID:           req.RequestID,
			EndOfStream:         true,
			Streaming:           sc.streaming,
			TerminationCause:    fwkrh.TerminationCauseNatural,
			ReceivedTimestamp:   arr,
			FirstTokenTimestamp: arr.Add(time.Duration(sc.ttftSec * float64(time.Second))),
			CompleteTimestamp:   arr.Add(time.Duration(sc.apiSec * float64(time.Second))),
			Usage: fwkrequest.Usage{
				PromptTokens:     sc.inTokens,
				CompletionTokens: sc.outTokens,
			},
		}, nil)
	}

	require.NoError(t, producer.Close())

	files, err := filepath.Glob(filepath.Join(outDir, "capture-*.jsonl"))
	require.NoError(t, err)
	captured, err := LoadCapturedRecords(files)
	require.NoError(t, err)
	require.Len(t, captured, len(originalCalls))

	traces, err := AssembleWekaTraces(captured, AssembleOptions{HashIDScope: HashIDScopeLocal})
	require.NoError(t, err)
	require.Len(t, traces, 1)

	got := traces[0]
	require.Len(t, got.Requests, 4) // turn 1, turn 2, subagent(2 turns), turn 3

	// Verify round-trip JSON serialization matches WEKA schema cleanly.
	jsonlPath := filepath.Join(t.TempDir(), "traces.jsonl")
	require.NoError(t, WriteWekaTracesJSONL(jsonlPath, traces))

	// Verify exact hash_ids isomorphism (1-based dense local numbering matches originalCalls!).
	assert.Equal(t, []int{1, 2, 3, 4}, got.Requests[0].Turn.HashIDs)
	assert.Equal(t, 270, got.Requests[0].Turn.In)
	assert.Equal(t, 42, got.Requests[0].Turn.Out)

	assert.Equal(t, []int{1, 2, 3, 4, 5, 6}, got.Requests[1].Turn.HashIDs)
	assert.Equal(t, 400, got.Requests[1].Turn.In)
	assert.Equal(t, 58, got.Requests[1].Turn.Out)

	require.NotNil(t, got.Requests[2].Subagent)
	require.Len(t, got.Requests[2].Subagent.Requests, 2)
	assert.Equal(t, []int{1, 2, 7}, got.Requests[2].Subagent.Requests[0].HashIDs)
	assert.Equal(t, []int{1, 2, 7, 8}, got.Requests[2].Subagent.Requests[1].HashIDs)

	// Parent turn 3 shares first 3 blocks [1, 2, 3] and diverges at block index 3 -> [9, 10, 11, 12]!
	assert.Equal(t, []int{1, 2, 3, 9, 10, 11, 12}, got.Requests[3].Turn.HashIDs)
	assert.Equal(t, 460, got.Requests[3].Turn.In)
	assert.Equal(t, 64, got.Requests[3].Turn.Out)
	assert.InDelta(t, 8.0, got.Requests[3].Turn.T, 1e-6)
}

func TestAssembleWekaTraces_NestedSubagentHierarchy(t *testing.T) {
	t.Parallel()

	// Root session R spawns depth-1 subagent S1, which spawns depth-2 subagent S2.
	// Also test self-referential ParentSessionID == SessionID on R (must stay a root turn).
	records := []CapturedRecord{
		{
			RequestID:       "r1",
			SessionID:       "root",
			ParentSessionID: "root",
			ArrivalUnixNano: 1_000_000_000,
			APITimeSeconds:  1.0,
			StreamType:      "s",
			Model:           "qwen3-4b",
			BlockSize:       64,
			TokenizedIn:     128,
			OutputTokens:    20,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2"},
		},
		{
			RequestID:       "s1-1",
			SessionID:       "sub1",
			ParentSessionID: "root",
			ArrivalUnixNano: 2_500_000_000,
			APITimeSeconds:  0.8,
			StreamType:      "s",
			Model:           "qwen3-4b",
			BlockSize:       64,
			TokenizedIn:     192,
			OutputTokens:    30,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3"},
		},
		{
			RequestID:       "s2-1",
			SessionID:       "sub2",
			ParentSessionID: "sub1",
			ArrivalUnixNano: 3_500_000_000,
			APITimeSeconds:  0.6,
			StreamType:      "s",
			Model:           "qwen3-4b",
			BlockSize:       64,
			TokenizedIn:     256,
			OutputTokens:    15,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3", "b4"},
		},
	}

	traces, err := AssembleWekaTraces(records, AssembleOptions{HashIDScope: HashIDScopeLocal})
	require.NoError(t, err)
	require.Len(t, traces, 1)

	tr := traces[0]
	require.Len(t, tr.Requests, 3)
	require.NotNil(t, tr.Requests[0].Turn)
	require.NotNil(t, tr.Requests[1].Subagent)
	require.NotNil(t, tr.Requests[2].Subagent)
	assert.Equal(t, "sub1", tr.Requests[1].Subagent.AgentID)
	assert.Equal(t, "sub2", tr.Requests[2].Subagent.AgentID)
	assert.Equal(t, []int{1, 2, 3, 4}, tr.Requests[2].Subagent.Requests[0].HashIDs)
	assert.Equal(t, 2, tr.Totals.SubagentCount)
}
