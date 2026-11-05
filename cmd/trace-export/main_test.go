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

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tracecapture"
)

func TestRunCLI_JSONLAndDirOutput(t *testing.T) {
	t.Parallel()

	inputDir := t.TempDir()
	inputFile := filepath.Join(inputDir, "capture-test-000000.jsonl")

	recs := []tracecapture.CapturedRecord{
		{
			SchemaVersion:   tracecapture.SchemaVersion,
			CaptureRunID:    "run1",
			RequestID:       "req1",
			SessionID:       "s1",
			ArrivalUnixNano: 1_000_000_000,
			APITimeSeconds:  0.5,
			StreamType:      "n",
			Model:           "qwen-test",
			BlockSize:       16,
			TokenizedIn:     64,
			OutputTokens:    10,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3", "b4"},
		},
		{
			SchemaVersion:   tracecapture.SchemaVersion,
			CaptureRunID:    "run1",
			RequestID:       "req2",
			SessionID:       "s1",
			ArrivalUnixNano: 2_000_000_000,
			APITimeSeconds:  0.6,
			StreamType:      "n",
			Model:           "qwen-test",
			BlockSize:       16,
			TokenizedIn:     128,
			OutputTokens:    20,
			Outcome:         "completed",
			BlockHashes:     []string{"b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8"},
		},
	}

	var buf bytes.Buffer
	for _, r := range recs {
		b, err := json.Marshal(r)
		require.NoError(t, err)
		buf.Write(b)
		buf.WriteByte('\n')
	}
	require.NoError(t, os.WriteFile(inputFile, buf.Bytes(), 0o600))

	// 1. Export to single JSONL file with 16->64 block downsampling.
	outJSONL := filepath.Join(t.TempDir(), "traces.jsonl")
	var stdout bytes.Buffer
	err := run([]string{
		"--input-dir", inputDir,
		"--output-jsonl", outJSONL,
		"--target-block-size", "64",
		"--hash-id-scope", "local",
	}, &stdout)
	require.NoError(t, err)

	rawJSONL, err := os.ReadFile(outJSONL)
	require.NoError(t, err)
	lines := strings.Split(strings.TrimSpace(string(rawJSONL)), "\n")
	require.Len(t, lines, 1)

	var tr tracecapture.WekaTrace
	require.NoError(t, json.Unmarshal([]byte(lines[0]), &tr))
	assert.Equal(t, 64, tr.BlockSize)
	require.Len(t, tr.Requests, 2)
	assert.Equal(t, []int{1}, tr.Requests[0].Turn.HashIDs)
	assert.Equal(t, []int{1, 2}, tr.Requests[1].Turn.HashIDs)

	// 2. Export to per-trace JSON directory (for inference-perf trace_directory).
	outDir := filepath.Join(t.TempDir(), "weka-dir")
	err = run([]string{
		"--input-dir", inputDir,
		"--output-dir", outDir,
	}, &stdout)
	require.NoError(t, err)

	perTraceBytes, err := os.ReadFile(filepath.Join(outDir, "trace_0000.json"))
	require.NoError(t, err)
	var trDir tracecapture.WekaTrace
	require.NoError(t, json.Unmarshal(perTraceBytes, &trDir))
	assert.Equal(t, 16, trDir.BlockSize)
	require.Len(t, trDir.Requests, 2)
}
