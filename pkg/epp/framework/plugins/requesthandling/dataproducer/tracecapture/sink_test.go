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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSink_PermissionsRotationPruningAndDrain(t *testing.T) {
	t.Parallel()

	outDir := filepath.Join(t.TempDir(), "nested", "traces")
	sink, err := NewSink(context.Background(), SinkOptions{
		OutputDir:       outDir,
		RunID:           "run01",
		BufferCapacity:  64,
		MaxFileBytes:    450, // small threshold so every ~2 records rotates
		MaxRotatedFiles: 2,
		FlushInterval:   50 * time.Millisecond,
		PluginName:      "test-capture",
	})
	require.NoError(t, err)

	dirInfo, err := os.Stat(outDir)
	require.NoError(t, err)
	assert.Equal(t, os.FileMode(0o700), dirInfo.Mode().Perm())

	for i := range 10 {
		ok := sink.Enqueue(CapturedRecord{
			SchemaVersion:   SchemaVersion,
			CaptureRunID:    "run01",
			RequestID:       "req-" + string(rune('a'+i)),
			SessionID:       "sess-01",
			ArrivalUnixNano: int64(1_000_000_000 + i*1_000_000),
			APITimeSeconds:  0.25,
			StreamType:      "n",
			Model:           "qwen-test",
			BlockSize:       64,
			TokenizedIn:     128,
			OutputTokens:    32,
			Outcome:         "completed",
			BlockHashes:     []string{"00112233445566778899aabbccddeeff", "ffeeddccbbaa99887766554433221100"},
		})
		require.True(t, ok)
	}

	require.NoError(t, sink.Close())
	// Idempotent second Close.
	require.NoError(t, sink.Close())
	// Enqueue after Close returns false without blocking or panicking.
	assert.False(t, sink.Enqueue(CapturedRecord{RequestID: "late"}))

	assert.Equal(t, int64(10), sink.WrittenRecords())
	assert.GreaterOrEqual(t, sink.Rotations(), int64(3))

	entries, err := os.ReadDir(outDir)
	require.NoError(t, err)
	var jsonlFiles []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".jsonl") {
			jsonlFiles = append(jsonlFiles, filepath.Join(outDir, e.Name()))
			info, statErr := e.Info()
			require.NoError(t, statErr)
			assert.Equal(t, os.FileMode(0o600), info.Mode().Perm())
		}
	}
	// Active file + at most MaxRotatedFiles (2) = at most 3 files retained.
	assert.LessOrEqual(t, len(jsonlFiles), 3)
	assert.GreaterOrEqual(t, len(jsonlFiles), 2)

	loaded, err := LoadCapturedRecords(jsonlFiles)
	require.NoError(t, err)
	assert.NotEmpty(t, loaded)
}

func TestSink_BackpressureDropsWithoutBlocking(t *testing.T) {
	t.Parallel()

	outDir := filepath.Join(t.TempDir(), "bp")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sink, err := NewSink(ctx, SinkOptions{
		OutputDir:       outDir,
		RunID:           "bp01",
		BufferCapacity:  1,
		MaxFileBytes:    10 << 20,
		MaxRotatedFiles: 4,
		FlushInterval:   time.Hour,
		PluginName:      "bp-test",
	})
	require.NoError(t, err)

	// Invoke non-blocking Enqueue in a tight loop; calls must return without blocking when buffer saturates.
	start := time.Now()
	dropped := 0
	for i := range 5000 {
		if !sink.Enqueue(CapturedRecord{
			SchemaVersion: SchemaVersion,
			CaptureRunID:  "bp01",
			RequestID:     "r",
			BlockHashes:   []string{"0123456789abcdef0123456789abcdef"},
			TokenizedIn:   i + 1,
		}) {
			dropped++
		}
	}
	assert.Less(t, time.Since(start), 2*time.Second)
	require.NoError(t, sink.Close())
	assert.Equal(t, int64(dropped), sink.DroppedRecords())
}
