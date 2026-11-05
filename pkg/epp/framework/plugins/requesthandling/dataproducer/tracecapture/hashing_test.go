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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

func makeSeqTokens(start, count uint32) []uint32 {
	out := make([]uint32, count)
	for i := range count {
		out[i] = start + i
	}
	return out
}

func TestHasher_FullBlockTruncationAndPrefixSharing(t *testing.T) {
	t.Parallel()

	key := [32]byte{1, 2, 3, 4, 5, 6, 7, 8}
	h := NewHasherWithKey(key, "run-test")

	// Short prompt (< blockSize) yields empty slice, never partial block hash.
	short := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeSeqTokens(10, 63)}},
	}
	shortHashes := h.HashTokenizedRequest(short, 64)
	assert.Empty(t, shortHashes)
	assert.NotNil(t, shortHashes)

	// Turn 1: 150 tokens -> 2 full blocks (128 tokens), trailing 22 tokens dropped.
	turn1Tokens := makeSeqTokens(100, 150)
	turn1 := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: turn1Tokens}},
	}
	h1 := h.HashTokenizedRequest(turn1, 64)
	require.Len(t, h1, 2)
	assert.Len(t, h1[0], 32)
	assert.NotEqual(t, h1[0], h1[1])

	// Turn 2 shares exact first 128 tokens, diverges in tokens 128..149, and extends to 260 tokens (4 blocks).
	turn2Tokens := make([]uint32, 260)
	copy(turn2Tokens, turn1Tokens[:128])
	for i := 128; i < 260; i++ {
		turn2Tokens[i] = uint32(9000 + i)
	}
	turn2 := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: turn2Tokens}},
	}
	h2 := h.HashTokenizedRequest(turn2, 64)
	require.Len(t, h2, 4)
	assert.Equal(t, h1[0], h2[0])
	assert.Equal(t, h1[1], h2[1])

	// Modifying token 0 changes every block in the chain (prefix commitment).
	divergedTokens := make([]uint32, len(turn2Tokens))
	copy(divergedTokens, turn2Tokens)
	divergedTokens[0] = 999999
	diverged := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: divergedTokens}},
	}
	hDiverged := h.HashTokenizedRequest(diverged, 64)
	require.Len(t, hDiverged, 4)
	for i := range h2 {
		assert.NotEqual(t, h2[i], hDiverged[i], "block %d must change when prefix changes", i)
	}
}

func TestHasher_IndependentInstancesProduceDisjointDigests(t *testing.T) {
	t.Parallel()

	hA, err := NewHasher()
	require.NoError(t, err)
	hB, err := NewHasher()
	require.NoError(t, err)

	assert.NotEqual(t, hA.CaptureRunID(), hB.CaptureRunID())

	tr := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{TokenIDs: makeSeqTokens(1, 128)}},
	}
	assert.NotEqual(t, hA.HashTokenizedRequest(tr, 64), hB.HashTokenizedRequest(tr, 64))
	assert.NotEqual(t, hA.AnonymizeSessionID("sess-1"), hB.AnonymizeSessionID("sess-1"))
}

func TestHasher_MultiModalAndCacheSaltSensitivity(t *testing.T) {
	t.Parallel()

	h := NewHasherWithKey([32]byte{9, 9, 9}, "run-mm")
	tokens := makeSeqTokens(50, 128) // placeholder tokens identical across two distinct images

	imgA := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{
			TokenIDs: tokens,
			MultiModalFeatures: []fwkrequest.MultiModalFeature{{
				Modality: reqcommon.ModalityImage,
				Hash:     "img-hash-aaa",
				Offset:   10,
				Length:   20,
			}},
		}},
	}
	imgB := &fwkrequest.TokenizedRequest{
		Prompts: []fwkrequest.PromptTokens{{
			TokenIDs: tokens,
			MultiModalFeatures: []fwkrequest.MultiModalFeature{{
				Modality: reqcommon.ModalityImage,
				Hash:     "img-hash-bbb",
				Offset:   10,
				Length:   20,
			}},
		}},
	}

	hashesA := h.HashTokenizedRequest(imgA, 64)
	hashesB := h.HashTokenizedRequest(imgB, 64)
	require.Len(t, hashesA, 2)
	require.Len(t, hashesB, 2)
	// Image sits in block 0 [0,64), and block 1 [64,128) chains from block 0, so both blocks differ.
	assert.NotEqual(t, hashesA[0], hashesB[0])
	assert.NotEqual(t, hashesA[1], hashesB[1])

	// CacheSalt isolates prefixes with identical tokens.
	salted := &fwkrequest.TokenizedRequest{
		CacheSalt: "tenant-a",
		Prompts:   imgA.Prompts,
	}
	hashesSalted := h.HashTokenizedRequest(salted, 64)
	assert.NotEqual(t, hashesA[0], hashesSalted[0])
}

func TestHasher_BlockDownsamplingPreservesPrefixEquivalence(t *testing.T) {
	t.Parallel()

	h := NewHasherWithKey([32]byte{42}, "run-ds")

	// Sequence A and Sequence B share the first 128 tokens, diverge at token 130, and are 200 tokens long.
	seqA := makeSeqTokens(1000, 200)
	seqB := makeSeqTokens(1000, 200)
	for i := 130; i < 200; i++ {
		seqB[i] = uint32(50000 + i)
	}

	// Capture at block_size=16 -> 12 blocks (192 tokens).
	hashesA16 := h.HashTokenizedRequest(&fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: seqA}}}, 16)
	hashesB16 := h.HashTokenizedRequest(&fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: seqB}}}, 16)
	require.Len(t, hashesA16, 12)
	require.Len(t, hashesB16, 12)

	// Downsample 16 -> 64 (factor k=4) -> 3 blocks covering [0,64), [64,128), [128,192).
	dsA64, err := DownsampleBlockHashes(hashesA16, 16, 64)
	require.NoError(t, err)
	dsB64, err := DownsampleBlockHashes(hashesB16, 16, 64)
	require.NoError(t, err)

	require.Len(t, dsA64, 3)
	require.Len(t, dsB64, 3)
	assert.Equal(t, dsA64[0], dsB64[0])
	assert.Equal(t, dsA64[1], dsB64[1])
	assert.NotEqual(t, dsA64[2], dsB64[2])
}

func TestHasher_DomainSeparatedIDsAndDeterministicSampling(t *testing.T) {
	t.Parallel()

	h := NewHasherWithKey([32]byte{7, 7, 7}, "run-id")

	assert.Empty(t, h.AnonymizeSessionID(""))
	assert.Empty(t, h.AnonymizeSubagentID(""))

	// Session and parent-session share domain so subagent joins match parent.
	parentRaw := "dynamo-root-session-123"
	assert.Equal(t, h.AnonymizeSessionID(parentRaw), h.AnonymizeSessionID(parentRaw))
	assert.NotContains(t, h.AnonymizeSessionID(parentRaw), parentRaw)

	// Subagent domain is separated from session domain even if given the same raw string.
	assert.NotEqual(t, h.AnonymizeSessionID(parentRaw), h.AnonymizeSubagentID(parentRaw))

	// Sampling boundary rates and session consistency.
	assert.False(t, h.ShouldSample("sess-x", 0.0))
	assert.True(t, h.ShouldSample("sess-x", 1.0))

	decision := h.ShouldSample("sess-consistent", 0.5)
	for range 20 {
		assert.Equal(t, decision, h.ShouldSample("sess-consistent", 0.5))
	}

	included := 0
	const trials = 2000
	for i := range trials {
		if h.ShouldSample(string(rune(i+1))+"-session", 0.25) {
			included++
		}
	}
	assert.InDelta(t, 0.25*trials, included, 0.06*trials)
}
