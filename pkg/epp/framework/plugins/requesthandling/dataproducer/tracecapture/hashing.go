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
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strconv"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
)

const (
	domainBlock     = "llm-d-trace-block-v1\x00"
	domainCacheSalt = "llm-d-trace-cachesalt-v1\x00"
	domainSession   = "llm-d-trace-session-v1\x00"
	domainSubagent  = "llm-d-trace-subagent-v1\x00"
	domainSample    = "llm-d-trace-sample-v1\x00"

	truncatedDigestBytes = 16 // 128-bit hex-encoded block and identifier tags
)

// Hasher computes keyed, prefix-chained block digests and domain-separated
// identifier tags using a per-instance HMAC-SHA256 secret key.
type Hasher struct {
	key          [32]byte
	captureRunID string
}

// NewHasher initializes a Hasher with an independent CSPRNG 256-bit HMAC key
// and an independent 128-bit public capture run identifier.
func NewHasher() (*Hasher, error) {
	var key [32]byte
	if _, err := rand.Read(key[:]); err != nil {
		return nil, fmt.Errorf("generate trace capture HMAC key: %w", err)
	}
	var runBytes [16]byte
	if _, err := rand.Read(runBytes[:]); err != nil {
		return nil, fmt.Errorf("generate trace capture run id: %w", err)
	}
	return &Hasher{
		key:          key,
		captureRunID: hex.EncodeToString(runBytes[:]),
	}, nil
}

// NewHasherWithKey constructs a Hasher from a caller-supplied key and run ID.
func NewHasherWithKey(key [32]byte, captureRunID string) *Hasher {
	return &Hasher{
		key:          key,
		captureRunID: captureRunID,
	}
}

// CaptureRunID returns the public identifier for this capture instance.
func (h *Hasher) CaptureRunID() string {
	return h.captureRunID
}

// HashTokenizedRequest hashes all complete blocks across prompts in tr.
// Trailing partial blocks (len(tokens) % blockSize) are omitted.
func (h *Hasher) HashTokenizedRequest(tr *fwkrequest.TokenizedRequest, blockSize int) []string {
	if tr == nil || blockSize <= 0 {
		return []string{}
	}
	totalBlocks := 0
	for _, p := range tr.Prompts {
		totalBlocks += len(p.TokenIDs) / blockSize
	}
	if totalBlocks == 0 {
		return []string{}
	}
	out := make([]string, 0, totalBlocks)
	for _, p := range tr.Prompts {
		out = append(out, h.HashPromptBlocks(p, tr.CacheSalt, blockSize)...)
	}
	return out
}

// HashPromptBlocks computes prefix-chained 128-bit hex digests over complete
// blocks of size blockSize. Multimodal placeholders overlapping a block mix
// their content hash into that block's digest, propagating to all subsequent blocks.
func (h *Hasher) HashPromptBlocks(prompt fwkrequest.PromptTokens, cacheSalt string, blockSize int) []string {
	if blockSize <= 0 || len(prompt.TokenIDs) < blockSize {
		return []string{}
	}
	numBlocks := len(prompt.TokenIDs) / blockSize
	out := make([]string, numBlocks)

	var prevDigest [sha256.Size]byte
	if cacheSalt != "" {
		mac := hmac.New(sha256.New, h.key[:])
		_, _ = mac.Write([]byte(domainCacheSalt))
		_, _ = mac.Write([]byte(cacheSalt))
		copy(prevDigest[:], mac.Sum(nil))
	}

	var tokBuf [4]byte
	var numBuf [32]byte
	var sumBuf [sha256.Size]byte

	for b := range numBlocks {
		start := b * blockSize
		end := start + blockSize

		mac := hmac.New(sha256.New, h.key[:])
		_, _ = mac.Write([]byte(domainBlock))
		_, _ = mac.Write(prevDigest[:])
		writeIntTag(mac, &numBuf, int64(b))
		writeIntTag(mac, &numBuf, int64(blockSize))

		for _, tok := range prompt.TokenIDs[start:end] {
			binary.LittleEndian.PutUint32(tokBuf[:], tok)
			_, _ = mac.Write(tokBuf[:])
		}

		for _, mm := range prompt.MultiModalFeatures {
			if mm.Length <= 0 {
				continue
			}
			if mm.Offset < end && mm.Offset+mm.Length > start {
				writeBytesTag(mac, &numBuf, string(mm.Modality))
				writeBytesTag(mac, &numBuf, mm.Hash)
				writeIntTag(mac, &numBuf, int64(mm.Offset))
				writeIntTag(mac, &numBuf, int64(mm.Length))
			}
		}

		sum := mac.Sum(sumBuf[:0])
		copy(prevDigest[:], sum)
		out[b] = hex.EncodeToString(sum[:truncatedDigestBytes])
	}

	return out
}

// AnonymizeSessionID returns a 128-bit hex HMAC tag for session and parent-session
// identifiers. Both share one domain so subagent parent links join directly.
func (h *Hasher) AnonymizeSessionID(raw string) string {
	return h.anonymizeWithDomain(domainSession, raw)
}

// AnonymizeSubagentID returns a 128-bit hex HMAC tag for subagent identifiers.
func (h *Hasher) AnonymizeSubagentID(raw string) string {
	return h.anonymizeWithDomain(domainSubagent, raw)
}

func (h *Hasher) anonymizeWithDomain(domain, raw string) string {
	if raw == "" {
		return ""
	}
	mac := hmac.New(sha256.New, h.key[:])
	_, _ = mac.Write([]byte(domain))
	_, _ = mac.Write([]byte(raw))
	var sumBuf [sha256.Size]byte
	sum := mac.Sum(sumBuf[:0])
	return hex.EncodeToString(sum[:truncatedDigestBytes])
}

// ShouldSample deterministically maps sampleKey into [0, 1) and returns whether
// it falls below rate, keeping multi-turn sessions and subagents intact.
func (h *Hasher) ShouldSample(sampleKey string, rate float64) bool {
	if rate <= 0 {
		return false
	}
	if rate >= 1 {
		return true
	}
	mac := hmac.New(sha256.New, h.key[:])
	_, _ = mac.Write([]byte(domainSample))
	_, _ = mac.Write([]byte(sampleKey))
	var sumBuf [sha256.Size]byte
	sum := mac.Sum(sumBuf[:0])
	v := binary.BigEndian.Uint64(sum[:8])
	u := float64(v>>11) * (1.0 / (1 << 53))
	return u < rate
}

// DownsampleBlockHashes converts prefix-chained digests captured at srcBlockSize
// to targetBlockSize (an integer multiple k * srcBlockSize) by selecting every
// k-th chained digest. Trailing blocks that do not fill targetBlockSize are dropped.
func DownsampleBlockHashes(hashes []string, srcBlockSize, targetBlockSize int) ([]string, error) {
	if srcBlockSize <= 0 || targetBlockSize <= 0 {
		return nil, fmt.Errorf("block sizes must be > 0 (src=%d, target=%d)", srcBlockSize, targetBlockSize)
	}
	if targetBlockSize < srcBlockSize || targetBlockSize%srcBlockSize != 0 {
		return nil, fmt.Errorf("targetBlockSize (%d) must be an integer multiple of srcBlockSize (%d)", targetBlockSize, srcBlockSize)
	}
	if targetBlockSize == srcBlockSize {
		out := make([]string, len(hashes))
		copy(out, hashes)
		return out, nil
	}
	factor := targetBlockSize / srcBlockSize
	n := len(hashes) / factor
	out := make([]string, n)
	for i := range n {
		out[i] = hashes[(i+1)*factor-1]
	}
	return out, nil
}

func writeIntTag(mac interface{ Write([]byte) (int, error) }, buf *[32]byte, val int64) {
	b := strconv.AppendInt(buf[:0], val, 10)
	b = append(b, 0)
	_, _ = mac.Write(b)
}

func writeBytesTag(mac interface{ Write([]byte) (int, error) }, buf *[32]byte, s string) {
	writeIntTag(mac, buf, int64(len(s)))
	_, _ = mac.Write([]byte(s))
}
