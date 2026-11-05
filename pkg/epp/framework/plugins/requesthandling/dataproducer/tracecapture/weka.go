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
	"bufio"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

const (
	// HashIDScopeLocal remaps block digests to 1-based integers independently per trace.
	HashIDScopeLocal = "local"
	// HashIDScopeGlobal remaps block digests to 1-based integers across all assembled traces.
	HashIDScopeGlobal = "global"

	// DefaultLCPOverlapThreshold is the default block-prefix overlap fraction
	// used to chain sessionless requests into reconstructed conversations.
	DefaultLCPOverlapThreshold = 0.90

	// ClassifierMaxOutputTokens is the max_output_tokens cap below or at which
	// tool-free auxiliary requests are classified as lightweight utility calls.
	ClassifierMaxOutputTokens = 64

	entryTypeSubagent = "subagent"
	maxScannerLine    = 32 << 20 // 32 MiB max JSONL line for ultra-long-context prompts
)

// WekaTrace is a single multi-turn session in the WEKA agentic trace format,
// compatible with aiperf and inference-perf WEKA trace replay engines.
type WekaTrace struct {
	ID           string      `json:"id"`
	Models       []string    `json:"models"`
	BlockSize    int         `json:"block_size"`
	HashIDScope  string      `json:"hash_id_scope"`
	ToolTokens   int         `json:"tool_tokens"`
	SystemTokens int         `json:"system_tokens"`
	Requests     []WekaEntry `json:"requests"`
	Totals       *WekaTotals `json:"totals,omitempty"`
}

// WekaEntry is one discriminated element ("n", "s", or "subagent") in WekaTrace.Requests.
type WekaEntry struct {
	Turn     *WekaTurnRequest
	Subagent *WekaSubagentEntry
}

// MarshalJSON serializes the underlying Turn or Subagent variant directly.
func (e WekaEntry) MarshalJSON() ([]byte, error) {
	switch {
	case e.Subagent != nil:
		return json.Marshal(e.Subagent)
	case e.Turn != nil:
		return json.Marshal(e.Turn)
	default:
		return nil, errors.New("empty WekaEntry")
	}
}

// UnmarshalJSON decodes a WekaEntry based on its "type" discriminator.
func (e *WekaEntry) UnmarshalJSON(data []byte) error {
	var disc struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(data, &disc); err != nil {
		return err
	}
	if disc.Type == entryTypeSubagent {
		var sub WekaSubagentEntry
		if err := json.Unmarshal(data, &sub); err != nil {
			return err
		}
		e.Subagent = &sub
		e.Turn = nil
		return nil
	}
	var turn WekaTurnRequest
	if err := json.Unmarshal(data, &turn); err != nil {
		return err
	}
	e.Turn = &turn
	e.Subagent = nil
	return nil
}

// WekaTurnRequest represents a normal ("n") or streaming ("s") LLM turn.
type WekaTurnRequest struct {
	T         float64  `json:"t"`
	Type      string   `json:"type"`
	Model     string   `json:"model"`
	In        int      `json:"in"`
	Out       int      `json:"out"`
	HashIDs   []int    `json:"hash_ids"`
	APITime   *float64 `json:"api_time,omitempty"`
	ThinkTime *float64 `json:"think_time,omitempty"`
	TTFT      *float64 `json:"ttft,omitempty"`
}

// WekaSubagentEntry represents a nested subagent span within a parent trace.
// Inner Requests use Type "n" to satisfy inference-perf WekaSubagentEntry validation.
type WekaSubagentEntry struct {
	T            float64           `json:"t"`
	Type         string            `json:"type"`
	AgentID      string            `json:"agent_id"`
	SubagentType string            `json:"subagent_type"`
	DurationMs   *int              `json:"duration_ms,omitempty"`
	Requests     []WekaTurnRequest `json:"requests"`
	ToolTokens   int               `json:"tool_tokens"`
	SystemTokens int               `json:"system_tokens"`
}

// WekaTokenPair holds aggregated input/output token counts.
type WekaTokenPair struct {
	Input  int `json:"input"`
	Output int `json:"output"`
}

// WekaTotals summarizes token usage across parent and subagent turns.
type WekaTotals struct {
	ParentTokens   WekaTokenPair `json:"parent_tokens"`
	SubagentTokens WekaTokenPair `json:"subagent_tokens"`
	CombinedTokens WekaTokenPair `json:"combined_tokens"`
	SubagentCount  int           `json:"subagent_count"`
}

// AssembleOptions configures offline conversion of CapturedRecord slices into WekaTrace entries.
type AssembleOptions struct {
	// TargetBlockSize downsamples captured block digests to an integer multiple of the captured block size.
	// Zero retains the captured block size.
	TargetBlockSize int
	// HashIDScope selects "local" (default, 1-based per trace) or "global" (1-based across dataset).
	HashIDScope string
	// LCPOverlapThreshold sets the minimum block-prefix overlap ratio in (0, 1] for chaining
	// sessionless requests into conversations. Zero defaults to 0.90.
	LCPOverlapThreshold float64
	// MaxIdleGapSeconds splits sessions or limits LCP recovery across idle gaps when > 0.
	MaxIdleGapSeconds float64
	// DropClassifierCalls filters out requests with max_output_tokens <= 64 and has_tools == false.
	DropClassifierCalls bool
	// IncludeNonCompleted retains non-completed records instead of filtering them out.
	IncludeNonCompleted bool
}

type preparedRecord struct {
	rec    CapturedRecord
	hashes []string
}

type subagentGroup struct {
	agentID string
	records []preparedRecord
	startNs int64
	endNs   int64
}

type sessionBucket struct {
	key       string
	parent    []preparedRecord
	subagents map[string]*subagentGroup
	subOrder  []string
	startNs   int64
}

// AssembleWekaTraces groups per-request CapturedRecord entries into multi-turn
// WEKA traces sorted chronologically by trace start time.
func AssembleWekaTraces(records []CapturedRecord, opts AssembleOptions) ([]WekaTrace, error) {
	scope := opts.HashIDScope
	if scope == "" {
		scope = HashIDScopeLocal
	}
	if scope != HashIDScopeLocal && scope != HashIDScopeGlobal {
		return nil, fmt.Errorf("invalid hash_id_scope %q: must be %q or %q", scope, HashIDScopeLocal, HashIDScopeGlobal)
	}
	lcpThreshold := opts.LCPOverlapThreshold
	if lcpThreshold == 0 {
		lcpThreshold = DefaultLCPOverlapThreshold
	}
	if lcpThreshold < 0 || lcpThreshold > 1 || math.IsNaN(lcpThreshold) {
		return nil, fmt.Errorf("lcpOverlapThreshold must be in (0, 1], got %v", lcpThreshold)
	}

	filtered := filterAndSortRecords(records, opts)
	if len(filtered) == 0 {
		return nil, nil
	}

	effectiveBlockSize := opts.TargetBlockSize
	if effectiveBlockSize == 0 {
		effectiveBlockSize = filtered[0].BlockSize
	}
	if effectiveBlockSize <= 0 {
		return nil, fmt.Errorf("invalid block_size %d", effectiveBlockSize)
	}

	prepared := make([]preparedRecord, 0, len(filtered))
	for _, r := range filtered {
		if r.BlockSize <= 0 {
			return nil, fmt.Errorf("record %q has invalid block_size %d", r.RequestID, r.BlockSize)
		}
		hashes, err := DownsampleBlockHashes(r.BlockHashes, r.BlockSize, effectiveBlockSize)
		if err != nil {
			return nil, fmt.Errorf("record %q: %w", r.RequestID, err)
		}
		prepared = append(prepared, preparedRecord{rec: r, hashes: hashes})
	}

	buckets := groupIntoSessionBuckets(prepared, lcpThreshold, opts.MaxIdleGapSeconds)
	slices.SortFunc(buckets, func(a, b *sessionBucket) int {
		if c := cmp.Compare(a.startNs, b.startNs); c != 0 {
			return c
		}
		return cmp.Compare(a.key, b.key)
	})

	globalMap := make(map[string]int)
	traces := make([]WekaTrace, 0, len(buckets))
	for idx, b := range buckets {
		idMap := globalMap
		if scope == HashIDScopeLocal {
			idMap = make(map[string]int)
		}
		traces = append(traces, buildWekaTrace(idx, b, effectiveBlockSize, scope, idMap))
	}
	return traces, nil
}

func filterAndSortRecords(records []CapturedRecord, opts AssembleOptions) []CapturedRecord {
	seenReq := make(map[string]struct{}, len(records))
	out := make([]CapturedRecord, 0, len(records))
	for _, r := range records {
		if r.RequestID != "" {
			if _, dup := seenReq[r.RequestID]; dup {
				continue
			}
			seenReq[r.RequestID] = struct{}{}
		}
		if !opts.IncludeNonCompleted && r.Outcome != "" && r.Outcome != outcomeCompleted {
			continue
		}
		if opts.DropClassifierCalls && r.MaxOutputTokens != nil && *r.MaxOutputTokens <= ClassifierMaxOutputTokens && !r.HasTools {
			continue
		}
		out = append(out, r)
	}
	slices.SortFunc(out, func(a, b CapturedRecord) int {
		if c := cmp.Compare(a.ArrivalUnixNano, b.ArrivalUnixNano); c != 0 {
			return c
		}
		return cmp.Compare(a.RequestID, b.RequestID)
	})
	return out
}

type lcpCandidate struct {
	bucket     *sessionBucket
	lastHashes []string
	lastEndNs  int64
}

func groupIntoSessionBuckets(records []preparedRecord, lcpThreshold, maxIdleGapSec float64) []*sessionBucket {
	bySession := make(map[string]*sessionBucket)
	var buckets []*sessionBucket
	var lcpActive []*lcpCandidate

	getOrCreate := func(key string, arrivalNs int64) *sessionBucket {
		if b, ok := bySession[key]; ok {
			if arrivalNs < b.startNs {
				b.startNs = arrivalNs
			}
			return b
		}
		b := &sessionBucket{
			key:       key,
			subagents: make(map[string]*subagentGroup),
			startNs:   arrivalNs,
		}
		bySession[key] = b
		buckets = append(buckets, b)
		return b
	}

	lcpSeq := 0
	for _, pr := range records {
		r := pr.rec
		isSubagent := r.ParentSessionID != "" || r.SubagentID != ""
		if isSubagent {
			parentKey := r.ParentSessionID
			if parentKey == "" {
				parentKey = r.SessionID
			}
			if parentKey == "" {
				parentKey = "subagent-root-" + r.SubagentID
			}
			agentKey := r.SubagentID
			if agentKey == "" {
				agentKey = r.SessionID
			}
			if agentKey == "" {
				agentKey = r.RequestID
			}
			b := getOrCreate("sess:"+parentKey, r.ArrivalUnixNano)
			grp, ok := b.subagents[agentKey]
			endNs := r.ArrivalUnixNano + int64(r.APITimeSeconds*1e9)
			if !ok {
				grp = &subagentGroup{
					agentID: shortenID(agentKey),
					startNs: r.ArrivalUnixNano,
					endNs:   endNs,
				}
				b.subagents[agentKey] = grp
				b.subOrder = append(b.subOrder, agentKey)
			} else {
				if r.ArrivalUnixNano < grp.startNs {
					grp.startNs = r.ArrivalUnixNano
				}
				if endNs > grp.endNs {
					grp.endNs = endNs
				}
			}
			grp.records = append(grp.records, pr)
			continue
		}

		if r.SessionID != "" {
			b := getOrCreate("sess:"+r.SessionID, r.ArrivalUnixNano)
			b.parent = append(b.parent, pr)
			continue
		}

		// Sessionless request: match via longest-common-prefix against active sessionless chains.
		bestIdx := -1
		bestLCP := 0
		for i, cand := range lcpActive {
			if maxIdleGapSec > 0 && float64(r.ArrivalUnixNano-cand.lastEndNs)/1e9 > maxIdleGapSec {
				continue
			}
			if len(cand.lastHashes) == 0 || len(pr.hashes) == 0 {
				continue
			}
			lcp := longestCommonPrefix(cand.lastHashes, pr.hashes)
			overlap := float64(lcp) / float64(len(cand.lastHashes))
			if overlap >= lcpThreshold && (lcp > bestLCP || (lcp == bestLCP && bestIdx >= 0 && cand.lastEndNs > lcpActive[bestIdx].lastEndNs)) {
				bestLCP = lcp
				bestIdx = i
			}
		}

		endNs := r.ArrivalUnixNano + int64(r.APITimeSeconds*1e9)
		if bestIdx >= 0 {
			cand := lcpActive[bestIdx]
			cand.bucket.parent = append(cand.bucket.parent, pr)
			cand.lastHashes = pr.hashes
			cand.lastEndNs = endNs
		} else {
			key := fmt.Sprintf("lcp:%06d", lcpSeq)
			lcpSeq++
			b := getOrCreate(key, r.ArrivalUnixNano)
			b.parent = append(b.parent, pr)
			lcpActive = append(lcpActive, &lcpCandidate{
				bucket:     b,
				lastHashes: pr.hashes,
				lastEndNs:  endNs,
			})
		}
	}

	return buckets
}

type rawTimelineItem struct {
	startNs  int64
	orderIdx int
	parent   *preparedRecord
	subagent *subagentGroup
}

func buildWekaTrace(idx int, b *sessionBucket, blockSize int, scope string, idMap map[string]int) WekaTrace {
	items := make([]rawTimelineItem, 0, len(b.parent)+len(b.subOrder))
	order := 0
	for i := range b.parent {
		items = append(items, rawTimelineItem{
			startNs:  b.parent[i].rec.ArrivalUnixNano,
			orderIdx: order,
			parent:   &b.parent[i],
		})
		order++
	}
	for _, subKey := range b.subOrder {
		grp := b.subagents[subKey]
		items = append(items, rawTimelineItem{
			startNs:  grp.startNs,
			orderIdx: order,
			subagent: grp,
		})
		order++
	}
	slices.SortFunc(items, func(a, b rawTimelineItem) int {
		if c := cmp.Compare(a.startNs, b.startNs); c != 0 {
			return c
		}
		return cmp.Compare(a.orderIdx, b.orderIdx)
	})

	traceStartNs := b.startNs
	if len(items) > 0 && items[0].startNs < traceStartNs {
		traceStartNs = items[0].startNs
	}

	assignHashIDs := func(hexHashes []string) []int {
		ids := make([]int, len(hexHashes))
		for i, h := range hexHashes {
			id, ok := idMap[h]
			if !ok {
				id = len(idMap) + 1
				idMap[h] = id
			}
			ids[i] = id
		}
		return ids
	}

	var models []string
	seenModel := make(map[string]struct{})
	recordModel := func(m string) string {
		if m == "" {
			m = "unknown"
		}
		if _, ok := seenModel[m]; !ok {
			seenModel[m] = struct{}{}
			models = append(models, m)
		}
		return m
	}

	var entries []WekaEntry
	var totals WekaTotals
	var prevParentEndSec float64
	hasPrevParent := false

	for _, item := range items {
		if item.parent != nil {
			pr := item.parent
			r := pr.rec
			tSec := roundSeconds(max(0.0, float64(r.ArrivalUnixNano-traceStartNs)/1e9))
			apiSec := roundSeconds(max(0.0, r.APITimeSeconds))
			thinkSec := 0.0
			if hasPrevParent {
				thinkSec = roundSeconds(max(0.0, tSec-prevParentEndSec))
			}
			prevParentEndSec = tSec + apiSec
			hasPrevParent = true

			inTokens := r.TokenizedIn
			if inTokens == 0 && r.UsageIn > 0 {
				inTokens = r.UsageIn
			}
			totals.ParentTokens.Input += inTokens
			totals.ParentTokens.Output += r.OutputTokens

			sType := r.StreamType
			if sType != streamTypeStreaming {
				sType = streamTypeNormal
			}
			var ttftPtr *float64
			if sType == streamTypeStreaming && r.FirstTokenSeconds > 0 {
				ttft := roundSeconds(r.FirstTokenSeconds)
				ttftPtr = &ttft
			}

			turn := &WekaTurnRequest{
				T:         tSec,
				Type:      sType,
				Model:     recordModel(r.Model),
				In:        inTokens,
				Out:       r.OutputTokens,
				HashIDs:   assignHashIDs(pr.hashes),
				APITime:   &apiSec,
				ThinkTime: &thinkSec,
				TTFT:      ttftPtr,
			}
			entries = append(entries, WekaEntry{Turn: turn})
			continue
		}

		grp := item.subagent
		totals.SubagentCount++
		subStartSec := roundSeconds(max(0.0, float64(grp.startNs-traceStartNs)/1e9))
		durMs := int(math.Round(max(0.0, float64(grp.endNs-grp.startNs)/1e6)))

		inner := make([]WekaTurnRequest, 0, len(grp.records))
		var prevSubEndSec float64
		for i, pr := range grp.records {
			r := pr.rec
			relT := roundSeconds(max(0.0, float64(r.ArrivalUnixNano-grp.startNs)/1e9))
			apiSec := roundSeconds(max(0.0, r.APITimeSeconds))
			thinkSec := 0.0
			if i > 0 {
				thinkSec = roundSeconds(max(0.0, relT-prevSubEndSec))
			}
			prevSubEndSec = relT + apiSec

			inTokens := r.TokenizedIn
			if inTokens == 0 && r.UsageIn > 0 {
				inTokens = r.UsageIn
			}
			totals.SubagentTokens.Input += inTokens
			totals.SubagentTokens.Output += r.OutputTokens

			inner = append(inner, WekaTurnRequest{
				T:         relT,
				Type:      streamTypeNormal,
				Model:     recordModel(r.Model),
				In:        inTokens,
				Out:       r.OutputTokens,
				HashIDs:   assignHashIDs(pr.hashes),
				APITime:   &apiSec,
				ThinkTime: &thinkSec,
			})
		}

		entries = append(entries, WekaEntry{
			Subagent: &WekaSubagentEntry{
				T:            subStartSec,
				Type:         entryTypeSubagent,
				AgentID:      grp.agentID,
				SubagentType: "captured",
				DurationMs:   &durMs,
				Requests:     inner,
			},
		})
	}

	if len(models) == 0 {
		models = []string{"unknown"}
	}
	if entries == nil {
		entries = []WekaEntry{}
	}
	totals.CombinedTokens = WekaTokenPair{
		Input:  totals.ParentTokens.Input + totals.SubagentTokens.Input,
		Output: totals.ParentTokens.Output + totals.SubagentTokens.Output,
	}

	return WekaTrace{
		ID:          fmt.Sprintf("trace_%04d", idx),
		Models:      models,
		BlockSize:   blockSize,
		HashIDScope: scope,
		Requests:    entries,
		Totals:      &totals,
	}
}

// LoadCapturedRecords reads CapturedRecord JSONL lines from one or more file paths.
func LoadCapturedRecords(paths []string) ([]CapturedRecord, error) {
	var all []CapturedRecord
	for _, p := range paths {
		clean := filepath.Clean(p)
		root, err := os.OpenRoot(filepath.Dir(clean))
		if err != nil {
			return nil, fmt.Errorf("open directory for %q: %w", clean, err)
		}
		f, err := root.Open(filepath.Base(clean))
		_ = root.Close()
		if err != nil {
			return nil, fmt.Errorf("open %q: %w", clean, err)
		}

		scanner := bufio.NewScanner(f)
		scanner.Buffer(make([]byte, 64*1024), maxScannerLine)
		lineNo := 0
		for scanner.Scan() {
			lineNo++
			line := strings.TrimSpace(scanner.Text())
			if line == "" {
				continue
			}
			var rec CapturedRecord
			if err := json.Unmarshal([]byte(line), &rec); err != nil {
				_ = f.Close()
				return nil, fmt.Errorf("%s:%d: %w", clean, lineNo, err)
			}
			all = append(all, rec)
		}
		scanErr := scanner.Err()
		_ = f.Close()
		if scanErr != nil {
			return nil, fmt.Errorf("read %q: %w", clean, scanErr)
		}
	}
	return all, nil
}

// WriteWekaTracesJSONL writes traces as one JSON object per line to path (mode 0600).
func WriteWekaTracesJSONL(path string, traces []WekaTrace) error {
	clean := filepath.Clean(path)
	dir := filepath.Dir(clean)
	if err := os.MkdirAll(dir, sinkDirPerm); err != nil {
		return fmt.Errorf("create output directory %q: %w", dir, err)
	}
	root, err := os.OpenRoot(dir)
	if err != nil {
		return fmt.Errorf("open output directory %q: %w", dir, err)
	}
	defer root.Close()

	f, err := root.OpenFile(filepath.Base(clean), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
	if err != nil {
		return fmt.Errorf("open %q: %w", clean, err)
	}
	bw := bufio.NewWriterSize(f, sinkWriterBufSize)
	for _, tr := range traces {
		b, err := json.Marshal(tr)
		if err != nil {
			_ = f.Close()
			return err
		}
		if _, err := bw.Write(b); err != nil {
			_ = f.Close()
			return err
		}
		if err := bw.WriteByte('\n'); err != nil {
			_ = f.Close()
			return err
		}
	}
	if err := bw.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// WriteWekaTracesDir writes each trace as <trace.ID>.json inside dir (mode 0700/0600).
func WriteWekaTracesDir(dir string, traces []WekaTrace) error {
	cleanDir := filepath.Clean(dir)
	if err := os.MkdirAll(cleanDir, sinkDirPerm); err != nil {
		return fmt.Errorf("create output directory %q: %w", cleanDir, err)
	}
	root, err := os.OpenRoot(cleanDir)
	if err != nil {
		return fmt.Errorf("open output directory %q: %w", cleanDir, err)
	}
	defer root.Close()

	for _, tr := range traces {
		name := sanitizeFileToken(tr.ID) + ".json"
		b, err := json.Marshal(tr)
		if err != nil {
			return err
		}
		f, err := root.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, sinkFilePerm)
		if err != nil {
			return fmt.Errorf("open %q: %w", name, err)
		}
		_, writeErr := f.Write(b)
		closeErr := f.Close()
		if writeErr != nil {
			return writeErr
		}
		if closeErr != nil {
			return closeErr
		}
	}
	return nil
}

func longestCommonPrefix(a, b []string) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func shortenID(id string) string {
	if len(id) <= 12 {
		return id
	}
	return id[:12]
}

func roundSeconds(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
