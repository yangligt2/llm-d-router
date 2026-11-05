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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const (
	// SchemaVersion identifies the per-request JSONL capture format.
	SchemaVersion = "llm-d-trace-capture/v1"

	sinkDirPerm       = 0o700
	sinkFilePerm      = 0o600
	sinkWriterBufSize = 64 * 1024
)

// CapturedRecord is one anonymized per-request JSONL record emitted at end of stream.
type CapturedRecord struct {
	SchemaVersion     string   `json:"schema_version"`
	CaptureRunID      string   `json:"capture_run_id"`
	RequestID         string   `json:"request_id"`
	SessionID         string   `json:"session_id,omitempty"`
	ParentSessionID   string   `json:"parent_session_id,omitempty"`
	SubagentID        string   `json:"subagent_id,omitempty"`
	ArrivalUnixNano   int64    `json:"arrival_unix_ns"`
	FirstTokenSeconds float64  `json:"ttft_s,omitempty"`
	APITimeSeconds    float64  `json:"api_time_s"`
	StreamType        string   `json:"type"`
	Model             string   `json:"model,omitempty"`
	BlockSize         int      `json:"block_size"`
	TokenizedIn       int      `json:"tokenized_in"`
	UsageIn           int      `json:"usage_in,omitempty"`
	CachedTokens      int      `json:"cached_tokens,omitempty"`
	OutputTokens      int      `json:"out"`
	MaxOutputTokens   *int64   `json:"max_output_tokens,omitempty"`
	HasTools          bool     `json:"has_tools,omitempty"`
	Outcome           string   `json:"outcome"`
	BlockHashes       []string `json:"block_hashes"`
}

// SinkOptions configures a non-blocking rotating JSONL Sink.
type SinkOptions struct {
	OutputDir       string
	RunID           string
	BufferCapacity  int
	MaxFileBytes    int64
	MaxRotatedFiles int
	FlushInterval   time.Duration
	PluginName      string
}

// Sink persists CapturedRecord entries asynchronously via a bounded channel
// into size-rotated JSONL files confined to an os.Root directory.
type Sink struct {
	root            *os.Root
	runID           string
	maxFileBytes    int64
	maxRotatedFiles int
	flushInterval   time.Duration
	pluginName      string

	ch     chan CapturedRecord
	stopCh chan struct{}
	doneCh chan struct{}

	stopOnce sync.Once
	closed   atomic.Bool

	file         *os.File
	writer       *bufio.Writer
	fileSeq      int
	currentBytes int64
	rotatedFiles []string

	activeName     atomic.Value // string
	writtenRecords atomic.Int64
	droppedRecords atomic.Int64
	writeErrors    atomic.Int64
	rotations      atomic.Int64

	errMu   sync.Mutex
	lastErr error
}

// NewSink creates OutputDir (mode 0700 when newly created), opens the initial capture file (mode 0600),
// and starts the background writer goroutine bound to ctx.
func NewSink(ctx context.Context, opts SinkOptions) (*Sink, error) {
	if opts.OutputDir == "" {
		return nil, errors.New("sink outputDir is required")
	}
	cleanDir := filepath.Clean(opts.OutputDir)
	if err := ensureSinkDir(cleanDir, os.Chmod); err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(cleanDir)
	if err != nil {
		return nil, fmt.Errorf("open trace capture root %q: %w", cleanDir, err)
	}

	capBuf := opts.BufferCapacity
	if capBuf <= 0 {
		capBuf = DefaultBufferCapacity
	}
	maxBytes := opts.MaxFileBytes
	if maxBytes <= 0 {
		maxBytes = DefaultMaxFileBytes
	}
	maxFiles := opts.MaxRotatedFiles
	if maxFiles <= 0 {
		maxFiles = DefaultMaxRotatedFiles
	}
	flushEvery := opts.FlushInterval
	if flushEvery <= 0 {
		flushEvery = time.Second
	}

	s := &Sink{
		root:            root,
		runID:           sanitizeFileToken(opts.RunID),
		maxFileBytes:    maxBytes,
		maxRotatedFiles: maxFiles,
		flushInterval:   flushEvery,
		pluginName:      opts.PluginName,
		ch:              make(chan CapturedRecord, capBuf),
		stopCh:          make(chan struct{}),
		doneCh:          make(chan struct{}),
	}
	if err := s.openNextFile(); err != nil {
		_ = root.Close()
		return nil, err
	}

	go s.runWriter(ctx)
	return s, nil
}

func ensureSinkDir(cleanDir string, chmodFn func(string, os.FileMode) error) error {
	info, statErr := os.Stat(cleanDir)
	preExisting := statErr == nil && info.IsDir()
	if err := os.MkdirAll(cleanDir, sinkDirPerm); err != nil {
		return fmt.Errorf("create trace capture outputDir %q: %w", cleanDir, err)
	}
	if err := chmodFn(cleanDir, sinkDirPerm); err != nil {
		// Pre-existing volume mountpoints (e.g. Kubernetes emptyDir/PVC with fsGroup)
		// are owned by root while writable by the container GID; POSIX forbids non-owner chmod.
		if preExisting && errors.Is(err, os.ErrPermission) {
			return nil
		}
		return fmt.Errorf("chmod trace capture outputDir %q: %w", cleanDir, err)
	}
	return nil
}

// Enqueue submits rec without blocking. Returns false when the sink is closed
// or the channel buffer is full.
func (s *Sink) Enqueue(rec CapturedRecord) bool {
	if s.closed.Load() {
		s.droppedRecords.Add(1)
		traceCaptureRecordsDropped.WithLabelValues(s.pluginName, dropReasonSinkClosed).Inc()
		return false
	}
	select {
	case <-s.stopCh:
		s.droppedRecords.Add(1)
		traceCaptureRecordsDropped.WithLabelValues(s.pluginName, dropReasonSinkClosed).Inc()
		return false
	default:
	}
	select {
	case s.ch <- rec:
		return true
	default:
		s.droppedRecords.Add(1)
		traceCaptureRecordsDropped.WithLabelValues(s.pluginName, dropReasonBufferFull).Inc()
		return false
	}
}

// Close stops accepting new records, drains buffered records, flushes and syncs
// the active file, and closes the underlying os.Root. Safe to call multiple times.
func (s *Sink) Close() error {
	s.closed.Store(true)
	s.stopOnce.Do(func() {
		close(s.stopCh)
	})
	<-s.doneCh
	return s.getLastErr()
}

// WrittenRecords returns the total records flushed/written to the file writer.
func (s *Sink) WrittenRecords() int64 { return s.writtenRecords.Load() }

// DroppedRecords returns the total records dropped due to backpressure or closure.
func (s *Sink) DroppedRecords() int64 { return s.droppedRecords.Load() }

// WriteErrors returns the total file/serialization errors encountered.
func (s *Sink) WriteErrors() int64 { return s.writeErrors.Load() }

// Rotations returns the number of file rotations performed after startup.
func (s *Sink) Rotations() int64 { return s.rotations.Load() }

// ActiveFile returns the basename of the current JSONL file.
func (s *Sink) ActiveFile() string {
	v, _ := s.activeName.Load().(string)
	return v
}

func (s *Sink) runWriter(ctx context.Context) {
	defer close(s.doneCh)

	ticker := time.NewTicker(s.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case rec := <-s.ch:
			s.writeRecord(rec)
		case <-ticker.C:
			if s.writer != nil {
				if err := s.writer.Flush(); err != nil {
					s.recordErr(err)
				}
			}
		case <-ctx.Done():
			s.closed.Store(true)
			s.drainAndClose()
			return
		case <-s.stopCh:
			s.closed.Store(true)
			s.drainAndClose()
			return
		}
	}
}

func (s *Sink) drainAndClose() {
	for {
		select {
		case rec := <-s.ch:
			s.writeRecord(rec)
		default:
			if err := s.closeActiveFile(); err != nil {
				s.recordErr(err)
			}
			if s.root != nil {
				if err := s.root.Close(); err != nil {
					s.recordErr(err)
				}
			}
			return
		}
	}
}

func (s *Sink) writeRecord(rec CapturedRecord) {
	if rec.BlockHashes == nil {
		rec.BlockHashes = []string{}
	}
	line, err := json.Marshal(rec)
	if err != nil {
		s.recordWriteErr(err)
		return
	}
	line = append(line, '\n')

	if s.currentBytes > 0 && s.currentBytes+int64(len(line)) > s.maxFileBytes {
		if err := s.rotate(); err != nil {
			s.recordWriteErr(err)
			return
		}
	}
	if s.writer == nil {
		s.recordWriteErr(errors.New("sink writer unavailable"))
		return
	}
	n, err := s.writer.Write(line)
	s.currentBytes += int64(n)
	if err != nil {
		s.recordWriteErr(err)
		return
	}
	s.writtenRecords.Add(1)
	traceCaptureRecordsWritten.WithLabelValues(s.pluginName).Inc()
}

func (s *Sink) rotate() error {
	prevName := s.ActiveFile()
	if err := s.closeActiveFile(); err != nil {
		return err
	}
	if prevName != "" {
		s.rotatedFiles = append(s.rotatedFiles, prevName)
		for len(s.rotatedFiles) > s.maxRotatedFiles {
			oldest := s.rotatedFiles[0]
			s.rotatedFiles = s.rotatedFiles[1:]
			if err := s.root.Remove(oldest); err != nil && !errors.Is(err, os.ErrNotExist) {
				s.recordErr(err)
			}
		}
	}
	s.fileSeq++
	if err := s.openNextFile(); err != nil {
		return err
	}
	s.rotations.Add(1)
	traceCaptureRotations.WithLabelValues(s.pluginName).Inc()
	return nil
}

func (s *Sink) openNextFile() error {
	name := fmt.Sprintf("capture-%s-%06d.jsonl", s.runID, s.fileSeq)
	f, err := s.root.OpenFile(name, os.O_CREATE|os.O_WRONLY|os.O_APPEND, sinkFilePerm)
	if err != nil {
		return fmt.Errorf("open capture file %q: %w", name, err)
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return fmt.Errorf("stat capture file %q: %w", name, err)
	}
	s.file = f
	s.writer = bufio.NewWriterSize(f, sinkWriterBufSize)
	s.currentBytes = info.Size()
	s.activeName.Store(name)
	return nil
}

func (s *Sink) closeActiveFile() error {
	var firstErr error
	if s.writer != nil {
		if err := s.writer.Flush(); err != nil {
			firstErr = err
		}
		s.writer = nil
	}
	if s.file != nil {
		if err := s.file.Sync(); err != nil && firstErr == nil {
			firstErr = err
		}
		if err := s.file.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		s.file = nil
	}
	return firstErr
}

func (s *Sink) recordWriteErr(err error) {
	s.writeErrors.Add(1)
	traceCaptureRecordsDropped.WithLabelValues(s.pluginName, dropReasonWriteError).Inc()
	s.recordErr(err)
}

func (s *Sink) recordErr(err error) {
	s.errMu.Lock()
	s.lastErr = err
	s.errMu.Unlock()
}

func (s *Sink) getLastErr() error {
	s.errMu.Lock()
	defer s.errMu.Unlock()
	return s.lastErr
}

func sanitizeFileToken(s string) string {
	if s == "" {
		return "default"
	}
	out := make([]byte, 0, len(s))
	for i := range len(s) {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		return "default"
	}
	return string(out)
}
