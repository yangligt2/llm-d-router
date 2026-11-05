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

// Command trace-export assembles per-request JSONL files emitted by
// trace-capture-producer into WEKA multi-turn trace datasets.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tracecapture"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintf(os.Stderr, "trace-export: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("trace-export", flag.ContinueOnError)
	fs.SetOutput(io.Discard)

	var (
		inputDir            string
		outputJSONL         string
		outputDir           string
		targetBlockSize     int
		hashIDScope         string
		lcpOverlap          float64
		maxIdleGap          float64
		dropClassifierCalls bool
		includeNonCompleted bool
	)

	fs.StringVar(&inputDir, "input-dir", "", "Directory containing capture-*.jsonl files")
	fs.StringVar(&outputJSONL, "output-jsonl", "", "Output file path for WEKA traces.jsonl")
	fs.StringVar(&outputDir, "output-dir", "", "Output directory path for per-trace trace_NNNN.json files")
	fs.IntVar(&targetBlockSize, "target-block-size", 0, "Target KV block size in tokens (integer multiple of captured block size; 0 keeps captured)")
	fs.StringVar(&hashIDScope, "hash-id-scope", tracecapture.HashIDScopeLocal, "Hash ID remapping scope: local or global")
	fs.Float64Var(&lcpOverlap, "lcp-overlap-threshold", tracecapture.DefaultLCPOverlapThreshold, "Block-prefix overlap threshold in (0, 1] for sessionless LCP chaining")
	fs.Float64Var(&maxIdleGap, "max-idle-gap-seconds", 0, "Maximum idle gap in seconds for sessionless LCP chaining (0 = unlimited)")
	fs.BoolVar(&dropClassifierCalls, "drop-classifier-calls", false, "Filter out tool-free requests with max_output_tokens <= 64")
	fs.BoolVar(&includeNonCompleted, "include-non-completed", false, "Include non-completed requests in assembled traces")

	if err := fs.Parse(args); err != nil {
		return err
	}

	paths := fs.Args()
	if inputDir != "" {
		cleanDir := filepath.Clean(inputDir)
		entries, err := os.ReadDir(cleanDir)
		if err != nil {
			return fmt.Errorf("read input-dir %q: %w", cleanDir, err)
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".jsonl") {
				paths = append(paths, filepath.Join(cleanDir, e.Name()))
			}
		}
	}
	slices.Sort(paths)
	if len(paths) == 0 {
		return errors.New("specify --input-dir or at least one capture JSONL file argument")
	}
	if (outputJSONL == "") == (outputDir == "") {
		return errors.New("specify exactly one of --output-jsonl or --output-dir")
	}

	records, err := tracecapture.LoadCapturedRecords(paths)
	if err != nil {
		return err
	}

	traces, err := tracecapture.AssembleWekaTraces(records, tracecapture.AssembleOptions{
		TargetBlockSize:     targetBlockSize,
		HashIDScope:         hashIDScope,
		LCPOverlapThreshold: lcpOverlap,
		MaxIdleGapSeconds:   maxIdleGap,
		DropClassifierCalls: dropClassifierCalls,
		IncludeNonCompleted: includeNonCompleted,
	})
	if err != nil {
		return err
	}

	if outputJSONL != "" {
		if err := tracecapture.WriteWekaTracesJSONL(outputJSONL, traces); err != nil {
			return err
		}
	} else {
		if err := tracecapture.WriteWekaTracesDir(outputDir, traces); err != nil {
			return err
		}
	}

	_, err = fmt.Fprintf(stdout, "exported %d traces from %d captured records\n", len(traces), len(records))
	return err
}
