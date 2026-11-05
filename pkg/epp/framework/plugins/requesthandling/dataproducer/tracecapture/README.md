# Trace Capture Producer (`trace-capture-producer`)

`trace-capture-producer` records anonymized KV-block prefix hashes, session and subagent correlation tags, token counts, and request timing from live EPP traffic. The companion CLI `cmd/trace-export` assembles captured per-request JSONL files into multi-turn WEKA agentic traces (`traces.jsonl` or per-trace `trace_NNNN.json`) consumed by `aiperf` and `inference-perf`.

## Lifecycle and Data Flow

- **Upstream dependency**: Consumes `TokenizedPrompt` from `token-producer`. By default (`allowEstimateBackend: false`), startup validation requires `token-producer` to run with `backend: vllm` so block hashes reflect model tokenizer output rather than byte-packed estimates.
- **`Produce`**: Extracts the tokenized prompt before the handler releases the parsed request body, drops trailing partial blocks (`len(tokens) % blockSizeTokens`), computes prefix-chained 128-bit HMAC-SHA256 digests (`block_hashes`), mixes overlapping multimodal content hashes (`PromptTokens.MultiModalFeatures`) into block digests, anonymizes session/parent-session/subagent identifiers, and records `max_output_tokens` and tool-definition presence (`has_tools`).
- **`ResponseHeader` / `ResponseBody`**: Records HTTP response status and, at `EndOfStream`, computes `ttft_s` (streaming only), `api_time_s`, `out` (`Usage.CompletionTokens`, falling back to `StreamedEvents` when streaming usage is absent), `usage_in`, and `cached_tokens`. Non-completed requests (HTTP non-200, evicted, cancelled) are skipped unless `includeNonCompleted: true`.
- **`RequestAbandoned`**: Releases per-request `PluginState` when admission or scheduling rejects a request prior to dispatch.
- **Persistence**: `ResponseBody` enqueues `CapturedRecord` structs without blocking onto a bounded channel drained by a single background writer goroutine into size-rotated JSONL files (`0700` directory, `0600` files) confined via `os.Root`.

## Anonymization

A 256-bit secret key is generated from `crypto/rand` at plugin startup, kept in process memory only, and excluded from logs, metrics, and `/debug/plugins/state`.

- **Block digests**: `HMAC-SHA256(key, domainBlock || prevDigest[32] || blockIdx || blockSize || blockTokenIDs || overlappingMMFeatures)`, truncated to 128 bits (32 hex characters).
- **Session correlation**: `session_id` and `parent_session_id` use the same HMAC domain (`domainSession`) so subagent turns carrying `x-dynamo-parent-session-id` join to their parent session's `session_id` without persisting raw header values.
- **Session-consistent sampling**: Sampling hashes the root session key (`parentSessionSources` when set, else `sessionSources`, else EPP `RequestID`), keeping multi-turn conversations and their subagents intact when `0 < sampleRate < 1`.

## Configuration

`trace-capture-producer` is registered at `Alpha` stability; pass `--allow-experimental-plugins` on the EPP command line.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `outputDir` | `string` | (required) | Directory path where `capture-<runID>-<seq>.jsonl` files are written. |
| `blockSizeTokens` | `int` | `64` | Tokens per KV block (`[1, 65536]`). Capture at `16` when offline export needs both `16` and `64` via integer downsampling. |
| `sampleRate` | `float64` | `1.0` | Fraction `[0.0, 1.0]` of sessions captured. |
| `bufferCapacity` | `int` | `4096` | Capacity of the in-memory channel feeding the file writer. |
| `maxFileBytes` | `int64` | `134217728` (128 MiB) | Byte threshold at which the active JSONL file rotates. |
| `maxRotatedFiles` | `int` | `16` | Maximum closed JSONL files retained alongside the active file. |
| `flushIntervalSeconds` | `float64` | `1.0` | Periodic flush interval of the buffered file writer. |
| `allowEstimateBackend` | `bool` | `false` | Permits `token-producer` `estimate` backend for offline/unit tests. |
| `includeNonCompleted` | `bool` | `false` | Emits evicted, cancelled, and failed requests in addition to `completed`. |
| `sessionSources` | `[]SessionIDSource` | `agent-identity` attr, `x-session-id`, `x-dynamo-session-id`, `x-correlation-id`, `x-smg-routing-key`, Claude/OpenCode/Codex headers, `x-llm-d-inference-fairness-id`, plus non-default `FairnessID` | Priority-ordered session identifier sources. |
| `parentSessionSources` | `[]SessionIDSource` | `x-dynamo-parent-session-id` | Priority-ordered parent session identifier sources for subagent calls. |
| `subagentSources` | `[]SessionIDSource` | `x-claude-code-agent-id` | Priority-ordered subagent identifier sources. |

### Example `EndpointPickerConfig`

```yaml
apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - type: agent-identity
  - type: token-producer
    parameters:
      backend: vllm
      modelName: Qwen/Qwen2.5-0.5B-Instruct
  - type: trace-capture-producer
    parameters:
      outputDir: /var/log/llm-d/traces
      blockSizeTokens: 64
      sampleRate: 1.0
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: max-score-picker
```

## Exporting WEKA Traces (`cmd/trace-export`)

Convert captured per-request JSONL files (from one or multiple EPP replicas sharing the same capture run, or per-replica directories) into WEKA traces:

```bash
# Single JSONL dataset (aiperf / HuggingFace WEKA format)
go run ./cmd/trace-export \
  --input-dir /var/log/llm-d/traces \
  --output-jsonl /tmp/captured-traces.jsonl \
  --hash-id-scope local

# Per-trace JSON directory (inference-perf trace_directory format) with 16 -> 64 block downsampling
go run ./cmd/trace-export \
  --input-dir /var/log/llm-d/traces \
  --output-dir /tmp/captured-weka-dir \
  --target-block-size 64 \
  --drop-classifier-calls
```

When requests carry no session header, `trace-export` chains requests into sessions via longest-common-prefix block overlap (`--lcp-overlap-threshold`, default `0.90`).

## End-to-End Validation with `aiperf` (AgentX / WEKA Trace)

1. Deploy a small model (for example `Qwen/Qwen2.5-0.5B-Instruct`) behind `llm-d` with `token-producer` (`backend: vllm`), `trace-capture-producer` (`blockSizeTokens: 64`, `outputDir` mounted to a volume), and `--allow-experimental-plugins` on the EPP binary.
2. Replay an AgentX WEKA trace through the `llm-d` gateway using `aiperf`, passing the served model's tokenizer so client prompt synthesis and `token-producer` `/render` tokenization align on block boundaries:
   ```bash
   aiperf profile \
     --model Qwen/Qwen2.5-0.5B-Instruct \
     --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
     --url http://<llm-d-gateway-host>:<port> \
     --endpoint-type chat \
     --input-file <original-weka-traces.jsonl> \
     --custom-dataset-type mooncake_trace \
     --streaming
   ```
3. Export the captured JSONL files to WEKA format:
   ```bash
   go run ./cmd/trace-export \
     --input-dir /var/log/llm-d/traces \
     --output-jsonl /tmp/captured-weka-traces.jsonl \
     --hash-id-scope local
   ```
4. Compare `/tmp/captured-weka-traces.jsonl` against `<original-weka-traces.jsonl>`:
   - Session count and per-session turn count (including `type: "subagent"` entries when `x-dynamo-parent-session-id` or `x-claude-code-agent-id` headers are sent).
   - Per-turn `len(hash_ids)` (`in // 64`) and pairwise longest-common-prefix `LCP(turn[i].hash_ids, turn[j].hash_ids)` across turns within each trace.
   - `in` and `out` token counts (accounting for chat-template framing overhead if the client replay harness wraps raw token blocks in chat messages) and relative arrival timestamps `t`.
