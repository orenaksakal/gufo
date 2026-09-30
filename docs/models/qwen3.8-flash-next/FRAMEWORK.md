# Gufo Framework / Flash-Next fork

This branch specializes optimization work for the Framework Desktop with Ryzen
AI Max+ 395, 128 GB RAM and Qwen3.8-Flash-Next `UD-Q4_K_XL` + shared Q8 MTP.
[Research and design decisions](FRAMEWORK_RESEARCH.md).

The retained production comparison at 196,608-token capacity improves greedy
copy/edit decoding **13.1% / 18.9%**, and complete HTTP request time **7.8% /
10.7%**. Prose and sampled controls are stable. All 15 measured completion
hashes/counts match upstream. [Benchmark table](BENCHMARKS.md#framework-fork-coding-fixtures)
and [bounded evidence](artifacts/framework-review.json).

The next iteration reduces complete **133K-prefix turns by 3.5–3.6%** relative
to that prompt-lookup fork, including sampled turns. Parallel preparation of
large snapshot buffers cuts checkpoint capture about **43–44%**; 32K controls
are stable. [Long-turn results](BENCHMARKS.md#framework-fork-long-turn-checkpoints).

Native-lane vector reductions add a modest **0.6–1.0%** greedy copy/edit decode
gain over the snapshot fork. Sampled gains are smaller; the 133K sampled turn
is effectively stable. All compared outputs remain exact.
[Reduction measurements](BENCHMARKS.md#framework-fork-exact-vector-reductions).

## Build on this machine

```sh
bash tools/qwen-flash/framework-build.sh
docker compose -f deploy/framework/compose.yaml config --quiet
```

The build uses a containerized ROCm 10 toolchain. The base OS image is pinned;
RPM repositories can advance, so record compiler/package versions for a new
qualification. `GUFO_TOOLCHAIN_IMAGE` can select an existing frozen local
toolchain image. No host ROCm installation is required.

The Compose service is deliberately machine-specific: existing local runtime
image, UID 1000, GPU group 990, this machine's model directory and loopback port
18080. It uses one session and 196,608 total context tokens. Only one model
server should hold the GPU at a time. To switch from the original service:

```sh
docker stop --timeout 120 gufo-flash-next
docker compose -f deploy/framework/compose.yaml up -d
curl --fail http://127.0.0.1:18080/ready
```

The service defaults to the model's thinking sampling preset: temperature 1,
top-p 0.95, top-k 20. Requests can override these. Prompt lookup accelerates
greedy text requests (`temperature: 0`); it does not change sampled inference
or reduce the model's reasoning effort to obtain speed. The 32,768-token output
default is a client-overridable limit covering both reasoning and answer.

Rollback:

```sh
docker compose -f deploy/framework/compose.yaml down
docker start gufo-flash-next
```

Also restore the previous Gufo model limits in OpenCode when switching back to
the original, smaller-capacity service.

## OpenCode and long conversations

Merge [`deploy/framework/opencode.json`](../../../deploy/framework/opencode.json)
into your OpenCode V2 project or global configuration. The profile exposes
196,608 total tokens, with 163,840 input and 32,768 output tokens. A 16,384-token
compaction buffer leaves approximately 147,456 input tokens before automatic
compaction, including system instructions and tool schemas. This provides room
for 126K+ coding transcripts without sacrificing the output reserve. The old
133,760 total limit could hold only 100,992 input tokens with that reserve.

The profile enables tool calls, streamed `reasoning_content` replay and exact
prefix reuse. Preserving prior thinking avoids rewriting the tool-cycle prefix
on each request. Keep the system prompt/tool definitions stable where possible;
Gufo reuses the unchanged prefix and prefills only the appended suffix. The
server keeps one GPU execution session for this single-user machine and queues
up to eight requests per client.

The default retains thinking and the model's sampling preset. Select
`gufo/flash-next#greedy` for deterministic requests eligible for
prompt-lookup acceleration; thinking remains enabled. Greedy decoding is a
separate sampling choice, not a demonstrated improvement in coding quality.
An OpenCode session already using another model needs an explicit selection
through `/models`. See the V2 [model](https://opencode.ai/v2/docs/models) and
[compaction](https://opencode.ai/v2/docs/compaction) references.

## Validation and measurements

Use the same compiler/runtime, weights, context capacity, draft ceiling and
concurrency for the baseline and candidate. Keep baseline binaries separately.

```sh
# Against the original server, then the fork server:
python3 tools/qwen-flash/framework-bench.py --label upstream \
  --output artifacts/framework/baseline.json
python3 tools/qwen-flash/framework-bench.py --label fork \
  --compare artifacts/framework/baseline.json \
  --output artifacts/framework/candidate.json

# Use --context 196608 for both servers. Long coding/tool-history contract:
python3 tools/qwen-flash/framework-opencode-check.py \
  --output artifacts/framework/opencode-baseline.json
python3 tools/qwen-flash/framework-opencode-check.py \
  --compare artifacts/framework/opencode-baseline.json \
  --output artifacts/framework/opencode-candidate.json

# Cancellation and three-turn replay with reasoning/tool history:
python3 tools/serving/check-continuation.py --url http://127.0.0.1:18080 \
  --model Qwen3.8-Flash-Next-UD-Q4_K_XL --tools --discard-assistant \
  --case content-preserve1-sampled0 \
  --case reasoning_content-preserve1-sampled1 \
  --output artifacts/framework/continuation.json
```

This driver uses Gufo's existing HTTP measurement implementation. It warms
each fixture once, measures three cache-bypassed 128-token completions, retains
actual prompt/output counts and completion hashes, and fails on replay drift.
Code-copy/edit fixtures are synthetic and do not represent a completed coding
agent benchmark. Prose and sampled code are controls. Raw results live under
the ignored `artifacts/framework/` directory.

The OpenCode contract driver uses synthetic archived Python source plus two
completed file-read tool calls. It requires at least 129,024 actual prompt
tokens and a cold tg128 response. It checks bounded greedy and seeded sampled
continuation/replay (allowing EOS) with at most 2,048 newly prefilled tokens,
then checks streamed reasoning/tool-call JSON, a tool
result and constrained JSON output. Baseline/candidate comparisons retain
message hashes, token counts, cache statistics and request latency.

The paired long-context run prefills **133,131 tokens**, then extends the
conversation to **134,063 tokens**, reusing 133,131 and prefilling only 932.
Repeated greedy and sampled requests reuse all 134,063 input tokens. The cold
response fills tg128; continuations reach EOS at 18 greedy / 13 sampled tokens.
All eight long-context/HTTP contract cases match upstream messages and token
counts. This checks state/replay and protocol behavior, not complete-file or
agent task correctness. Cancellation checks retain 602/648 prefix tokens and
prefill only 13/11 tokens when resuming discarded greedy/reasoning assistants;
three-turn replay remains exact.

An actual OpenCode **v2.0.20** standalone run using this profile also completes
a `read` tool call and returns the exact marker from a synthetic local file.
This is a client integration smoke check, not a coding task score.

The focused model test is `qwen38_flash_next_session_test --lookup-only`, with
`--model FIRST.gguf --mtp-model MTP.gguf` before that switch. It exercises
accepted/rejected copies against scalar AR, complete vocabulary logits, RNG,
snapshot rebuild and sampled replay. Existing `--batch-only`, `--sampling-only`
and `qwen38_flash_next_snapshot_test` cover shared batch state, cancellation,
filters/penalties and persistent state. Missing image weights are reported as
an untested image path, never a quality pass.

Exact replay is a regression criterion against the same quantized Gufo model.
It does not establish equivalence to the unquantized original model; upstream's
[quality report](QUALITY.md) records those separate qualification limits.

## Optimization roadmap

1. **Checkpoint latency: completed.** Parallel huge-page population improves
   133K whole turns 3.5–3.6%, with exact snapshot/state checks and busy-desktop
   controls. Continue including allocation/capture/release in turn measurements.
2. **Vector reductions: qualified.** Native-lane exchange preserves the original
   addition tree and lowers decode cost modestly. The largest Q8 head remains
   bandwidth-bound; further gains require better projection reuse. Preserve
   FP64 probabilities and seeded replay when investigating sampled execution.
3. **Deep prefill: next.** Refresh the dense/routed projection profile on this
   toolchain at 32K. Prioritize measured weight/activation reuse and tile costs,
   then expand a winning pp2048/tg128 comparison to 133K+. The existing chunk
   sweep favored 2048; wider chunks need fresh evidence and memory measurements.
4. **Coding-task evidence.** Add bounded, executable repository-edit tasks with
   test-based success criteria, tool-call counts and total completion time.
   Keep synthetic replay fixtures for precise regression diagnosis.

Every retained change uses matched production builds on this toolchain and
the 196,608-token deployment capacity. The OpenCode input/output reserves and
126K+ history requirement are qualification gates for each iteration.

### Long-turn checkpoint measurements

The turn driver appends 2,187 new tokens to a cached coding/tool history. Each
branch has a distinct request so it captures a fresh checkpoint. It measures
one warmup and three greedy/sampled turns, compares completion hashes and token
counts, and records complete HTTP wall time separately from snapshot time.
The 128-token output budget permits natural EOS: this measures bounded turn
latency, rather than fixed-length decode throughput or coding-task success.

```sh
# Run each depth against the saved production baseline, then the candidate.
python3 tools/qwen-flash/framework-turn-bench.py --depth 32768 \
  --output artifacts/framework/turn-baseline-32k.json
python3 tools/qwen-flash/framework-turn-bench.py --depth 32768 \
  --compare artifacts/framework/turn-baseline-32k.json \
  --output artifacts/framework/turn-candidate-32k.json
# Repeat both arms with --depth 133120 and distinct output paths.
```

For a controlled busy-desktop case, add `--resident-mib 8192` to **both** arms.
This holds 8 GiB of ordinary host pages idle after cold prefill, leaving at
least 12 GiB of available-memory headroom at allocation. Use a local, exclusive
server; the available-memory and compaction counters describe the entire host.

The allocation/transfer microbenchmark also measures release time and checks
every destination byte. Its standalone numbers are diagnostic, not HTTP speed:

```sh
tools/bench/build.sh tools/qwen-flash/snapshot_transfer_bench.hip
/tmp/snapshot_transfer_bench 1024
/tmp/snapshot_transfer_bench 4096
```

The existing `qwen38_flash_next_snapshot_test` accepts `--depth 32767` after
the model/MTP arguments and prints a hash of the complete serialized snapshot.
It checks full-logit restoration, persistent bytes, sampled continuation,
deferred residual/RNG replay and rejection of malformed snapshots. The session
test also captures a large frozen peer during decode graph capture.

### Vector-reduction measurements

The reduction comparison uses saved production `61a617b` and candidate
binaries with the same toolchain. Short fixtures use the existing
`framework-bench.py` driver in both measurement orders. Thinking-enabled
long turns preserve the full assistant message, including reasoning, in their
replay hashes:

```sh
python3 tools/qwen-flash/framework-turn-bench.py --depth 32768 --think on \
  --output artifacts/framework/thinking-baseline-32k.json
python3 tools/qwen-flash/framework-turn-bench.py --depth 32768 --think on \
  --compare artifacts/framework/thinking-baseline-32k.json \
  --output artifacts/framework/thinking-candidate-32k.json
# Repeat both arms with --depth 133120 and distinct output paths.
```

These C1 runs retain 196,608-token capacity. For a separate batched regression,
start each binary with `--sessions 8 --context 4096 --think off` and
`--max-pending-per-client 8`, retaining the same model/MTP and seven-draft cap:

```sh
python3 tools/qwen-flash/framework-batch-bench.py --source-revision 61a617b \
  --fingerprint artifacts/framework/fingerprint.json \
  --output artifacts/framework/batch-baseline.json
python3 tools/qwen-flash/framework-batch-bench.py --source-revision "$(git rev-parse HEAD)" \
  --fingerprint artifacts/framework/fingerprint.json \
  --source-dirty --compare artifacts/framework/batch-baseline.json \
  --output artifacts/framework/batch-candidate.json
```

Supply the same machine fingerprint on both arms, with compiler/runtime
versions verified against the installed toolchain as described in the research
notes. The batch driver warms complete tg128 cohorts at every width, then
prepares every session before each of three measured decode cohorts. It mixes
prose and code-edit fixtures and checks C1/2/4/6/8 completion/cache counts.
Restore the normal C1/196,608 deployment after this short-context control.
The focused operator binaries accept `--decode-only`:
`qwen38_flash_next_projection_ops_test` and
`qwen38_flash_next_routed_wmma_ops_test`. The standalone reduction diagnostic
checks original-versus-candidate bytes while rotating weights beyond MALL:

```sh
tools/bench/build.sh tools/qwen-flash/q8_reduce_bench.hip
/tmp/q8_reduce_bench
```
