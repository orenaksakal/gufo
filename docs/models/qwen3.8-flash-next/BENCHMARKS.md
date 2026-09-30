# Qwen3.8 Flash-Next benchmarks

AMD Strix Halo `gfx1151`, 128 GB unified memory. Unsloth `UD-Q4_K_XL` target and
shared-Q8_0 MTP sidecar; Gufo uses adaptive MTP. HTTP, greedy, thinking off.
Gufo single-user tg: September 27, 2026 (`f797b5b`); pp and other results:
September 22–23. Concurrency uses the unchanged short-context attention path.
llama.cpp uses `b11069` for AR and `6fcaa16f` for MTP.

Positive gain favors Gufo.
[Quality and measurement details](QUALITY.md#benchmark-method) · [Model identities](artifacts/model-identities.json)

## Framework fork: coding fixtures

September 30, 2026 UTC, Framework Desktop Ryzen AI Max+ 395 / Radeon 8060S,
128 GB. This machine's GCC 16.2.1 / ROCm 10 stack on both arms; upstream
`f783fedb` versus the prompt-lookup fork. Production binaries, identical weights,
C1, **196,608-token context capacity**, MTP ceiling seven, thinking off, seed 73.
One warmup and three cache-bypassed tg128 requests per fixture; medians below.
Decode rates exclude prefill; HTTP wall time includes the complete request.

| Fixture | Prompt tokens | Upstream decode (tok/s) | Fork decode (tok/s) | Gain | HTTP wall, upstream → fork (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose control | 1,965 | 37.35 | 37.40 | +0.1% | 4,745 → 4,746 |
| Copy code, greedy | 1,068 | 73.67 | 83.30 | +13.1% | 2,659 → 2,452 |
| Edit code, greedy | 1,088 | 71.33 | 84.83 | +18.9% | 2,733 → 2,439 |
| Repetition, greedy | 1,911 | 83.24 | 105.73 | +27.0% | 2,830 → 2,493 |
| Sampled code control | 1,088 | 67.29 | 67.37 | +0.1% | 2,838 → 2,837 |

All **15/15** measured completion byte hashes and token counts match upstream.
Sampled control uses temperature 1, top-p 0.95 and top-k 20; lookup is greedy
text-only. Copy/edit complete-request times improve 7.8% / 10.7%. These synthetic
fixtures do not measure completed OpenCode tasks. A separate long tool-history
check reaches 134,063 prompt tokens with exact replay and prefix reuse; its
bounded EOS continuations are qualification cases, not tg128 speed samples.

[Reproduction and OpenCode profile](FRAMEWORK.md) ·
[Retained timings, hashes and qualification](artifacts/framework-review.json).
The upstream reference tables below retain their original dates and toolchain.

## Framework fork: long-turn checkpoints

September 30, 2026 UTC; prompt-lookup fork `49306f3` versus parallel snapshot
page population. Same production toolchain/weights, C1, context **196,608**,
MTP ceiling seven, thinking off, seed 73. Each distinct branch reuses the cached
prefix and prefills 2,187 new tokens. One warmup and three measured turns per
row; medians below. Positive reduction means a shorter complete HTTP request.

| Cached prefix | Mode | Extra resident RAM | Snapshot ms, before → after | HTTP wall ms, before → after | Wall reduction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 32,829 | Greedy | 0 | 67.4 → 41.1 | 3,682.9 → 3,657.9 | 0.7% |
| 32,829 | Sampled | 0 | 65.4 → 40.2 | 1,936.2 → 1,939.2 | −0.2% |
| 133,131 | Greedy | 0 | 201.5 → 114.8 | 2,444.1 → 2,355.7 | 3.6% |
| 133,131 | Sampled | 0 | 198.9 → 111.7 | 2,495.1 → 2,407.0 | 3.5% |
| 133,131 | Greedy | 8 GiB | 213.5 → 121.5 | 2,505.4 → 2,374.7 | 5.2% |
| 133,131 | Sampled | 8 GiB | 195.1 → 132.3 | 2,505.1 → 2,450.6 | 2.2% |

All **18/18** completion hashes, prompt/output counts and cached-token counts
match. Sampled requests use temperature 1, top-p 0.95, top-k 20. The 128-token
budget permits EOS: measured outputs are 90/126 greedy and 6/16 sampled at 32K,
and 18/25/54 greedy and 6/22/26 sampled at 133K. These are bounded turn-latency
measurements, not fixed-tg128 throughput or completed coding-task scores.

The 8 GiB fixture holds idle ordinary host pages to model a busy desktop;
host memory pressure and compaction are recorded. Small tg128 controls remain
stable: prose **37.40 → 37.31**, sampled code **67.37 → 67.15** decode tok/s,
with all six outputs exact. Snapshot population starts at 256 MiB, using four
workers including the caller. It retains huge-page advice and falls back to
normal demand paging on page-population failure.

[Reproduction](FRAMEWORK.md#long-turn-checkpoint-measurements) ·
[Timings, hashes and qualification](artifacts/framework-snapshot-review.json).

## Framework fork: exact vector reductions

September 30, 2026 UTC; snapshot fork `61a617b` versus gfx1151 native-lane
reductions in dense and routed vector projections. Same production toolchain,
weights, C1, **196,608-token capacity**, seven-draft cap and seed 73. One warmup
and three cache-bypassed tg128 samples per fixture. This table shows the
candidate-first/baseline-second confirmation; the opposite order also passed.

| Fixture | Decode tok/s, before → after | Decode gain | HTTP wall ms, before → after |
| --- | ---: | ---: | ---: |
| Prose | 37.38 → 37.50 | +0.32% | 4,738.3 → 4,733.1 |
| Copy code | 83.64 → 84.12 | +0.57% | 2,453.1 → 2,437.5 |
| Edit code | 85.03 → 85.84 | +0.95% | 2,441.0 → 2,422.9 |
| Repetition | 105.64 → 106.56 | +0.88% | 2,499.4 → 2,485.2 |
| Sampled code | 67.68 → 67.77 | +0.14% | 2,826.6 → 2,821.2 |

All **30/30** measured completion hashes and counts match across the two
orders. Thinking is off; sampled code uses temperature 1/top-p .95/top-k 20.
The sampled gain is small: +0.70% in the first order and +0.14% in the reverse
order. The optimization preserves the original addition tree and adds no
allocation or weight copy.

Thinking-enabled turns append **2,059 tokens** to the cached coding/tool
history at the same 196,608 capacity. One warmup and three distinct branches per
mode; medians include checkpoint capture and the complete HTTP request:

| Cached tokens | Mode | HTTP wall ms, before → after | Wall reduction |
| ---: | --- | ---: | ---: |
| 32,920 | Greedy | 3,688.4 → 3,662.3 | +0.71% |
| 32,920 | Sampled | 4,614.2 → 4,593.1 | +0.46% |
| 133,295 | Greedy | 3,744.0 → 3,714.8 | +0.78% |
| 133,295 | Sampled | 4,141.9 → 4,159.6 | −0.43% |

All **12/12** messages and token/cache counts match. Outputs fill tg128 except
one deep sampled branch ending at 67 tokens. The deep sampled median pair
spends 20.1 ms more in checkpoint capture and 1.0 ms less in decoding; it does
not demonstrate a long sampled-turn speedup. Thinking-off 32K controls are
effectively stable: **+0.37% / −0.11%** greedy/sampled whole-request reductions,
with all six bounded-EOS responses exact.

Prepared mixed prose/code-edit concurrency controls use a separate eight-session
server at capacity 4,096. Complete tg128 warmup cohorts visit every width before
three measured cohorts; every session is prefilled before decoding. Rates are
the median sum of individual request decode rates, rather than total tokens
divided by cohort wall time:

| Clients | Decode tok/s, before → after | Gain |
| ---: | ---: | ---: |
| 1 | 58.18 → 58.40 | +0.38% |
| 2 | 85.88 → 86.76 | +1.03% |
| 4 | 137.86 → 140.15 | +1.66% |
| 6 | 161.07 → 162.44 | +0.85% |
| 8 | 173.82 → 178.30 | +2.58% |

All **66/66** completion hashes and token/cache counts match, with the requested
physical batch widths observed. C1 summarizes both fixture types and is not
directly comparable to either individual fixture above. These remain synthetic
regression workloads; executable coding-task observations are reported below.

The separate sampled/thinking CLI profile retains the same 128-token hash and
78,005 inference launches. Decode-tail kernel time falls **2,477.5 → 2,465.0
ms**, and its wall span **2,821.5 → 2,811.3 ms**. Instrumented timings explain
the mechanism; the table above uses unprofiled HTTP requests.

[Reproduction](FRAMEWORK.md#vector-reduction-measurements) ·
[Bounded measurements and qualification](artifacts/framework-dpp-review.json).

### Exact prefill-tail merging

September 30, 2026 UTC; retained `3469f4e` runtime versus terminal-tail merging,
same production toolchain/weights, C1, **196,608-token capacity**, thinking on,
seed 73. Ordinary chunks remain 2048; a terminal suffix of at most 128 tokens
shares target projection work while attention and MTP retain their original
chunk shapes. Each branch appends 2059 tokens to the cached history. One warmup
and three measured branches per mode; baseline first, candidate second.

| Cached tokens | Mode | Prefill ms, before → after | Complete HTTP ms, before → after | Wall reduction |
| ---: | --- | ---: | ---: | ---: |
| 32,920 | Greedy | 1,501.9 → 1,438.1 | 3,717.6 → 3,649.0 | +1.85% |
| 32,920 | Sampled | 1,501.3 → 1,446.2 | 4,650.9 → 4,585.5 | +1.41% |
| 133,295 | Greedy | 1,645.9 → 1,549.9 | 3,848.6 → 3,727.2 | +3.16% |
| 133,295 | Sampled | 1,628.1 → 1,550.3 | 4,250.6 → 4,133.4 | +2.76% |

All **24/24** measured long-turn messages and token/cache counts match, as do
sampled draft/acceptance counts. Outputs fill tg128 except the same 67-token
deep sampled branch on both arms. All twelve matched branches are faster; a
slow first deep greedy baseline branch is excluded by the reported median.

Short controls are stable: prose **37.21 → 37.17**, sampled code **67.19 →
67.08** decode tok/s; complete request medians change by less than 1 ms. All
12 short responses and eight OpenCode contract responses match, for **44/44**
measured HTTP/contract responses. The isolated one-logit-row executor probe
uses approximately **107 MiB** more allocation; it is not total process memory.

The initial wider-chunk candidates failed exactness and were rejected. The
retained implementation preserves dense/sparse attention dispatch, padded KV
load bounds and predictor catch-up shapes. [Diagnosis and qualification](QUALITY.md#framework-deep-prefill-follow-up)
· [Measurements and raw hashes](artifacts/framework-boundary-review.json)
· [Reproduction](FRAMEWORK.md#tail-merging-qualification).

### Executable OpenCode edits

OpenCode v2.0.20, greedy/thinking off, seed 73; one fresh small repository per
task and arm. The external grader runs the original contract tests against the
edited source. Both runtimes pass **2/2 tasks** with unchanged tests, four model
steps and **four tool calls per task** (two reads, one write, one test command),
with no tool errors.

| Task | Baseline wall time | Tail-merging wall time | External tests |
| --- | ---: | ---: | --- |
| Configuration normalization | 17.29 s | 15.85 s | 3 methods, including signed inputs and invalid-value subcases |
| LRU recency/eviction | 13.99 s | 12.89 s | 4 methods |

Wall time includes client startup and agent tools, excluding external grading.
These single trials use fresh temporary paths and can generate different text;
they establish bounded task success rather than a matched inference speedup.
[Contracts and evidence](artifacts/framework-task-review.json) ·
[Reproduction](FRAMEWORK.md#executable-opencode-tasks).

## Single user, autoregressive

Approximately pp2048 / tg128; depth is the cached prefix in tokens.
Context capacities differ between engines; see the measurement details.

<!-- bench:single-ar -->
| Flash-Next Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1628.52 | 489.59 | +232.6% | 25.87 | 22.20 | +16.5% |
| 4,096 | 1523.39 | 455.24 | +234.6% | 25.84 | 21.21 | +21.8% |
| 8,192 | 1499.13 | 428.85 | +249.6% | 25.84 | 20.40 | +26.7% |
| 12,288 | 1477.98 | 400.21 | +269.3% | 25.80 | 19.64 | +31.4% |
| 16,384 | 1457.16 | 375.89 | +287.7% | 25.79 | 18.95 | +36.1% |
| 32,768 | 1421.93 | 301.31 | +371.9% | 25.69 | 16.54 | +55.3% |
| 65,536 | 1304.01 | 221.94 | +487.6% | 25.41 | 11.62 | +118.7% |
| 131,072 | 1292.02 | 144.78 | +792.4% | 22.89 | 7.98 | +186.8% |
<!-- /bench -->

![Single user, autoregressive](artifacts/charts/single-ar.svg)

## Single user, MTP

pp is the highest measured rate per engine and depth across mixed/repetitive
text, including Gufo predictor catch-up.

<!-- bench:single-mtp -->
| Flash-Next Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain pp | Gufo tg mixed (tok/s) | llama.cpp tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1602.82 | 468.76 | +241.9% | 32.20 | 31.73 | +1.5% | 59.30 | 48.12 | +23.2% |
| 4,096 | 1506.32 | 423.79 | +255.4% | 32.50 | 34.48 | -5.7% | 46.74 | 45.71 | +2.3% |
| 8,192 | 1492.95 | 394.98 | +278.0% | 36.08 | 32.20 | +12.0% | 50.26 | 44.31 | +13.4% |
| 12,288 | 1488.03 | 365.91 | +306.7% | 34.26 | 32.32 | +6.0% | 44.81 | 43.53 | +2.9% |
| 16,384 | 1471.81 | 341.98 | +330.4% | 35.41 | 32.07 | +10.4% | 53.53 | 43.66 | +22.6% |
| 32,768 | 1449.56 | 277.61 | +422.2% | 31.03 | 25.41 | +22.1% | 44.96 | 39.28 | +14.5% |
| 65,536 | 1316.70 | 208.04 | +532.9% | 32.63 | 19.65 | +66.1% | 46.12 | 27.75 | +66.2% |
| 131,072 | 1335.91 | 135.42 | +886.5% | 34.20 | 14.63 | +133.8% | 45.01 | 20.96 | +114.7% |
<!-- /bench -->

![Single user, MTP](artifacts/charts/single-mtp.svg)

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
All sessions prefilled before timed decoding; throughput sums individual rates.
llama.cpp re-evaluates its four-token checkpoint tail.

<!-- bench:multi-ar -->
| Flash-Next Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 25.85 | 22.34 | +15.7% |
| 2 | 45.70 | 37.08 | +23.2% |
| 4 | 76.29 | 54.82 | +39.2% |
| 6 | 95.66 | 65.34 | +46.4% |
| 8 | 108.67 | 68.79 | +58.0% |
<!-- /bench -->

![Multiple users, autoregressive](artifacts/charts/multi-ar.svg)

## Multiple users, MTP

Same pp2048 mixed/repetitive prompts as single-user d0, tg128, context 4096
per user. All sessions prefilled before timed decoding; rates sum individual
request decode rates. C1 cross-checks the single-user table.

<!-- bench:multi-mtp -->
| Flash-Next Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32.09 | 31.77 | +1.0% | 59.20 | 46.93 | +26.1% |
| 2 | 51.68 | 42.79 | +20.8% | 93.16 | 52.57 | +77.2% |
| 4 | 75.92 | 48.78 | +55.6% | 127.91 | 48.23 | +165.2% |
| 6 | 89.18 | 52.59 | +69.6% | 142.14 | 48.51 | +193.0% |
| 8 | 106.47 | 61.92 | +71.9% | 157.22 | 58.13 | +170.5% |
<!-- /bench -->

![Multiple users, MTP](artifacts/charts/multi-mtp.svg)

## Loading time

C1, context capacity 262144, MTP. Cold target/sidecar files to HTTP readiness.

<!-- bench:loading -->
| Flash-Next Q4<br>Target | Gufo ready (s) | llama.cpp ready (s) | Gain |
| --- | ---: | ---: | ---: |
| Q4 | 15.45 | 117.83 | +662.7% |
<!-- /bench -->

![Loading time](artifacts/charts/loading.svg)

## Memory occupation

C1, context capacity 133121, AR. Peak memory reported by HIP.

<!-- bench:memory -->
| Flash-Next Q4 AR<br>Workload | Gufo GiB | llama.cpp GiB | Gain |
| --- | ---: | ---: | ---: |
| pp2048 + tg128 | 85.55 | 84.84 | -0.8% |
| 16K prefix, pp4096 + tg128 | 86.27 | 85.31 | -1.1% |
<!-- /bench -->

![Memory occupation](artifacts/charts/memory.svg)
