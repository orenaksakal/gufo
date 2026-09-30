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
