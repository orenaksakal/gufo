# Qwen3.8 Flash-Next quality

**All 63 measured tg128 requests match fresh AR completions:** 21 AR,
21 mixed MTP and 21 repetitive MTP at C1/2/4/6/8. Unsloth UD-Q4_K_XL target,
shared Q8_0 MTP; [identities](artifacts/model-identities.json).
These are consistency checks, not original unquantized-model or GGUF-conversion
qualification. HTTP measurements: September 20–23, 2026; attention review:
September 27–28.

| Check | Result |
| --- | --- |
| MTP versus scalar CPU formulas, eight text/image states | Fusion/attention relative RMS <0.0008 (limit 0.002); full-width normalization, split projections, recursive carry and full Q8 head checked |
| Batched MTP/AR, C2/C4/C6/C8 | Logits, tokens, acceptance, RNG and every 1–8-token rollback prefix match isolated execution |
| Sampling | 23 AR/MTP configurations; shared FP64 target filtering/CDF, p/q acceptance and residual correction pass |
| Prefill and cache | Full logits match across tested chunk boundaries, short tails and restored state through 4096 tokens |
| Scalar versus bulk prefill, 2176 tokens | Same top-1; logit RMSE 0.18, not bit-identical |
| Serving | Cancellation, three-turn continuation, reasoning/tool history, concurrent image/text isolation and disk restart pass |
| Sparse attention | Independent FP64 operator error ≤2.83e-7 (limit 1e-6). At 32K/128K, 256 fixed-token code/prose rows: mean KL 5.82e-4 and 256/256 top-1 agreement with an FP64-attention diagnostic. [Evidence](artifacts/attention-tiles-review.json). |

The attention diagnostic retains the quantized weights and other native
operators. Regrouping FP32 sums can change long-context text across builds;
matched-prefill AR/MTP logits and sampled snapshot replay are exact at
32K/128K. Session tests also cover image/text restoration and C2/4/6/8.

Sampled MTP can consume different RNG draws from AR. Seeded replay requires
the same build, request budget, capacity and sampling configuration; live cost
timings only steer greedy decoding. Draft sampling uses the full Q8 head's
top 64 logits; upstream draft-sampler equivalence is not claimed.

## Vision

**One Gufo encoder comparison fails:** relative L2 **6.47%** versus official
Transformers BF16, above the **5%** limit, on a 1024×1024 synthetic texture.
Both use the same converted GGUF weights. Against FP32, Gufo and Transformers
BF16 differ by **7.83% / 8.10%**, respectively. This is numerical drift, not an
image-answer score; **llama.cpp was not tested**. Optimizations retain native
embedding bytes but do not resolve this gap. [Evidence](artifacts/vision-parity.json).

## Reproduce

Tests live in [`tests/models/qwen38_flash_next`](../../../tests/models/qwen38_flash_next).
Use `--batch-only`, `--sampling-only` or `--prefill-only` on the session test;
the snapshot test covers persistent image/text state. For independent MTP checks:

```sh
nix develop -c cmake --build --preset gpu-test \
  --target qwen38_flash_next_model_tests qwen38_flash_next_gpu_probe
nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model "$MODEL" --mtp-model "$MTP" --mtp-audit
```

The [vLLM](https://github.com/vllm-project/vllm/blob/751f6807d9cb3de50c27a5f27188c4fb04fe0e2b/vllm/models/qwen4_exp/amd/mtp.py)
and [SGLang](https://github.com/sgl-project/sglang/blob/993d1fccbaafe3e79d91567d2fc1d665cc94fa50/python/sglang/srt/models/qwen4_exp_mtp.py)
formulas supply independent predictor checks; pinned Transformers ignores MTP
weights. [Vision reproduction](../qwen3.8-27b/QUALITY.md#vision).

## Framework prompt-lookup fork

The Framework qualification uses the same target/MTP hashes listed above, on
the local GCC 16.2.1 / ROCm 10 runtime. The new lookup index supplies greedy
proposals to the existing target verifier; target arithmetic, weights, attention
selection and KV precision are preserved. Sampled and image requests bypass
lookup. These are regression checks against this quantized model.

| Check | Result |
| --- | --- |
| CPU lookup bounds/state | Append, budget limits, reset/rebuild and forced table collisions pass; standalone ASan/UBSan pass |
| Greedy lookup versus scalar AR | At 224 and 32,768 prompt tokens: output tokens, complete vocabulary logits and RNG exact at every committed frontier |
| Lookup coverage | 41/45 copied proposals accepted in the short fixture; 48/48 at 32K, across eight cycles each |
| Restored lookup | Rebuilt index preserves greedy tokens/full logits and seeded sampled replay; sampled lookup-cycle count remains zero |
| Shared batch behavior | C2/C4/C6/C8 AR/MTP logits, tokens, RNG and lookup statistics exact; 21 accepted copied tokens in mixed sampled/greedy cohorts; independent state and cancellation isolation pass |
| Sampling | Existing 23-configuration serving suite passes with AR/MTP and C2/budget replay |
| Persistent state | Snapshot round trips exact at 4,095 tokens |
| Paired production HTTP | 15/15 tg128 byte hashes and token counts match upstream at capacity 196,608; prose, copy/edit, repetition and sampled control |
| Long tool history | 133,131-token cold input and 134,063-token continued input; baseline message/count agreement and greedy/sampled cached replay exact; only 932 appended tokens prefilled, zero on repeat |
| OpenCode HTTP contract | Streamed reasoning and tool-call arguments, tool-result replay and strict JSON output match baseline; two cancellation/three-turn cases retain their prefixes and seeded output |
| OpenCode V2 client | v2.0.20 standalone run completes a real `read` tool round trip and returns the exact marker from a synthetic local file |

Long continuations reach EOS at 18 greedy / 13 sampled tokens; the cold request
fills tg128. These are bounded replay/serving checks, not complete-file task
scores. [Retained evidence](artifacts/framework-review.json).

No vision sidecar is installed on this machine, so local image qualification is
unavailable. The upstream vision gap described above still applies.
Reproduction and HTTP/OpenCode results: [Framework guide](FRAMEWORK.md).

## Framework snapshot population

Parallel page population is qualified against the prompt-lookup fork `49306f3`
on the same local production toolchain and weights. The independent transfer
oracle checks every byte of 1/4 GiB destinations; model checks cover serialized
state and seeded replay.

| Check | Result |
| --- | --- |
| Standalone transfer | All 48 shape/advice/worker/repetition cases copy the expected bytes exactly; allocation preparation, transfer and release timed separately |
| Complete 32K snapshot | At 32,767 tokens, all 1,019,197,388 serialized bytes have the same SHA256 as the baseline |
| Snapshot round trips | Full logits, fresh/used sessions, persistent byte form, sampled continuation, deferred residual/RNG, extension and malformed-payload rejection pass |
| Concurrent capture | A frozen 6,143-token peer takes the parallel-population path during graph capture; snapshot bytes and captured/replayed logits exact |
| Batch state | C2/C4/C6/C8 full logits, tokens and RNG exact; cancellation, rollback, independent state and sampled residual checks pass |
| Long production turns | 18/18 measured completion hashes, prompt/output counts and cache counts exact at 32K/133K prefixes, including the 8 GiB resident-memory fixture |
| Short controls | Six tg128 prose/sampled-code completion hashes and counts exact |
| OpenCode HTTP contract | All eight long-history, greedy/sampled replay, streamed reasoning/tool-call, tool-result and constrained-JSON cases match the prior fork |
| Cancellation and continuation | Both greedy/content and sampled/reasoning three-turn cases pass; resume reuses 602/648 tokens and prefills only 13/11 |

These remain bounded quantized-model regression checks. Long branch outputs
can reach EOS before the 128-token budget. The existing vision qualification
limits above apply. [Bounded evidence](artifacts/framework-snapshot-review.json)
and [reproduction](FRAMEWORK.md#long-turn-checkpoint-measurements).

## Framework native-lane reductions

The gfx1151 vector kernels retain their descending 32-lane addition tree,
including independent halves of the wave64 expert kernel. Qualification uses
the same local toolchain and quantized weights as the snapshot iteration.

| Check | Result |
| --- | --- |
| Standalone Q8 projections | All 32 shape/token cases match the original shuffle reduction byte-for-byte; every cold-weight ring slot checked |
| Dense operators | Scalar/batched/gated/ragged outputs and guards exact; graph replay and independent FP64 head dots pass, including small activations |
| Routed operators | Q4_K/Q5_K/Q5_1/Q8_0 grouping, separate/fused SwiGLU, duplicated/inactive experts, ragged rows, nonfinite scales and guards pass |
| Model state | C2/C4/C6/C8 full logits, tokens and RNG match isolated execution; cancellation, rollback, independent state, sampled residuals and peer snapshots during graph capture pass |
| 32K replay | Lookup versus AR full logits/tokens/RNG, restored state and sampled replay exact at 224/32,768 tokens |
| Sampling | All 23 configurations pass AR/MTP, C2 and budget replay |
| Sampled/thinking profile | Identical 128-token ID hash and 78,005 inference kernel launches |
| Short HTTP | 30/30 tg128 completion hashes and token counts match across both measurement orders |
| Long HTTP | 18/18 completion hashes and token/cache counts match; thinking-on reaches 133,295 cached tokens at 196,608 capacity |
| Prepared batch HTTP | 66/66 tg128 completion hashes and token/cache counts match at C1/2/4/6/8 after full decode warmup; physical widths verified |
| OpenCode HTTP contract | All eight cases match the snapshot fork, including streamed reasoning/tools, tool-result continuation, strict JSON and exact reuse of 134,063 prompt tokens |

The local vision-sidecar limitation above still applies.
[Bounded evidence](artifacts/framework-dpp-review.json) ·
[Reproduction](FRAMEWORK.md#vector-reduction-measurements).

## Framework deep-prefill follow-up

The proposed 2176-token chunk limit is **rejected**: the assertion-enabled
`qwen38_flash_next_session_test --prefill-only` reports changed full logits
for a 4096-token prompt split at 2048, before decoding step zero. The preceding
136-token boundary cases and 2048/1025 split pass. The first divergent operator
is not yet isolated. The candidate was reverted before HTTP timing or deeper
qualification; the deployed chunk limit remains 2048.

All six newly measured baseline 32K-turn messages and token/cache counts match
the prior retained build. The instrumented cold-prefix and branch responses
also match. These are regression observations, not a quality pass for the
rejected candidate. [Evidence](artifacts/framework-prefill-review.json).

## Benchmark method

Gufo single-user TG refreshed September 27, 2026 (`f797b5b`); PP and other
measurements retain September 22–23 provenance. One warmed sample per point,
greedy, thinking off.
Single-user uses pp2048/tg128; MTP pp is the maximum across mixed/repetitive
workloads. C1/2/4/6/8 use the same d0 prompts; every session prefills before
measured tg128, with at most four prompt-tail tokens reevaluated. Rates sum
individual decode rates. Gufo d0/C1 agree within 0.4% with matching drafts/output.
Depth calibration depends on the ordered sweep. Paired speed controls use
the same depth list. Deep AR/MTP HTTP cache frontiers differ by one token;
exact replay is checked separately with identical prefill boundaries.
AR reference is llama.cpp b11069; MTP uses pinned `6fcaa16f`.
Loading: cold files, C1/MTP/capacity 262144. Memory: C1/AR/capacity 133121,
peak global HIP allocation including idle memory. Full commands, counts and
identities remain in [artifacts](artifacts/bench.json) and the
[benchmark workflow](../../../.agents/skills/benchmark-model/SKILL.md).
