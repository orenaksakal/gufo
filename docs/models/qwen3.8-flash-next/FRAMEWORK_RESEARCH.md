# Framework Flash-Next fork: research and design

Target: this Framework Desktop, Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`),
128 GB shared memory, Ubuntu 26.04, kernel `7.0.0-34-generic`, NVMe model storage.
Base: Gufo `f783fedb9bea2ec7de941f6da4e02f4a4596b29e`.
The local runtime is ROCm 10 (HIP reports `7.15.26333`), GCC 16.2.1 and AMD
Clang 23. Comparisons must use that same stack on both sides.
Use compiler output and installed RPM versions for this identity: upstream
`gufo diagnose` currently reports hard-coded Nix toolchain versions even in a
non-Nix build.

## Reviewed implementations

| Project / inspected revision | Useful mechanism | Decision for this machine |
| --- | --- | --- |
| [Halogen Flash](https://github.com/peonist-ai/halogen-flash-server/tree/82c92af2289f6f1086ab8362b18669ccff36968b), 0.15.1 documentation | MTP-anchored prompt lookup; exact greedy verification; cache-aware end-to-end timing; protecting resident weights from lookup-table page-cache pressure | Implement independent prompt-lookup code on Gufo's existing verifier. Halogen reports 13–15% agent-turn gains on its own prompts and native checkpoint; these are not this fork's measurements. Its public repository does not publish engine source. |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/19e28a27702117d8f2eb16b825b9a308111f67d9/common/speculative.cpp) | Multiple draft sources, bounded n-gram maps, committed-history updates, explicit speculative rollback | Keep the model's target verification and rollback authoritative; index only committed tokens; discard/rebuild lookup metadata at reset/restore. |
| [vLLM](https://github.com/vllm-project/vllm/blob/8039d3cbfd55be2aa172d544042c53b4e966369d/vllm/v1/spec_decode/ngram_proposer.py) | Per-request suffix matching and proposal limits; no auxiliary model for lookup | Use a bounded incremental CPU index instead of scanning the full long context every cycle. Respect token/context/draft budgets. |
| [SGLang HiCache](https://docs.sglang.io/advanced_features/hicache_design.html), repository reviewed at `7b30ff26eea8b593efe00d429eb001ad446c3f2a` | Exact prefix reuse; asynchronous cache persistence; distinguish GPU/host/storage tiers | Gufo already has exact live-frontier reuse and asynchronous snapshots. Retain these. GPU and CPU share physical RAM here: adding another full host KV tier is not free capacity. |
| [hipEngine](https://github.com/shisa-ai/hipEngine/tree/9c31ba9fed0c3cc85dfe286a5e88a845c37ab629) | CPU-oracle qualification, measured automatic dispatch, prefix caching for agent transcripts | Require full-logit AR agreement and snapshot replay; measure matched HTTP requests with byte hashes, rather than adopting advertised peak rates. |

This fork's lookup implementation is original C++; no external engine kernels
or source fragments were incorporated. Existing Gufo third-party notices apply.

## Retained numerical contract

The target GGUF shards, Q8 MTP head, target arithmetic, attention selection, KV
precision, recurrent state and sampler mathematics are preserved. The new CPU
index recognizes a four-token suffix including the target anchor and first MTP
proposal. It appends up to six committed-history tokens to the verification
batch, bounded by the existing seven-draft ceiling and remaining context/output.
Every token still passes the target verifier. A 256 KiB direct-mapped index is
allocated lazily per participating session; collisions only discard candidates.

Lookup is greedy text-only. Sampled requests retain their original MTP
distributions, acceptance/residual correction and RNG consumption. Image
contexts retain the existing path. Copy acceptance trains neither the deeper
MTP acceptance estimates nor its runtime batch-cost table. The index can be
reconstructed from the snapshot's committed tokens, so it adds no persistent
snapshot payload or mutable cross-request cache.

The vector-reduction iteration substitutes gfx1151 lane-exchange instructions
for the existing 32-lane XOR shuffles. It keeps the descending 16/8/4/2/1
addition tree, including independent wave64 halves, and adds no weight copy or
scratch buffer. The largest Q8 vocabulary projection remains bandwidth-bound;
its roughly 675 MB weight pass takes about 3 ms in the cold-ring diagnostic.
The measured full-model gain is consequently much smaller than the best
small-projection microbenchmark gain.

## Ideas requiring separate qualification

- **Wider prefill chunks / GEMM plans:** Halogen's 8K arena cannot be copied as
  a number into Gufo's different scratch layout. Profile Gufo's 2048-token
  projection/attention shapes first, including its ragged tails.
- **Fewer bits, attention-budget reduction, composable context, reduced
  recurrent precision:** these change numerical/model behavior. They do not
  satisfy this fork's exactness criterion without new independent quality
  evidence; they are not used as speedups here.
- **MTP cost calibration:** the upstream curves were measured with a different
  toolchain. The existing `qwen38_flash_next_gpu_probe --cost-audit 0` is the
  appropriate measurement; a headline speed from another engine is not a cost
  curve. Sampled policy changes also require seeded-replay qualification.
- **N-gram embedding I/O:** this is the model's learned PLE table, distinct from
  prompt lookup. Gufo already overlaps direct I/O with layer 0 and uses a
  bounded quantized-row cache. An unbounded page-cache approach competes with
  approximately 85 GiB of resident model/session allocations on this host.

Results and reproduction commands belong in [FRAMEWORK.md](FRAMEWORK.md).
