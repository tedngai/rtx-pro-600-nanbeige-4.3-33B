# Results — Nanbeige4.3-33B FP8 on RTX Pro 6000 Blackwell (GPU 0 MIG slice)

Measured 2026-10-26 on the `MIG-b12c2fdf` 4g.96gb slice of GPU 0; **10/10 screens and 30/30 rounds passed**, plus 1/1 smoke interface run. All reports live in `evidence/` (append-only); tables below are medians rendered from them. An earlier interrupted run left one extra `short-c1-cold` report (36.32 tok/s) — kept per the append-only rule; the completed-run value is used here.

## Decode (1,024 in / 256 out, cold)

| Concurrency | Aggregate tok/s | Per-request tok/s | Latency p50 | TTFT p50 | vs GB10 C1 |
| --- | --- | --- | --- | --- | --- |
| C1 | **39.07** | 39.07 | 6.55 s | 0.122 s | 5.9× |
| C2 | 76.85 (1.97×) | 38.46 | 6.66 s | 0.196 s | |
| C4 | 147.50 (3.77×) | 36.89 | 6.94 s | 0.483 s | |
| C8 | 272.87 (6.98×) | 34.12 | 7.50 s | 0.974 s | 41.4× (GB10 C8: 42.11) |

Batch-1 decode reads 33.09 GiB of FP8 weights per token, i.e. ~1.29 TiB/s of effective weight streaming — about 39 tok/s. As on the GB10, batching amortizes the weight read (7.0× aggregate at C8) but does not improve per-stream latency; per-request tok/s stays ~34–39 at every concurrency.

## Warm prefix reuse (C1, identical prompts)

| Metric | Value |
| --- | --- |
| Aggregate tok/s | 39.76 |
| TTFT p50 | **0.033 s** (cold: 0.122 s) |
| Pass criterion | ≥90% `cached_tokens` on every request — passed |

## Prefill (cold, C1)

| Prompt | TTFT p50 | Prefill rate (prompt/TTFT) | GB10 TTFT |
| --- | --- | --- | --- |
| 16,384 | 2.683 s | ~6,100 tok/s | 22.8 s |
| 32,768 | 6.803 s | ~4,800 tok/s | 50.6 s |

Engine-side chunked-prefill windows logged ~20,000 prompt tok/s during long prefill; the end-to-end rates above include tokenization and scheduling of the full request.

## Retrieval (planted ACCESS CODEs, all found — pass at every size)

| Document size | TTFT p50 | Latency p50 | GB10 TTFT |
| --- | --- | --- | --- |
| 32K | 6.80 s | 7.58 s | 50.5 s |
| 100K | 37.99 s | 38.91 s | 214 s |
| 200K | 128.76 s | 129.69 s | 625 s |

The 200K prompt is the practical limit: ~2.2 minutes to first token (5.3× faster on this GPU than the 10.4 minutes on GB10).

## Capacity and engine

| | |
| --- | --- |
| Model load | 33.44 GiB in 8.6 s; engine init 8.9 s |
| KV cache (FP8) | 44.57 GiB → **449,376 tokens** |
| Max concurrency @ 262,144 ctx | 1.71× per request |
| Served context | 262,144 (checkpoint supports 524,288) |
| Container VRAM | ~83.2 GiB of the ~94 GiB MIG slice at `--gpu-memory-utilization 0.85` |

## Interpretation

- Batch-1 decode is bandwidth-bound at ~39 tok/s. If the slice streamed weights at the full ~8 TB/s GDDR7 rate, the ceiling would be ~240 tok/s; the measured 1.29 TiB/s suggests the MIG slice does not receive the full memory bandwidth and/or FP8 GEMM kernels are not saturating it. This is a measured observation, not a diagnosed cause.
- Aggregate throughput scales near-linearly to C8 (7.0×), so this slice is a good batch server but a modest single-stream server.
- Long-context prefill is the practical constraint for interactive RAG-style use; 32K is fine (6.8 s), 200K is batch-oriented.

## Profile A/B: v1 vs v2 (2026-10-26)

Two profiles were screened back-to-back with the identical matrix (cold restart per screen). v2 raises `gpu-memory-utilization` 0.85 → 0.92, `max-num-seqs` 16 → 32 and `max-num-batched-tokens` 8192 → 16384.

| Screen | v1 (06:29 run) | v2 (12:12 run) | v1 control (12:35–12:45) |
| --- | --- | --- | --- |
| Decode C1 tok/s | 39.07 | 35.96 | **35.88** |
| Decode C2 / C4 / C8 | 76.85 / 147.50 / 272.87 | 70.68 / 135.56 / 253.52 | — |
| Warm C1 tok/s (TTFT) | 39.76 (0.033 s) | 36.40 (0.036 s) | — |
| Prefill 16K TTFT | 2.683 s | 2.885 s | **2.798 s** |
| Prefill 32K TTFT | 6.803 s | 7.278 s | **7.222 s** |
| Retrieval 32K / 100K / 200K TTFT | 6.80 / 37.99 / 128.76 s | 7.22 / 40.79 / 134.73 s | — |
| Screens passed | 10/10 | 10/10 | 2/2 |
| KV cache | 44.57 GiB, 449,376 tok | **50.94 GiB, 513,616 tok** | — |
| Full-context concurrency @ 262K | 1.71× | **1.96×** | — |
| Container VRAM | 83.2 GiB | 87.4 GiB | — |

**The 7–8% apparent slowdown is clock drift, not the profile.** A same-time v1 control (labels `-20261026c` / `-20261026d`) reproduces v2's numbers to within 0.2% on decode and 1–3% on prefill. The morning v1 run was measured on cooler clocks; absolute throughput across the day ranges ~35.9–39.1 tok/s. Any future A/B must interleave profiles or run a control.

**What v2 actually buys:** +14.3% KV capacity and 1.96× full-context concurrency, at the cost of ~4.4 GiB of headroom (87.4 of ~91.8 GiB). Decode and prefill throughput are unchanged; the larger prefill chunk showed no measurable benefit, so it could be reverted in a later screen without loss.

**Decision:** promoted. `configs/production.json` now carries the v2 serving flags (10/10 screens, no regression against the same-time control); the v1 contract is archived as `configs/production-v1.json`, and v1 evidence remains under labels `-20261026`.

## Scope and caveats

- Three-round screens are screening evidence, not stable tails; p95 is omitted below 30 samples by the harness.
- The slice is shared with desktop processes (Xorg et al., ~0.5 GiB) on GPU 0; no inference load ran concurrently, but desktop activity is not controlled for.
- The GPU 1 d3-systemone service ran its own load during the screens; the MIG slices are memory-isolated but share the host CPU and PCIe.
- The checkpoint is the community `primitive-ai/Nanbeige4.3-33B-FP8` conversion, not a publisher-signed artifact. `production_qualified` remains false.
- GB10 comparison numbers are from `evidence/dgx-spark-reference/` on a DGX Spark (different hardware class); they indicate order of magnitude, not a controlled A/B.
