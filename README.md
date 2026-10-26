# RTX Pro 6000 Blackwell — Nanbeige4.3-33B on vLLM

An evidence-first serving workspace for **Nanbeige4.3-33B** on one NVIDIA RTX Pro 6000 Blackwell DC-4-96Q (GPU 0 MIG slice). vLLM serves an FP8 checkpoint behind an OpenAI-compatible API; every measurement is written to `evidence/` and rendered by [`tests/summarize.py`](tests/summarize.py).

Adapted from [tedngai/dgx-spark-nanbeige-4.3-33B](https://github.com/tedngai/dgx-spark-nanbeige-4.3-33B) (the original GB10 screening workspace; its evidence is preserved under `evidence/dgx-spark-reference/`).

This is a screening workspace, not a production deployment. `production_qualified` is false everywhere.

## Host

| | |
|---|---|
| GPU | NVIDIA RTX Pro 6000 Blackwell DC-4-96Q, MIG `4g.96gb` slice of GPU 0 |
| MIG device | `MIG-b12c2fdf-03dc-5a04-8cbb-46624a8df192` (~94 GiB, set in `.env` as `GPU_DEVICE`) |
| Driver | 580.159.03 (CUDA 13.0) |
| Coexistence | GPU 1 runs the `d3-systemone` vLLM service on :8000; ComfyUI uses :8188; this server binds :8012 on 127.0.0.1 |

Unlike the GB10, this card reports VRAM through NVML and has ~8 TB/s GDDR7, so decode and prefill ceilings are much higher than the bandwidth-bound Spark numbers.

## Why FP8

Nanbeige4.3-33B is a **dense** 33B model. Decode is bound by memory bandwidth:

| Weights | Bytes read per token | Ideal decode, batch 1 |
| --- | --- | --- |
| BF16 | ~62 GiB | ~130 tok/s @ 8 TB/s |
| FP8 (this profile) | ~33 GiB | ~240 tok/s @ 8 TB/s |

FP8 keeps 8-bit weights *and* 8-bit activations with dynamic per-token scaling, which is close to BF16 quality while roughly doubling the decode ceiling. The frozen checkpoint is `primitive-ai/Nanbeige4.3-33B-FP8`, a compressed-tensors W8A8 conversion of `Nanbeige/Nanbeige4.3-33B`.

## Production recipe (frozen)

| Setting | Value |
| --- | --- |
| Model | `Nanbeige/Nanbeige4.3-33B` @ `c1093c5…` |
| Served weights | `primitive-ai/Nanbeige4.3-33B-FP8` @ `0140c15…` (33.1 GiB) |
| Architecture | `LlamaForCausalLM`, 52 layers, 56 heads, 8 KV heads |
| Engine | vLLM 0.32.0, `vllm/vllm-openai` pinned by digest |
| Parallelism | TP1, one MIG slice |
| Served context | 262,144 tokens (checkpoint supports 524,288) |
| Concurrency | 32 sequences, 8,192-token prefill chunks |
| KV cache | FP8, prefix caching and chunked prefill on |
| Memory plan | `--gpu-memory-utilization 0.92` |
| Parsers | `qwen3` reasoning, `qwen3_coder` tool calls, auto tool choice |
| API | `http://127.0.0.1:8012/v1`, `/health`, `/metrics`, `/tokenize` |
| Auth | **None** — API bound to 127.0.0.1; `secrets/api_key` retained only for harness compatibility (`REQUIRE_API_KEY=1` re-enables `--api-key`) |

`configs/production.json` is the frozen contract; `scripts/start.sh` assembles the flags and targets the MIG slice via `--gpus '"device=…"'`. `configs/workloads.json` is the screen matrix.

## Host prerequisites

- Docker with the NVIDIA runtime or a CDI spec (`docker info` shows both here).
- Docker access for the current user.
- ~120 GiB free disk for the weights and image (698 GiB free at setup).
- Python 3.12 for the harness (standard library only).

## Quick start

```
cp .env.example .env                 # or use the committed .env for this host
scripts/download-model.sh            # 33 GiB, resumes, verifies every SHA-256
scripts/start.sh                     # serve and wait for /health
scripts/status.sh

python3 tests/validate_setup.py --expect-running --hash-model
python3 -B -m unittest discover -s tests -v
```

Then, with an idle service:

```
export NANBEIGE_API_KEY="$(cat secrets/api_key)"
python3 -B tests/smoke_test.py --label smoke-interface --execute --confirm-idle
scripts/run-screens.sh --execute --confirm-idle     # restart per screen
python3 tests/summarize.py                          # Markdown tables
```

`scripts/stop.sh` stops and preserves the container; `scripts/stop.sh --remove` deletes it.

## What the harness measures

- **Decode screens** — 1,024-token prompts, 256 forced output tokens, C1/2/4/8. Aggregate tok/s is completion tokens divided by request elapsed time.

- **Prefill screens** — 16K and 32K prompts, TTFT and end-to-end rate.

- **Retrieval** — 32K / 100K / 200K documents with three planted `ACCESS CODE` values; a run fails if any code is missing from the answer.

- **Warm reuse** — identical prompts, requiring ≥90% `cached_tokens` on every request, which exercises vLLM's prefix cache.

- **Telemetry** — Prometheus counters/gauges sampled every 0.2 s and host `MemAvailable` every second. (On this host VRAM *is* available via NVML: `nvidia-smi` on the MIG slice.)

## Results

Measured 2026-10-26 on the GPU 0 MIG slice — **10/10 screens, 30/30 rounds passed** on the promoted profile (0.92 utilization / 32 seqs / 8K prefill chunks):

| Workload | Result |
| --- | --- |
| Decode, 1K in / 256 out, C1 | **38.82 tok/s**, 0.126 s TTFT (clock-sensitive; 35.9–39.1 across runs) |
| Decode, C2 / C4 / C8 | 76.80 / 147.49 / **271.30 tok/s** |
| Warm prefix reuse, C1 | 39.63 tok/s, TTFT 0.033 s, ≥90% reuse |
| Prefill 16K / 32K | TTFT 2.69 s / 6.78 s (~4,800–6,100 tok/s) |
| Retrieval 32K / 100K / 200K | 3/3 codes at every size; TTFT 6.7 s / 38.4 s / **129.0 s** |
| KV capacity | 516,224 tokens (51.2 GiB), 1.97× concurrency @ 262K ctx |

The profile A/B (v1 vs v2), including the clock-drift control, is in [`docs/RESULTS.md`](docs/RESULTS.md). `tests/summarize.py` renders every attempt in `evidence/`, including failures.

## Contents

```
configs/     frozen profile and screen matrix
docs/        method, runbook, results
scripts/     download, start, stop, status, screen driver
tests/       client, prompts, benchmark, smoke, validation, unit tests
evidence/    append-only JSON reports (dgx-spark-reference/ = original host)
models/      downloaded weights (gitignored)
secrets/     API key (gitignored)
state/       model verification record (gitignored)
```

## Qualification boundary

Three-round screens are screening evidence, not stable tails. No sustained load, overload, fairness, coding-quality or agentic benchmark is claimed. The checkpoint is a community FP8 conversion, not a publisher-signed artifact, and the base model card's benchmark table is not reproduced here.
