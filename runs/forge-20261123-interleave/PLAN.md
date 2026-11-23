# Forge DGX Spark controlled experiment — 2026-11-23

## Pre-registered hypotheses and accounting

- H1: With no competing decode and no pending spill work, `PREFILL_INTERLEAVE_MS=250` adds no mandatory 250ms wait. Changing it to 0 should not materially change single-request prefill. Falsify using matched cold natural prompts and isolated active-request counters.
- H2: Under actual competing decode, disabling interleave may shorten cold prompt time at the cost of longer decode stalls. Test separately; never attribute this to faster GPU compute.
- Targets: cold prefill 3000 tok/s; single-stream decode peak 200+ tok/s. Preserve full-run rate, TTFT and timing. No multi-stream aggregate or SSE burst counted as a single-stream target.
- Roofline: 3000 / measured baseline is a required speed ratio, not promised headroom. Need measured compute/communication/I/O timing; no speculative multiplication of gains.

## Controls and safety

Existing image `tensorfold-dsv41:exl3-tp2-8348338a8d90`; existing EAQ v1.1 + abliterated overlay; context 524288; pool 1052672; TP2; spill64GiB; KV sharding on; existing graphs/prefetch/side/AIO; clocks unchanged. No checkpoint modifications, no image rebuild/download, no lowered memory guards.

Human approved 20–30 minute maintenance window and up to approximately four restarts. Both ranks supervised by head's user systemd `dsv41-tensorfold.service`; direct start.sh is lock-blocked while supervisor runs. Restart via systemd, no killing foreign processes or removing lock files.

Before snapshots and remote scripts archived in this directory. Remote original settings backup: `/home/ztxjzh/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/scripts/local.sh.forge-before-20261123`, sha256 c8bd78f42a17611895952ca44be5d5a3054d7f74ccda47aac68f37e6038c1b60.

Only change: append recipe `PREFILL_INTERLEAVE_MS=0` AND explicit exported `TF_DS_PREFILL_INTERLEAVE_MS=0` (avoid absent-variable/default ambiguity). Revert both to 250 for A arms. Original image and complete config retained.

## Evidence and limitations

`probe.py` writes raw request, SSE timestamps, usage, actual server block/hash, before/after metrics. Credentials are loaded from existing local key path, never saved in evidence. Secret values are excluded from rank snapshot env. Every request <=50s socket timeout; harness calls <=55s. Cold requests require explicit `cached_tokens=0`, not assumed.

A0 warm and initial 5715-token pilot precede final corpus lengths (28000/55000 chars, actual tokenizer usage approximately 8K/16K); pilot excluded from matched performance comparisons. Prompts identical for matched arm/rep; cache/SPILL-restored repetitions flagged and excluded from cold timing. Greedy code/prose 512 tokens, thinking/tool smoke. Compare strong server token_sha and counts/text, not only first token. Finite smoke does not prove general distribution equality or baseline quantization quality.

No per-token peak declared from chunk counts. Full usage-based output-window rate and raw SSE gaps retained; SSE gap P99 is not individual-token P99. Raw hashes are supplied by server (12 hex chars), and text/output equality checked as additional evidence.

## Retractions of earlier assistant assertions

Cache token proportion != wall-clock prefill proportion. No evidence that GB10 shares Mac's 94GB knee, that allocation residency eliminates all VM cost, or that clock ceiling is broken merely because measured SM clock differs. Old 91/99s concurrent experiment does not predict this deployment's isolated gain. Upper bound 20–30% and unavailable 2x were unsupported. Multiple optimizations reducing time can have independent mechanisms; classify subsystem/overlap before composition.
