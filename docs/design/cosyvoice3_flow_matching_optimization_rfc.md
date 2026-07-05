# RFC: CosyVoice3 Flow-Matching Optimization Plan

## Summary

This RFC proposes a staged optimization plan for the CosyVoice3 (CY3) Stage1 Code2Wav flow-matching path. The primary goal is to improve throughput under concurrent streaming TTS workloads while keeping the implementation upstream-friendly: feature gated, measurable, easy to roll back, and validated by both performance and quality gates.

The scope is focused on the flow-matching portion of CY3, especially the 10-step CFM Euler loop and DiT estimator. The proposal covers:

- FP16 flow-matching execution
- Attention batching with varlen layout
- "KV Cache" as an Attention metadata/workspace cache for the MVP
- CUDA Graph capture of the full CFM Euler loop
- Prompt prefix cache for repeated prompt/reference inputs

## Background

Current profiling shows Stage1 flow matching is a major contributor to end-to-end latency and RTF. Existing work has made the flow path batch-capable, but real SeedEN traffic can still form singleton shape groups because codec token lengths and prompt/reference lengths differ across requests.

Recent attention-only varlen experiments also showed that simply enabling varlen FlashAttention is not sufficient:

- Flow batching reduced CFM Euler range count in profile.
- The bf16 varlen Attention path regressed end-to-end performance.
- Fine-grained Attention profiling showed the actual varlen Attention kernel was small, while `_upad_input`, `_pad_input`, and helper overhead dominated the new cost.

Therefore, this RFC treats batching, varlen metadata reuse, dtype conversion, CUDA Graph, and prompt prefix reuse as one coordinated flow-matching optimization stack rather than independent toggles.

## Goals

- Improve CY3 Stage1 throughput under concurrent streaming workloads.
- Avoid padding-heavy dense execution where varlen can be made efficient.
- Make the whole flow-matching path run in FP16.
- Reduce repeated CPU launch overhead with CUDA Graph capture.
- Reuse stable prompt/reference preprocessing across requests.
- Keep all changes feature gated and incrementally mergeable.
- Add quality gates strict enough to catch FP16 or cache-induced audio regressions.

## Non-Goals

- Do not optimize HiFT/vocoder in this RFC.
- Do not change Stage0 text/code generation semantics.
- Do not claim traditional LLM-style K/V tensor reuse across DiT layers or CFM steps in the MVP.
- Do not enable varlen Attention by default until pack/unpack and metadata overhead are addressed.
- Do not accept performance gains that regress quality metrics.

## Proposed PR Staging

### PR1: FP16 Flow

Run the full flow-matching path in FP16, not only the Attention submodule.

Scope:

- token embedding
- pre-lookahead
- repeat to mel timeline
- prompt mel condition construction
- CFM Euler loop
- DiT estimator
- final flow output boundary conversion as needed

Requirements:

- Add an explicit config or environment feature gate.
- Keep fallback to the current FP32 path.
- Avoid per-block Q/K/V casts by moving dtype policy to the flow boundary.
- Define the dtype boundary for mel output consumed by HiFT.

Open design point:

- Whether any small numerically sensitive buffers, such as timestep buffers or Euler accumulators, must remain FP32 should be decided by quality results. The target is full FP16 flow, but this RFC allows narrowly scoped exceptions if quality gates require them.

### PR2: Attention Batching, Varlen, and "KV Cache"

Implement varlen-first Attention batching for CY3 DiT while avoiding repeated padding computation.

The term "KV Cache" is kept as the user-facing optimization name, but the MVP does not cache semantic K/V tensors. In CY3 DiT self-attention, K/V are generated from each layer's current hidden states, and those states change by layer and CFM step. Reusing K/V tensors across layers or steps is not equivalent.

MVP cache contents:

- valid sequence lengths
- packed token indices
- `cu_seqlens`
- `max_seqlen`
- pack/unpack workspace buffers
- reusable temporary Q/K/V/output buffers where safe

Requirements:

- Build varlen metadata once per flow batch shape and reuse it across CFM steps and DiT layers where the mask and lengths are unchanged.
- Avoid reconstructing indices from a boolean mask inside every Attention call.
- Prefer direct right-padding length metadata over generic `attention_mask.nonzero()` paths.
- Minimize `_upad_input` and `_pad_input` overhead.
- Keep dense fallback for unsupported shapes, failed allocation, or backend limitations.

Expected implementation direction:

- Add a CY3-specific `FlowAttentionBatchMetadata` or equivalent abstraction.
- Pass metadata through DiT blocks without leaking CY3-specific logic into generic Attention backends where possible.
- Keep the existing `AttentionMetadata` backend contract intact unless a minimal extension is required.

### PR3: CUDA Graph for the CFM Euler Loop

Capture and replay the full 10-step CFM Euler loop when shape buckets are stable.

Scope:

- Capture the repeated CFM loop including estimator calls.
- Use static input/output buffers per bucket.
- Replay when `(batch_size, max_mel_len, dtype, device, finalize/streaming mode)` match a captured graph.
- Fall back to eager execution when shapes or runtime conditions are unsupported.

Requirements:

- Add a graph manager with explicit bucket keys.
- Include warmup and capture lifecycle management.
- Avoid graph capture for dynamic allocation paths.
- Make graph replay compatible with FP16 flow and varlen Attention metadata/workspace cache.
- Keep profiler ranges to compare eager vs graph behavior.

Open design point:

- CUDA Graph and varlen can conflict if pack/unpack allocates dynamically. PR2 should make metadata and workspace allocation explicit enough for PR3 graph capture.

### PR4: Prompt Prefix Cache

Cache stable prompt/reference preprocessing results across requests.

Cache candidates:

- prompt token embedding and pre-lookahead prefix output
- prompt mel condition prefix
- normalized and affine speaker embedding

Cache key inputs:

- prompt text/reference token ids
- reference audio or prompt mel hash
- speaker embedding hash
- model revision / flow config version
- dtype and device

Requirements:

- Add bounded memory accounting and eviction.
- Return immutable or cloned tensors to avoid cross-request mutation.
- Invalidate on model, config, or dtype changes.
- Instrument hit rate and saved preprocessing time.
- Keep miss path identical to current behavior.

## Performance Gates

This RFC intentionally does not claim a fixed speedup target. Each PR must provide benchmark and profile evidence.

Minimum benchmark matrix:

- SeedEN fixed 100-request test set
- streaming enabled
- concurrency 4 and 8
- request rate `inf`
- current FP32/dense baseline vs candidate

Required metrics:

- completed / failed requests
- request throughput
- audio throughput
- mean/P50/P90/P99 E2EL
- mean/P50/P90/P99 TTFP
- mean/P50/P90/P99 RTF
- underrun
- generated audio duration distribution

Required profile evidence:

- Stage1 CFM Euler total and average time
- estimator total and average time
- DiT transformer block time
- Attention backend time
- varlen pack/unpack time
- CUDA Graph capture/replay hit rate once PR3 lands
- prompt prefix cache hit rate once PR4 lands

Acceptance rule:

- A PR should not merge if it regresses c4/c8 end-to-end performance without explaining the regression and keeping the feature disabled by default.

## Quality Gates

FP16 flow, varlen Attention, CUDA Graph replay, and prompt prefix cache can all introduce subtle quality issues. Quality validation is required, not optional.

Required checks:

- deterministic mel diff against FP32 baseline with fixed seeds
- audio duration consistency
- non-silent output validation
- clipping / NaN / Inf checks
- ASR-based WER or CER on generated speech
- speaker similarity against reference audio where reference voice is part of the prompt
- sampled listening review for failures or borderline cases

Acceptance rule:

- Performance improvements are not acceptable if WER/CER or speaker similarity regresses beyond the agreed threshold.
- Any cache path must prove cache-hit output is equivalent to the miss path within the same quality threshold.

## Risks

- FP16 may change CFM integration behavior and degrade audio quality.
- Varlen Attention can regress performance if pack/unpack overhead exceeds saved padded compute.
- CUDA Graph requires stable shapes and explicit workspace management.
- Prompt prefix cache can return stale or mutated tensors if ownership is unclear.
- Traditional K/V reuse is not mathematically valid across CY3 DiT layers or CFM steps without a separate approximation study.

## Rollout Plan

- All features are disabled or conservative by default until benchmark and quality gates pass.
- Each PR adds an independent feature gate.
- Each PR includes fallback to the current path.
- Each PR includes profile ranges and benchmark instructions.
- Enable features progressively only after c4/c8 SeedEN validation.

## Open Questions

- What WER/CER and speaker-similarity thresholds should be used for merge gating?
- Should the FP16 flow path allow a small FP32 accumulator exception if full FP16 fails quality gates?
- What shape bucket set should be used for CUDA Graph capture?
- Should prompt prefix cache live inside the CY3 Code2Wav model or a shared vLLM-Omni prefix-cache abstraction?
- Should varlen Attention include a dense fallback based on measured padding ratio, or should the first version be strictly varlen-first behind a feature gate?

## Prior Profile Summary

The RFC is based on local Stage1 profiling and SeedEN c4 benchmark runs collected during development:

- Stage1 scheduler batching was possible, but exact-shape grouping often split real traffic into singleton flow batches.
- Attention-only bf16 varlen regressed the c4/n100 benchmark despite reducing the number of CFM Euler ranges.
- Fine-grained Attention profiling showed `flash_attn_varlen_func` was small for the tested shapes, while `_upad_input` and `_pad_input` dominated the new varlen overhead.

Future implementation PRs should include their own reproducible benchmark artifacts and trace summaries.
