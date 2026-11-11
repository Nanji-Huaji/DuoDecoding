# CEE effective wire-draft law: `wire-draft-v1`

Base: PR39/main `6314b426`. This is an intentional correction of CEE intermediate probability semantics, not a promise of historical token/metric replay.

## Approved law

The user approved **Q_D = Decode(Encode(D))**:

- Little candidates use their decoded wire law Q_L.
- Inner acceptance, positive residual and all-accepted bonus all target Q_D.
- Carried tokens retain their original Q_D payloads, including the continuation. Its marginal proposal law is Q_D, not the conditional rejection residual.
- Fresh draft candidates use the same Q_D codec before sampling.
- Outer verification still targets the full target-model sampler law P.

The existing sampler contract is unchanged: validate raw rows, apply the established FP32 sampling-weight preprocessing, then use FP64 normalized laws for comparisons and residuals. Claims concern that finite-precision sampler law, not ideal real-arithmetic softmax.

Model filtering and transmission compression remain separate. The model cache receives the **original requested** top-k (or the existing model default); the communication cap selects the codec width afterward. Moving the cap earlier, changing only bonus sampling, hard truncation, or relabeling already sampled raw-D tokens would not implement this contract.

## Owned boundaries

- [CEE proposal/codec](<../src/cee_proposal.py>): owned payload snapshots, decoding, little/fresh-draft production and carried-row preservation.
- [CEE numerical stages](<../src/cee_cascade.py>): strict verification preparation, decisions, continuation and cache endpoints. No simulator state.
- [CEE protocols](<../src/baselines/cee_distributed.py>): separate DSD block and DSSD selected-scalar loops. No flag-driven universal round engine.

The six old CEE-DSD/DSSD stage/entry methods are removed from the central baseline implementation. Public registry names and NTT defaults remain unchanged; the methods' defining class is now `CeeDistributedMixin`. `cee_cuhlm`, `ceesd_without_arp`, their live helpers and callers are not retired or redirected.

## Wire precision and accounting

For `0 < K < V`, encode exactly K entries using stable descending order (lower vocabulary index wins ties). Preserve probability dtype, store actual int32 indices, and store one tail scalar in that same probability dtype. Compute the tail from the **sum of omitted entries** in FP64, divide by `V-K`, then cast once to payload dtype. Do not derive tiny omitted mass by subtracting a rounded top-k sum from one. A full-width row has implicit indices and no tail metadata.

Decoded rows are the source sampler inputs as well as the receiver's verification inputs. Raw model cache histories are not rewritten. Sparse top-k cutoff ties can retain more model support than K; the codec still emits exactly K indices plus a uniform tail.

- DSD uploads candidate tokens plus complete encoded Q payloads.
- DSSD uploads candidate tokens plus **FP64 normalized selected-Q scalars**, and passes those same scalar values to the strict verifier. This intentionally changes scalar precision/accounting; it does not bill FP16 while using an untransmitted FP64 value.
- Token bytes follow the actual tensor dtype. Sparse probability indices are separately int32.
- On rejection, the sender already owns Q. Honest residual accounting includes the exact encoded Q_D verifier payload on the inner link, or the full native target-P row on the outer link, before continuation sampling. The legacy `charge_residual_payload=False` discount omits those bytes only, not the correction mathematics or rejection signal.
- Message-history entries correspond to actual charged messages, not extra synthetic draft records.

CEE does not consume `prob_payload_bits`; that setting does not quantize this codec. Probability payloads retain their native dtype, and selected-Q scalars use FP64. Per-call metrics expose `cee_probability_policy=wire-draft-v1` and `rng_policy=private-device-v1`. Existing aggregate evaluators need not preserve these non-additive fields; retain the run's Git revision when comparing results.

This remains an in-process communication simulation, not tested serialization/RPC transport.

## RNG, budget and cache invariants

One lazy `DecodingRng` spans the whole invocation and both stages, including rejection paths. Roles on the same physical device advance the same private generator; different devices retain separate private generators. No reseeding or cloned role streams. `forward_context` performs verifier/cache work without a discarded categorical sample.

For remaining budget R:

- R=1: target-only tail.
- Otherwise little proposals are bounded by R-2, then fresh draft proposals by R-1 minus the **actual** carried count.
- Rejected speculative work can consume more draws than emitted output; unused verifier samples cannot.
- The final emitted token stays unforwarded. Confirmed EOS truncates output and all longer caches to the preceding context. Never extend a physically shorter cache through rollback.

## Acceptance evidence and limits

[Joint RNG tests](<../test/test_cee_rng.py>) and [CEE behavior tests](<../test/test_cee_cascade_law.py>) cover:

- Real tiny CPU Llama/KV cache cap counterexample and uncapped threshold-tie behavior.
- Exact finite dependency-tree enumeration of nontrivial stopped inner chunks, both rejection layers, variable outer window lengths, fresh Q_D suffixes and all 64 three-token joint outcomes.
- A diagnostic mutation retaining correct bonus/payloads but reverting inner acceptance/residual to raw D. Both modes reject it: first marginal becomes `[1/4, 1/3, 5/24, 5/24]`, not uniform.
- Native CPU samplers/cache budgets 1, 2, 3 and multi-round 8; generator identity, EOS and zero-budget behavior.
- Real context-dependent CPU models: repeatedly reject at both layers, then compare every retained cache probability row against fresh full-context inference.
- FP16/BF16/FP32/FP64 payloads, tail support, selected scalar equality, exact message bytes, both residual-accounting settings, and invalid rows before RNG.

These are bounded, explicit mathematical/implementation tests, not a universal proof. CPU substitutes do **not** establish CUDA/Philox independence, CUDA Graph behavior or multi-GPU transport. No GPU experiment is authorized on an occupied device.

## CPU batch acceptance

The single whole-batch GLM-5.3 review returned no actionable findings and confirmed genuine decomposition. Frozen strict gates (`--require-golden-assets`) passed in both default and importlib modes: **1985 passed, 7 skipped, 163 subtests passed** each. All14 historical checkpoint fingerprint checks passed in each mode; all351 source hashes matched before/after execution. Only the two intentional CEE logic entries changed; the other13 and historical output goldens stayed unchanged.

This does not close GPU acceptance. CUDA/Philox, CUDA Graph replay and multi-GPU placement remain pending until idle-device checks permit experiments. The new focused tests also do not separately certify precise/stochastic simulator variants. At2026-11-11 20:21 +08:00, all four remote GPUs had active compute processes and100% utilization; no experiment was started.
