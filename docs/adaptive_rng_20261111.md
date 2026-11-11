# Adaptive RNG repair — 2026-11-11

## Scope and status

This batch repairs the **two-model `adaptive_decoding` token RNG ownership only**. Its base is `bf618fc6a0a1ec72fc44079086004907653580c4` (TK PR #38). It does not migrate CEE, change the shared sampler/model cache, or claim that adaptive's legacy raw-probability ratio is now a fully strict numerical verifier. GPU validation remains pending.

RNG policy identifier: **`private-device-v1`**. Reusing the same user seed no longer promises the old token bitstream. Experiments must retain their source commit and distinguish this policy from the old default-stream implementation; historical output fixtures have not been regenerated.

## Defect and evidence

Previously proposals drew on the draft default generator. Acceptance uniforms also drew there, but bonus and remaining-budget tail tokens were sampled on the target device **before** moving the token back. `torch.manual_seed` initializes CUDA default generators alike. Consequently different devices can use equal keys/offsets for equal-ordinal categorical draws.

Consider a context-independent uniform vocabulary of four tokens, equal draft/target laws, `gamma=1`, and a two-token budget. Acceptance is certain. If the first proposal and first target bonus use cloned streams, the pair is always diagonal. The intended independent target law gives each of the16 pairs probability1/16. Correct single-position marginals do not imply a correct sequence joint law.

The deterministic public-entry regression enumerates an8×8 grid of categorical tape values. Before the fix it produces only four diagonal pairs, each16 times; after the fix each of the16 pairs occurs4 times. This is an **inverse-CDF dependency substitute on CPU**, not a CUDA multinomial emulator or a real multi-device run. A separate pre-fix native-CPU logical-role probe also observed64/64 repeated pairs, versus21/64 with one sequential stream; that finite control is not a mathematical correctness proof.

## Implementation and preserved contracts

- One lazy `DecodingRng` is created per public decode invocation and passed through proposal, serial acceptance and continuation. The target-only tail uses the same context. Same actual device: one advancing generator. Distinct actual devices: distinct private keys.
- Adaptive now owns its adapter-only proposal loop: `forward_context → rebuild_topk_uniform_probs → sample → adapter.predict`. This is the existing shared loop's reachable adapter path, not a new sampler law. Rebuilt proposal rows, final raw entropy source and adapter-stop ordering are retained.
- The shared `SharedOperations`, `KVCacheModel`, `sampling`, `decoding_ops`, CEE, dual and standalone CUHLM implementations are untouched. There is no ambient/global RNG monkeypatch in production and no new shared callback interface.
- Policy/exploration randomness is not token RNG and is not migrated. Policy selection/reward timing, top-k application, communication, rollback, budgets and EOS behavior retain their previous implementation. In particular this patch does not silently fix unrelated numerical or stopping issues.
- Direct callers of the three private adaptive helpers may omit the optional context and retain their old sampler behavior. The public entry always supplies the context. Tests exercise the public entry; the existing helper-level compatibility test only retargets its sampler patch to the loop's new owner.
- The metadata seed draw is lazy, not a discarded token sample. Zero budget consumes no default CPU RNG state. Forwarding still does not sample a discarded target token. Bonus remains sampled on the actual target probability device; no full bonus P row is moved to draft merely to avoid RNG work.

## Tests and review

- Initial public-entry RED: **4 failures,1 pass**, all intended (joint law; missing explicit generators for accept/bonus, residual, tail; zero-budget guard passes).
- Seven new tests cover the exact joint-law grid, explicit generators for all token/acceptance paths, multi-round reuse including final tail, a real tiny CPU Llama/cache/communication run, and zero-budget state. CUDA timing is replaced even in the real-model CPU test; no GPU inference is claimed.
- An in-memory mutation moving context creation inside the round loop fails the public multi-round test (`3 != 1` generators); the source file is never mutated and the original callable is restored.
- Before implementation the four existing adaptive suites passed38 tests and73 subtests. After implementation the only observed compatibility failure was a sampler patch aimed at the old shared-loop location; retargeting it preserves every assertion. The historical orchestration comparison still checks communication, policy/reward, cache and output behavior on controlled laws; its existing exclusion of intentionally changed RNG values is unchanged.
- Modified source/test files pass Ruff. Fingerprint/package tests:33 passed.
- Strict `--require-golden-assets` CPU gate: **1920 passed,7 skipped,163 subtests passed** in each mode (final v2: default95.49s, importlib111.73s). Both JUnit files contain1927 cases and **14 actual checkpoint-fingerprint passes**. The7 skips are optional original-run artifact comparisons, not weight checks or CUDA runs. All346 source hashes match before, after and the current worktree.
- Single GLM whole-batch review: **no P0/P1/P2 findings; one P3** obsolete adapter-stop test docstring. Lead corrected that docstring only, then reran the strict gate in a new v2 directory. The reviewer independently checked the producer-loop move, RNG routing, unchanged other families and test claims. No second review pipeline was started.

Selective logic footprint: `AdaptiveDecodingMixin.adaptive_decoding` **147→159** events, `202515d01e0d1702→cf5b6a6ae102b131`. These are static structural events, not executed random draw counts. Other14 entries and the include list are unchanged.

## Remaining work and hardware

CEE-DSD and CEE-DSSD have separate confirmed logical-role counterexamples and are **not fixed by this patch**. Modern CEE, CEE-CUHLM and raw-ratio/sampler-law numerics still need their own scoped work. Scoped audit found no analogous cross-device default-stream pair in the currently supported SD/DSD/DSSD and standalone-CUHLM layouts: their verification and continuation settle on the proposal device. That is not an overall correctness certificate.

Legacy `ceesd_without_arp` is still called by `scripts/run_baseline_table.sh`; external usage is unknown. It is not retired or redirected to `cee_sd_no_arp`. Historical recipes and weights remain untouched. The old script's process-name wait is not an acceptable GPU-idleness guard and was not executed.

At **2026-11-11 16:30:26 +08:00**, all four available A6000 GPUs were busy (utilization100/100/99/99%, memory19047/19027/19055/46350MiB). No GPU experiment was launched. Five earlier bounded idle waits still total about181min with0/22 completed Stage-A cases; later point observations are not additional waits. The Stage-A frozen checkout does not validate this later adaptive implementation. Actual multi-device routing and CUDA Graph validation are pending; renting one card can test native CUDA kernels with separate generators but cannot replace a real same-host two-GPU integration run.
