# Bounded native CEE validation

This is a **new native-validation result contract**, not Stage-A/v1. Entry point:
[run_native_validation.py](../tools/run_native_validation.py). Its import, help,
plan-only mode and legacy-report mode do not import Torch, query GPUs or launch a
worker. Native kernel behavior still needs an explicitly authorized native run;
CPU contract tests are not CUDA acceptance evidence.

## Commands and authorization

Run from the checkout, with the **same Python executable** used for the successful
[CPU gate](../tools/run_cpu_gate.py). Freeze changes before producing that gate.
Keep configuration/evidence outside source-snapshot directories, e.g. `.refactor/`.
The CPU gate must have both complete default and importlib runs, corresponding
JUnit/log files, unchanged source endpoints, and matching environment metadata.

```bash
.venv/bin/python -I -B tools/run_native_validation.py \
  --validate-only --config CONFIG.json --cpu-manifest CPU/manifest.json

.venv/bin/python -I -B tools/run_native_validation.py \
  --config CONFIG.json --cpu-manifest CPU/manifest.json \
  --output-dir NEW_RUN_DIRECTORY --authorize-gpu-execution

.venv/bin/python -I -B tools/run_native_validation.py --legacy-report OLD_RESULT.json
```

Authorization is the literal full flag; abbreviations are rejected. It is checked
before reading run inputs. Output must not already exist. There is no arbitrary
command, environment-selected adapter, device discovery, spare GPU selection,
threshold override or reduced-matrix fallback. `--validate-only` is read-only,
`execution=not_started`, `verdict=plan-only`, `acceptance_eligible=false` even when
binding checks succeed. It requires neither output directory nor GPU permission.

A strict tiny configuration (replace both UUIDs with explicitly approved GPUs):

```json
{
  "schema": "duodecoding.native-config.v1",
  "profile": "tiny",
  "gpu_uuids": [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002"
  ],
  "max_seconds": 3600,
  "continue_on_numerical_mismatch": true,
  "ignore_xorg": false,
  "local": null
}
```

The UUID list is canonical, ordered, unique and exactly two entries. Its order is
the worker's logical CUDA order. The worker independently verifies actual CUDA
UUIDs. The `max_seconds` worker budget is an integer from 1 to 14400; it excludes
admission, source/asset hashing and cleanup. Each hardware query subprocess has a
15-second timeout (two subprocesses per observation), so cancellation can exceed
the worker budget by a bounded query and cleanup interval. Cleanup has separate
bounded TERM/KILL/reaping waits; no unbounded child `wait()` is used.

## Exact profiles and numerical scope

Tiny is exactly **336 cases**:

- FP32 and BF16; placements `single=(0,0,0)`, `ld-t=(0,0,1)`, `l-dt=(0,1,1)`;
- CEE DSD and DSSD; eager and Graph; temperatures 0 and 0.7;
- budgets 0, 1, 2, 3, 12; same-owner alternate sample budget 12; synthetic
  first-generated-token EOS crop budget 12.

Seeds, tiny Llama dimensions, LM-head scaling, prompts, gamma1/gamma2=2,
transfer_top_k=4, effective wire cap=1 and graph sizes 2/4/8 match the scratch
protocol. Each planned case exposes its required checks and fixed tolerances.
Successful completion also requires 168 eager/Graph token pairs and 168 DSD/DSSD
pairs, including positive-temperature cases. Pair equality does not establish
universal distributional equivalence.

Local profile changes only `profile` to `local` and supplies:

```json
"local": {
  "topology": "ld-t",
  "asset_manifest": "/absolute/path/to/assets.json"
}
```

Only `ld-t` and `l-dt` are supported for local models. It is explicitly BF16 and
exactly **56 cases**, with 28 pairs of each kind. No fixed model/revision/path is
assumed. The manifest must explicitly inventory every regular file under each
role directory, including tokenizer/configuration files. File symlinks are hashed
through their actual resolved targets; directory symlinks are refused. Example
shape (the actual three role inventories must be complete):

```json
{
  "schema": "duodecoding.native-assets.v1",
  "models": {
    "little": {"path": "/absolute/little", "files": [
      {"name": "config.json", "size": 123, "sha256": "<64-hex-actual-file-hash>"}
    ]},
    "draft": {"path": "/absolute/draft", "files": []},
    "target": {"path": "/absolute/target", "files": []}
  }
}
```

Empty, missing, extra or changed file inventories fail. Assets are actually
SHA-256 read, not trusted from size/mtime alone. Loading is offline, local-only,
no remote code, safetensors, explicit SDPA/dtype/placement, no quantization. The
production resource loader is intentionally bypassed; these are explicitly
constructed native models and real production CEE/cache/communication/timer paths,
not evidence for the production loader, real distributed RPC, quality or speed.

Unchanged allclose tolerances:

| dtype / temperature | atol | rtol |
|---|---:|---:|
| FP32 | 0.00004 | 0.0005 |
| BF16, positive temperature | 0.008 | 0.02 |
| BF16, greedy | 0.00004 | 0.02 |

Fresh eager references compare committed cache probability history. Additionally,
all observed generation-region forward rows are checked, including subsequently
rolled-back proposals (the scratch harness's optional trace scope). Comparisons
record their role/kind, row count, tolerances, mismatch rows and maximum absolute
difference. For nonfinite inputs, `nonfinite=true` and `passed=false`; the finite
JSON magnitude describes finite differences only. Continuing after numerical
mismatch collects evidence but never permits acceptance.

## Guard and provenance

[The guard](../tools/native_gpu_guard.py) takes per-user UUID locks, observes only
the authorized UUIDs twice at least 30 seconds apart, requires no compute **or
graphics** PIDs by default, zero utilization and memory below 256 MiB, rechecks bindings,
then performs an immediate prelaunch idle check. Missing/unsupported hardware data
refuses; it does not search for an alternative device. During execution it polls
UUID/index mappings and process ownership. A vanished PID is unknown, not proof
of ownership. Locks/polling are not exclusive reservations against unrelated GPU
users, and cannot rule out activity between observations.

`ignore_xorg` is an optional strict boolean and defaults to `false`. Explicitly
setting it to `true` exempts only fully described NVML `G` records whose process
name is exactly `/usr/lib/xorg/Xorg`. Missing or conflicting records, invalid
PIDs, `C`/`C+G` processes and other graphics clients are not exempt. The same policy
applies at admission, prelaunch and during execution. Original PIDs, process XML
and exemption records remain in the result; memory and utilization thresholds do
not change. Enable this only when the operator has authorized ignoring Xorg.

The worker receives an allowlisted environment, isolated Python flags, local
output cache directories, ordered UUID visibility, offline flags and a per-run
supervisor token. Inherited CUDA/PyTorch/NCCL/Python/distributed behavior overrides
are not forwarded. A fixed worker is started in a new POSIX session. SIGINT,
SIGTERM and SIGHUP (including SSH disconnect/cancellation) are deferred over
ownership handoff, then cause cleanup. Repeated cancellation signals are ignored
during cleanup; original supervisor handlers and masks are restored afterwards.
The child unblocks only these three signals before CUDA import, preserving other
inherited mask bits. Cleanup considers surviving session descendants
and their process groups even when the leader already exited. It never signals
foreign GPU PIDs. SIGKILL/host failure cannot be intercepted; inspect such an
interrupted result and external process state before retrying.

The binding includes CPU manifest/JUnit/log hashes, full CPU-gate source snapshot,
configuration bytes, actual Python/package metadata, Python and nvidia-smi binary
hashes, local asset bytes and resolved paths. Dirty/uninitialized/mismatched
submodules are refused because CPU-gate v1 does not hash their interior dirty
bytes. The same bindings are checked immediately before launch and after cleanup;
the worker independently checks its source/environment/tool endpoints. GPU driver
and CUDA runtime, model class/dtype/role placements, parameters and peak allocation
are recorded. Package versions are not package-directory content hashes. This
trusts the local CPU producer/environment, not a signed remote attestation.
**Equal endpoints do not prove continuous source/asset freezing.**

Same-run smoke covers each authorized GPU's native BF16 GEMM, changed-input and
poisoned-output explicit-stream CUDA Graph replay, private RNG sanity/default-CUDA
RNG preservation, and two-device copy. The subsequent matrix runs in that worker.
Graph observation calls original methods unchanged: `verify -> None` is fallback,
not replay. Weak live-object ordinals prevent Python `id()` reuse. All role runner/
StaticCache/capture inventories must survive budget12 -> new-sample. At least one
causal reset-before-genuine-replay reused runner is required for each new sample,
and successful step and verify calls must occur globally. This is deliberately
not an assertion that every role/branch/size/device context was replayed. Prefill,
call counts, capture counts and unchanged identities alone are not replay proof.
No device/stream diagnostic override is installed.

## Result and exit policy

[The result schema](../tools/native_validation_schema.py) rejects nonfinite JSON,
unknown/missing required checks, incomplete matrices, altered tolerances, missing
actual model inventory, and missing causal Graph evidence. Results are published
by atomic replacement; failed writes remove only the temporary file created by
that write, leaving the previous published result intact.

The public supervisor writes `result.json`; after successful binding it also
writes the exact `request.json`. Only a launched worker produces `execution.log`
and, once initialized, `worker-result.json`. A pre-execution refusal has no worker
artifacts. The worker report is retained even if its proof is rejected.
Supervisor results retain errors,
cleanup evidence, unexecuted-check reasons, before/after bindings, ordered cases,
pair checks, smoke and Graph evidence. Execution (`not_started`, `incomplete`,
`completed`), functional/numerical outcomes (`not_run`, `passed`, `failed`),
`binding_status`, `outcome_category` and `acceptance_eligible` are separate.

| exit | meaning |
|---:|---|
| 0 | accepted native run; or successful read-only operation / injected test run |
| 1 | executed failure, incomplete execution, numerical failure or interruption |
| 2 | CLI/configuration/CPU/source/environment/result contract invalid |
| 3 | pre-execution resource/idle/lock/query refusal |

Only a genuine production-adapter run with a completed plan, successful smoke,
functional and numerical passes, unchanged verified binding, successful worker
exit and successful cleanup can set `acceptance_eligible=true`. Never infer
acceptance from process exit alone.

The supported Python testing seam is `main(argv, adapters=external_os_adapter)`.
Its OS dependencies are snapshot/environment/tools, fsync, clock/sleep,
lease/query/launch/owns/cleanup. The `nvidia_smi` process-output boundary also permits
parser tests through the same public CLI adapter. No private validation helper
needs mocking.
**Every explicitly injected run is permanently ineligible**, even if all simulated
checks pass. Test adapters must honor ownership handoff or clean any process they
create before raising. Tests hide CUDA; real CPU process-tree tests use this seam
and never query hardware. See [the public tests](../test/test_native_validation_cli.py).

Legacy reporting preserves the entire original payload and input hash, reports
missing provenance as `unknown`, writes nothing and never upgrades acceptance.
The historical tiny result remains **336 cases passed**; the historical local Qwen
result remains **56 cases completed, 18 numerical-mismatch cases,
failed-numerical-equivalence**. Existing archives are not rewritten or relabeled.

## 2026-11-14 integration result

All source/test/tool writers froze before integration. Public CLI contracts passed
**59 tests in default and 59 in importlib mode**, including real CPU process-tree
cleanup, launch-handoff interruption and SSH SIGHUP regressions. Independent guard
and worker/schema reviews closed all findings. Scoped Ruff and diff checks passed.

Both local and isolated water4090 checkouts then ran the complete CPU gate with
`--require-golden-assets`, in both collection modes. All four runs reported
**2079 passed, 15 skipped, 163 subtests passed** with unchanged source endpoints.
The 15 skips comprise 7 absent historical experiment directories and 8 opt-in
native Graph tests; they are not claimed as executed. The remote run also retained
15 warnings about an optional absent acc-head registry/default-path fallback.
This is not a production-loader acceptance claim.

All 359 source-file hashes, main commit, dirty/untracked inventory and submodule
revisions match across the two checkouts. Only the submodule display annotation
(`heads/main` vs `d340008`) differs; raw snapshot JSON and environment paths are
not claimed identical. Each CPU/native binding uses its own exact snapshot.

The real public CLI successfully produced a read-only 336-case plan, then refused
native admission with **exit 3 / resource-refused**. The separate immediately
preceding UUID/index precheck saw GPU2 at 10295 MiB / 24% and GPU3 at 10307 MiB / 23%,
both with existing process IDs. No alternate GPUs, process termination or retry
was attempted. Final result: `execution=not_started`, functional/numerical
`not_run`, verified unchanged binding, `acceptance_eligible=false`, **0/336 cases
executed**; no worker or CUDA smoke started. This does not establish that the new
worker matrix passes on CUDA, and it does not supersede historical results.

Local evidence: [integration summary](<../.refactor/native-validation-20261114/integration-summary.json>),
[actual refusal](<../.refactor/native-validation-20261114/remote-evidence/native-tiny-v1/result.json>),
[independent review](<../.refactor/native-validation-20261114/independent-review.md>).
The prior two remote source snapshots and 144 package versions remained unchanged;
prior Graph code/tests/report and user HTML were preserved. No install, upgrade,
commit or push was performed. CEE-CUHLM coverage in that run was CPU-only; see
[the report for that revision](<https://github.com/Nanji-Huaji/DuoDecoding/blob/43026d08210f3456d4e49710e698108388a9c70b/docs/cee_cuhlm_refactor_20261114.md>).

## 2026-11-15 A6000 single-device experiment

The operator requested a direct single-device run rather than waiting for a new
full CPU gate. Physical GPU2 completed all **112 single-placement tiny cases**:
FP32/BF16, CEE-DSD/DSSD, eager/Graph, both temperatures and all seven scenarios.
All 1,344 numerical observations passed; 56 eager/Graph and 56 DSD/DSSD token
pairs matched. No case failed. The worker exited and released its GPU memory.

This used the real matrix implementation through a local experiment wrapper, not
the formal CLI. The separate smoke was explicitly skipped; an earlier two-device
attempt had stopped at `cross-device copy failed`. Its cause remains unresolved.
The single-device result does not establish cross-device correctness or formal
acceptance: the new full CPU gate was cancelled and `acceptance_eligible=false`.
CEE-CUHLM is not part of this matrix. Algorithms and tolerances were unchanged.
The raw reports remain in the local `.refactor/a6000-xorg-20261115/` archive.
