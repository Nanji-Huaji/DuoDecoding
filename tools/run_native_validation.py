"""Explicitly authorized native validation; imports never query or initialize CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import uuid
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import native_validation_schema as schema
from tools.native_gpu_guard import NativeSystem, admit, monitor, worker_environment


def main(argv=None, *, adapters=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpu-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--legacy-report", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--authorize-gpu-execution", action="store_true")
    args = parser.parse_args(argv)
    if args.legacy_report:
        if (
            args.config
            or args.cpu_manifest
            or args.output_dir
            or args.authorize_gpu_execution
            or args.validate_only
        ):
            parser.error("legacy-report is read-only and exclusive")
        try:
            raw = args.legacy_report.read_bytes()
            original = schema.read_json(args.legacy_report)
            print(
                json.dumps(
                    {
                        "kind": "legacy-report",
                        "original": original,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "acceptance_eligible": False,
                        "provenance": "unknown",
                    },
                    allow_nan=False,
                )
            )
            return 0
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    if not args.validate_only and not args.authorize_gpu_execution:
        parser.error("explicit GPU execution authorization is required")
    if not args.config or not args.cpu_manifest:
        parser.error("--config and --cpu-manifest required")
    external = adapters if adapters is not None else NativeSystem()
    if args.validate_only:
        if args.output_dir or args.authorize_gpu_execution:
            parser.error("validate-only is read-only; omit output and authorization")
        try:
            config, bound = schema.binding(args.config, args.cpu_manifest, external)
            print(
                json.dumps(
                    {
                        "execution": "not_started",
                        "verdict": "plan-only",
                        "binding_status": "verified",
                        "acceptance_eligible": False,
                        "configuration": config,
                        "binding": bound,
                        "plan": schema.plan(config),
                    },
                    allow_nan=False,
                )
            )
            return 0
        except Exception as exc:
            parser.error(str(exc))
    if not args.output_dir:
        parser.error("exclusive --output-dir required")
    output = args.output_dir.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    return execute(args, output, external, injected=adapters is not None)


def execute(args, output, external, *, injected):
    report = {
        "schema": "duodecoding.native-validation.v1",
        "cases": [],
        "execution": "not_started",
        "functional": "not_run",
        "numerical": "not_run",
        "binding_status": "not_checked",
        "outcome_category": "invalid-contract",
        "verdict": "not-accepted",
        "acceptance_eligible": False,
        "injected_external_adapter": injected,
        "guard_events": [],
    }
    child = bound = None
    cleanup_attempted = False
    previous_signals = {}
    phase, exit_code = "binding", 2

    def interrupted(number, frame):
        raise KeyboardInterrupt(f"received {signal.Signals(number).name}")

    def publish(path, value):
        schema.atomic_json(path, value, fsync=external.fsync)

    def record(event, **data):
        report["guard_events"].append(
            {"event": event, "time": external.clock(), **data}
        )
        publish(output / "result.json", report)

    def recheck_binding():
        nonlocal phase
        phase = "binding"
        _, current = schema.binding(args.config, args.cpu_manifest, external)
        schema.require(
            current == bound,
            "source/environment/config/assets/CPU evidence changed before launch",
        )
        phase = "resources"

    def cleanup():
        nonlocal cleanup_attempted
        if child is not None and not cleanup_attempted:
            cleanup_attempted = True
            for number in previous_signals:
                signal.signal(number, signal.SIG_IGN)
            report["cleanup"] = external.cleanup(child)
            schema.require(
                not report["cleanup"]["remaining"], "owned processes survived cleanup"
            )

    publish(output / "result.json", report)
    try:
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_signals[number] = signal.signal(number, interrupted)
        config, bound = schema.binding(args.config, args.cpu_manifest, external)
        report.update(
            configuration=config,
            binding_before=bound,
            binding_status="verified",
            plan=schema.plan(config),
        )
        report["worker_environment"] = worker_environment(output, config["gpu_uuids"])
        authorization = uuid.uuid4().hex
        report["worker_environment"]["DUODECODING_NATIVE_AUTHORIZATION"] = authorization
        request = {
            "configuration": config,
            "binding": bound,
            "plan": report["plan"],
            "authorization": authorization,
            "worker_environment": report["worker_environment"],
        }
        publish(output / "request.json", request)
        phase = "resources"
        with external.lease(config["gpu_uuids"]):
            try:
                mapping = admit(
                    external,
                    config["gpu_uuids"],
                    record,
                    recheck_binding,
                    ignore_xorg=config["ignore_xorg"],
                )
                old_mask = signal.pthread_sigmask(
                    signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
                )
                try:
                    child = external.launch(output, report["worker_environment"])
                finally:
                    signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                phase = "execution"
                report["execution"] = "incomplete"
                schema.require(
                    not getattr(child, "native_launch_error", None),
                    getattr(child, "native_launch_error", "worker launch failed"),
                )
                record("launched", pid=child.pid)
                report["worker_returncode"] = monitor(
                    external, child, config, mapping, record
                )
            finally:
                cleanup()
            worker = schema.read_json(output / "worker-result.json")
            phase = "worker-contract"
            if isinstance(worker, dict):
                if "error" in worker:
                    report["worker_error"] = worker["error"]
                if isinstance(worker.get("cases"), list):
                    report["worker_case_errors"] = [
                        {"case_id": case.get("id"), "error": case["error"]}
                        for case in worker["cases"]
                        if isinstance(case, dict) and "error" in case
                    ]
            report.update(schema.worker_outcome(worker, request))
            report["worker_result_sha256"] = schema.file_hash(
                output / "worker-result.json"
            )
            if "error" in worker:
                report["worker_error"] = worker["error"]
                report["functional"] = "failed"
            exit_code = 1
            report["outcome_category"] = "execution-incomplete"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, KeyboardInterrupt):
            exit_code, report["outcome_category"] = 1, "interrupted"
        elif phase in ("binding", "worker-contract"):
            exit_code, report["outcome_category"] = 2, "invalid-contract"
            report["binding_status"] = "failed"
        elif phase == "resources" and child is None:
            exit_code, report["outcome_category"] = 3, "resource-refused"
        else:
            exit_code, report["outcome_category"] = 1, "execution-incomplete"
    finally:
        try:
            # Outer lifetime guard also covers a signal between Popen assignment
            # and the inner finally, and failures while publishing launch events.
            if child is not None and report["execution"] == "not_started":
                report["execution"] = "incomplete"
            try:
                cleanup()
            except Exception as exc:
                report["cleanup_error"] = str(exc)
                report["error"] = "owned-process cleanup failed"
                exit_code, report["outcome_category"] = 1, "execution-incomplete"
            if bound is not None:
                try:
                    _, after = schema.binding(args.config, args.cpu_manifest, external)
                    report["binding_after"] = after
                    schema.require(
                        after == bound,
                        "source/environment/config/assets/CPU evidence changed",
                    )
                except Exception as exc:
                    report["error"] = f"binding recheck failed: {exc}"
                    report["binding_status"] = "failed"
                    exit_code, report["outcome_category"] = 2, "invalid-contract"
            success = (
                "error" not in report
                and "worker_error" not in report
                and report.get("worker_returncode") == 0
                and report["execution"] == "completed"
                and report["functional"] == report["numerical"] == "passed"
            )
            report["acceptance_eligible"] = success and not injected
            report["verdict"] = (
                "failed-numerical-equivalence"
                if report["numerical"] == "failed"
                else "accepted"
                if report["acceptance_eligible"]
                else "injected-not-accepted"
                if success
                else "not-accepted"
            )
            if success:
                exit_code, report["outcome_category"] = (
                    0,
                    "injected" if injected else "accepted",
                )
            elif "error" not in report and report["numerical"] == "failed":
                exit_code, report["outcome_category"] = 1, "numerical-failed"
            elif "error" not in report and report["functional"] == "failed":
                exit_code, report["outcome_category"] = 1, "functional-failed"
            report["exit_code"] = exit_code
            observed = {row["id"]: row for row in report["cases"]}
            report["unexecuted_checks"] = []
            for spec in report.get("plan", []):
                case = observed.get(spec["id"], {})
                missing = [
                    name
                    for name in spec["required_checks"]
                    if case.get("checks", {}).get(name, "not_run") == "not_run"
                ]
                if missing:
                    report["unexecuted_checks"].append(
                        {
                            "case_id": spec["id"],
                            "checks": missing,
                            "reason": case.get("error")
                            or report.get("worker_error")
                            or report.get("error")
                            or "worker did not reach this planned check",
                        }
                    )
            publish(output / "result.json", report)
        finally:
            for number, previous in previous_signals.items():
                signal.signal(number, previous)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
