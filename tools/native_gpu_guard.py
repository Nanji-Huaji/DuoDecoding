"""External OS adapter and bounded authorized-GPU process guard; stdlib only."""

import fcntl
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack, contextmanager
from pathlib import Path

from tools.native_validation_schema import file_hash, require
from tools.run_cpu_gate import environment_metadata, source_snapshot

ROOT = Path(__file__).resolve().parents[1]


class NativeSystem:
    """Production external adapter. Python callers may inject a fake at main()."""

    def snapshot(self):
        snapshot = source_snapshot(ROOT)
        require(
            all(line.startswith(" ") for line in snapshot["submodules"].splitlines()),
            "uninitialized/mismatched/conflicted submodule not bound",
        )
        # CPU-gate v1 binds gitlinks, not dirty submodule bytes: refuse them.
        subprocess.run(
            [
                "git",
                "submodule",
                "foreach",
                "--quiet",
                "--recursive",
                "status=$(git status --porcelain --untracked-files=all) "
                '&& test -z "$status"',
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return snapshot

    def environment(self):
        return environment_metadata()

    def tools(self):
        def describe(path):
            return (
                {
                    "path": path,
                    "resolved": str(Path(path).resolve()),
                    "sha256": file_hash(path),
                }
                if path
                else None
            )

        return {
            "python": describe(sys.executable),
            "nvidia_smi": describe(shutil.which("nvidia-smi", path=os.defpath)),
        }

    clock = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)
    fsync = staticmethod(os.fsync)

    @contextmanager
    def lease(self, ids):
        # Cross-output-directory coordination, not a reservation against others.
        directory = Path(f"/tmp/duodecoding-native-{os.getuid()}")
        directory.mkdir(mode=0o700, exist_ok=True)
        with ExitStack() as stack:
            for identity in sorted(ids):
                stream = stack.enter_context((directory / identity).open("a"))
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def nvidia_smi(self, ids, options):
        """External process boundary; query parsing remains production behavior."""
        binary = shutil.which("nvidia-smi", path=os.defpath)
        require(binary is not None, "nvidia-smi unavailable")
        return subprocess.run(
            [binary, "--id=" + ",".join(ids), *options],
            check=True,
            text=True,
            capture_output=True,
            timeout=15,
            env={"PATH": os.defpath, "LANG": "C"},
        ).stdout

    def query(self, ids):
        def invoke(options):
            return self.nvidia_smi(ids, options)

        selected = {}
        fields = "index,uuid,memory.used,utilization.gpu,driver_version"
        for line in invoke(
            ["--query-gpu=" + fields, "--format=csv,noheader,nounits"]
        ).splitlines():
            index, identity, memory, utilization, driver = [
                x.strip() for x in line.split(",")
            ]
            require(
                identity in ids and identity not in selected,
                "unexpected GPU query identity",
            )
            selected[identity] = {
                "index": int(index),
                "memory_mib": int(memory),
                "utilization": int(utilization),
                "driver": driver,
                "pids": [],
                "processes": [],
            }
        require(set(selected) == set(ids), "authorized GPU UUID missing")
        # compute-apps alone omits graphics processes. The UUID-restricted XML
        # process table includes both C and G clients; unsupported data refuses.
        xml = ET.fromstring(invoke(["-q", "-x"]))
        seen = set()
        for gpu in xml.findall("gpu"):
            identity = gpu.findtext("uuid")
            require(
                identity in selected and identity not in seen,
                "unexpected process-query identity",
            )
            seen.add(identity)
            processes = gpu.find("processes")
            require(
                processes is not None
                and not (processes.text or "").strip()
                and all(child.tag == "process_info" for child in processes),
                "GPU process evidence unavailable",
            )
            for process in processes.findall("process_info"):
                pid_fields = process.findall("pid")
                require(
                    len(pid_fields) == 1 and not len(pid_fields[0]),
                    "ambiguous or missing GPU process PID evidence",
                )
                pid = int(pid_fields[0].text)
                selected[identity]["pids"].append(pid)
                evidence = {
                    "pid": pid,
                    "raw_xml": ET.tostring(process, encoding="unicode"),
                }
                for name in ("type", "process_name", "used_memory"):
                    fields = process.findall(name)
                    evidence[name] = (
                        fields[0].text
                        if len(fields) == 1 and not len(fields[0])
                        else None
                    )
                selected[identity]["processes"].append(evidence)
        require(seen == set(ids), "GPU process evidence missing authorized UUID")
        return selected

    def launch(self, output, env):
        log = (output / "execution.log").open("x", buffering=1)
        child = None
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(ROOT / "tools/native_validation_worker.py"),
                    "--request",
                    str(output / "request.json"),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            try:
                log.close()
            except Exception as exc:
                if child is None:
                    raise
                # Never discard a live child's ownership receipt after spawn.
                child.native_launch_error = f"execution log close failed: {exc}"
        return child

    def owns(self, pid, child):
        try:
            return os.getsid(pid) == child.pid
        except ProcessLookupError:
            return False  # Vanished is unknown, never positive ownership proof.

    def cleanup(self, child):
        # Enumerate the owned session, NOT just leader.poll(). Its children may
        # outlive a successful/failed leader or establish additional groups.
        def members():
            found = {}
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    pid = int(entry.name)
                    try:
                        if os.getsid(pid) == child.pid:
                            found[pid] = os.getpgid(pid)
                    except ProcessLookupError:
                        pass
            return found

        def send(sig):
            for group in set(members().values()):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass

        send(signal.SIGTERM)
        deadline = time.monotonic() + 3
        while members() and time.monotonic() < deadline:
            child.poll()
            time.sleep(0.05)
        send(signal.SIGKILL)
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass

        # SIGKILL delivery is asynchronous even after the leader is reaped.
        # Zombies cannot execute GPU work and belong to their actual parent.
        def live_members():
            remaining = []
            for pid in members():
                try:
                    state = (
                        (Path("/proc") / str(pid) / "stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[0]
                    )
                    if state != "Z":
                        remaining.append(pid)
                except FileNotFoundError:
                    pass
            return remaining

        deadline = time.monotonic() + 3
        remaining = live_members()
        while remaining and time.monotonic() < deadline:
            time.sleep(0.02)
            remaining = live_members()
        return {"session": child.pid, "remaining": remaining}


def worker_environment(output, ids):
    # Allowlist: strip CUDA/PyTorch/NCCL/accelerator/distributed/plugin behaviour
    # overrides rather than trying to enumerate every harmful inherited knob.
    return {
        "PATH": os.defpath,
        "HOME": str(output / "home"),
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": ",".join(ids),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_CACHE_PATH": str(output / "cuda-cache"),
        "XDG_CACHE_HOME": str(output / "runtime-cache"),
        "HF_HOME": str(output / "hf-cache"),
        "TORCH_HOME": str(output / "torch-cache"),
    }


def process_policy(rows, ignore_xorg):
    """Only unanimous, fully described system-Xorg graphics records are exempt."""
    by_pid = {}
    if ignore_xorg:
        for row in rows.values():
            processes = row.get("processes", [])
            complete = (
                isinstance(processes, list)
                and all(isinstance(process, dict) for process in processes)
                and [process.get("pid") for process in processes] == row["pids"]
            )
            # Missing, extra or conflicting details cannot corroborate that PID
            # on another authorized GPU, even when that other row looks like Xorg.
            if isinstance(processes, list):
                for process in processes:
                    if isinstance(process, dict):
                        by_pid.setdefault(process.get("pid"), []).append(
                            process if complete else {}
                        )
            if not complete:
                for pid in row["pids"]:
                    by_pid.setdefault(pid, []).append({})
    exempt = {
        pid
        for pid, observations in by_pid.items()
        if ignore_xorg
        and type(pid) is int
        and pid > 0
        and all(
            process.get("type") == "G"
            and process.get("process_name") == "/usr/lib/xorg/Xorg"
            and isinstance(process.get("used_memory"), str)
            and re.fullmatch(r"[0-9]+ MiB", process["used_memory"]) is not None
            for process in observations
        )
    }
    return {
        "ignore_xorg": ignore_xorg,
        "effective_pids": {
            identity: [pid for pid in row["pids"] if pid not in exempt]
            for identity, row in rows.items()
        },
        "ignored_xorg": [
            {"gpu_uuid": identity, "process": process}
            for identity, row in rows.items()
            for process in row.get("processes", [])
            if process["pid"] in exempt
        ]
        if exempt
        else [],
    }


def monitor(external, child, config, mapping, record):
    start = external.clock()
    while True:
        require(
            external.clock() - start < config["max_seconds"],
            "worker time budget exceeded",
        )
        rows = external.query(config["gpu_uuids"])
        require(
            {u: r["index"] for u, r in rows.items()} == mapping,
            "GPU identity changed during run",
        )
        policy = process_policy(rows, config["ignore_xorg"])
        foreign = [
            pid
            for pids in policy["effective_pids"].values()
            for pid in pids
            if not external.owns(pid, child)
        ]
        record("monitor", gpus=rows, foreign=foreign, **policy)
        require(
            not foreign, "foreign GPU process detected; terminate only owned session"
        )
        require(
            external.clock() - start < config["max_seconds"],
            "worker time budget exceeded",
        )
        status = child.poll()
        if status is not None:
            return status
        external.sleep(min(1, config["max_seconds"] - (external.clock() - start)))


def admit(external, ids, record, prelaunch=lambda: None, *, ignore_xorg=False):
    def idle(label):
        rows = external.query(ids)
        require(set(rows) == set(ids), "authorized UUID query mismatch")
        policy = process_policy(rows, ignore_xorg)
        record(label, gpus=rows, **policy)
        require(
            all(
                type(row["index"]) is int
                and row["index"] >= 0
                and type(row["memory_mib"]) is int
                and 0 <= row["memory_mib"] < 256
                and type(row["utilization"]) is int
                and row["utilization"] == 0
                and policy["effective_pids"][identity] == []
                for identity, row in rows.items()
            ),
            "authorized GPUs not idle",
        )
        return {identity: row["index"] for identity, row in rows.items()}

    first = idle("idle-observation-1")
    start = external.clock()
    external.sleep(30)
    require(
        external.clock() - start >= 30, "idle observations less than 30 seconds apart"
    )
    require(idle("idle-observation-2") == first, "GPU mapping changed")
    prelaunch()
    require(idle("immediate-prelaunch-check") == first, "GPU mapping changed")
    return first
