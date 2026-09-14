#!/usr/bin/env python3
"""迁移后的环境自检：逐项核对依赖、GPU、模型、检查点、数据集与网络。

为什么要单独写这个
------------------
本仓库的失败模式很分散，且**报错信息往往指向错误的方向**：
  · 代理"看起来通"（CONNECT 返回 200）但实际不转发 ⇒ 所有 TLS 握手 EOF；
  · acc_head 不在 git 里（在 HF Hub），缺了它代码能 import 但结果全错；
  · gsm8k 等数据集是运行时从 HF 拉的，缓存缺失时评测会卡在重试里。
这个脚本把这几类问题在一次运行里全查出来，并给出可执行的修复建议。

用法
----
    python scripts/setup/check_env.py            # 常规自检
    python scripts/setup/check_env.py --smoke    # 额外做一次真实前向（需 GPU，约 30s）
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

def _bootstrap_interpreter() -> None:
    """直接执行本脚本时，若当前解释器缺依赖，则自动切到仓库自带的 .venv。

    为什么需要：`./scripts/setup/check_env.py` 走 shebang，用的是系统 python3，
    那里没有 torch/transformers，会报一堆看不懂的"依赖缺失"。而 conda 等
    已经装好依赖的环境则应当沿用，所以判据是"当前解释器能不能 import torch"。
    """
    import importlib.util
    if importlib.util.find_spec("torch") is not None:
        return
    venv_py = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python"
    if venv_py.exists() and Path(sys.executable) != venv_py:
        print(f"[setup] 当前解释器缺依赖，切换到 {venv_py}", flush=True)
        os.execv(str(venv_py), [str(venv_py), *sys.argv])


_bootstrap_interpreter()

ROOT = Path(__file__).resolve().parents[2]
ASSETS = json.loads((ROOT / "scripts/setup/assets.json").read_text(encoding="utf-8"))

OK, BAD, WARN = "✓", "✗", "!"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, item: str, detail: str = "") -> None:
        self.rows.append((status, item, detail))

    def dump(self, title: str) -> tuple[int, int]:
        print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
        ok = bad = 0
        for status, item, detail in self.rows:
            print(f"  {status} {item:38} {detail}")
            ok += status == OK
            bad += status == BAD
        self.rows.clear()
        return ok, bad


def probe(host: str, port: int, timeout: float = 0.8) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def proxy_really_works(port: int = 7890) -> tuple[bool, str]:
    """判定代理**是否真的在转发**，而不是只看 CONNECT 应答。

    教训：clash 对 CONNECT 一律回 '200 Connection established'，即使上游节点已死；
    必须真正读一次 HTTP 状态码（或完成 TLS 握手）才能判断。
    """
    import ssl

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=8)
        s.sendall(b"CONNECT github.com:443 HTTP/1.1\r\nHost: github.com:443\r\n\r\n")
        if b"200" not in s.recv(256):
            return False, "代理拒绝 CONNECT"
        ctx = ssl.create_default_context()
        t = ctx.wrap_socket(s, server_hostname="github.com")
        ver = t.version()
        t.close()
        return True, f"TLS 握手成功（{ver}）"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:60]}"


def check_python(rep: Report) -> None:
    v = sys.version_info
    rep.add(OK if (v.major, v.minor) >= (3, 10) else BAD,
            "Python 版本", f"{v.major}.{v.minor}.{v.micro}（需要 >=3.10）")
    req = ROOT / ".python-version"
    if req.exists():
        want = req.read_text().strip()
        rep.add(OK if want in f"{v.major}.{v.minor}" else WARN,
                ".python-version 一致", f"声明 {want}，实际 {v.major}.{v.minor}")


def check_packages(rep: Report) -> None:
    import importlib

    for mod, why in [
        ("torch", "训练/推理"), ("transformers", "模型"), ("accelerate", "启动器"),
        ("bitsandbytes", "4bit 量化（论文默认）"), ("datasets", "数据集"),
        ("huggingface_hub", "模型下载"), ("numpy", "数值"), ("pandas", "分析"),
    ]:
        try:
            m = importlib.import_module(mod)
            rep.add(OK, f"依赖 {mod}", getattr(m, "__version__", "已安装") + f"  ({why})")
        except Exception as exc:  # noqa: BLE001
            rep.add(BAD, f"依赖 {mod}", f"缺失: {type(exc).__name__}  ({why})")


def check_gpu(rep: Report, smoke: bool) -> None:
    try:
        import torch
    except Exception:  # noqa: BLE001
        rep.add(BAD, "CUDA", "torch 不可用，跳过")
        return
    if not torch.cuda.is_available():
        rep.add(BAD, "CUDA 可用", "torch.cuda.is_available() == False")
        return
    rep.add(OK, "CUDA 可用", f"torch {torch.__version__} / cuda {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        rep.add(OK, f"GPU{i} {p.name}", f"{total / 2**30:.0f} GiB（空闲 {free / 2**30:.0f} GiB）")
    if smoke:
        try:
            import bitsandbytes  # noqa: F401
            a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
            torch.matmul(a, a)
            rep.add(OK, "bf16 前向", "正常")
        except Exception as exc:  # noqa: BLE001
            rep.add(BAD, "bf16 前向", f"{type(exc).__name__}: {str(exc)[:60]}")


def load_registry() -> dict:
    path = ROOT / ASSETS["acc_head_registry"]
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {f"{e['source']}--to--{e['target']}": e for e in data}
    except Exception:  # noqa: BLE001
        return {}


def has_weights(path: Path) -> bool:
    if not (path / "config.json").exists():
        return False
    return any(p.suffix in (".safetensors", ".bin", ".pt", ".gguf")
               for p in path.rglob("*") if p.is_file())


def check_assets(rep: Report, model_set: str) -> None:
    wanted = ASSETS["sets"][model_set]["models"]
    for name in wanted:
        spec = ASSETS["models"][name]
        local = ROOT / spec["local"]
        if has_weights(local):
            rep.add(OK, f"模型 {name}", f"{spec['local']}  ({spec['tier']})")
        else:
            hint = "需 HF_TOKEN + 接受许可" if spec.get("gated") else "可脚本下载"
            rep.add(BAD, f"模型 {name}", f"缺失 {spec['local']}（{hint}）")

    for key in ASSETS["acc_head_required"]:
        local = ROOT / "src/SpecDec_pp/checkpoints/acc_head" / key
        if has_weights(local):
            rep.add(OK, f"acc_head {key.split('/')[0]}", "已就位")
        else:
            rep.add(BAD, f"acc_head {key.split('/')[0]}",
                    "缺失（不在 git 里！用 download_assets.py 从 HF 拉")

    for spec in ASSETS["datasets"]:
        name = spec["name"]
        cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        hits = list(cache.glob(f"datasets--{name.replace('/', '--')}*"))
        rep.add(OK if hits else WARN, f"数据集 {name}",
                "已缓存" if hits else "未缓存（评测时会联网拉取；可用 download_assets.py 预热）")

    for f in sorted((ROOT / "data").glob("*.jsonl")):
        rep.add(OK, f"数据 {f.name}", f"{f.stat().st_size / 1e6:.1f} MB")
    mc = ROOT / "configs/compute_cost_model.json"
    rep.add(OK if mc.exists() else WARN, "成本模型 configs/compute_cost_model.json",
            "存在" if mc.exists() else "缺失（经济性分析会退化为默认值）")


def check_network(rep: Report, args) -> None:
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if os.environ.get(k):
            rep.add(OK, f"环境变量 {k}", os.environ[k])
            break
    else:
        rep.add(WARN, "代理环境变量", "未设置（脚本可用 --proxy auto 探测）")

    port = args.proxy_port
    if not probe("127.0.0.1", port):
        rep.add(WARN if not args.require_proxy else BAD, f"本地代理 :{port}", "端口未监听")
        return
    ok, detail = proxy_really_works(port)
    rep.add(OK if ok else BAD, f"本地代理 :{port} 实际转发", detail)
    if not ok:
        rep.add(WARN, "代理修复建议",
                "CONNECT 返回 200 但上游已死 ⇒ 更新订阅后需重载："
                "curl -X PUT 'http://127.0.0.1:9090/configs?force=true' -d '{\"path\":\"<config.yaml>\"}'")


def main() -> int:
    ap = argparse.ArgumentParser(description="DuoDecoding 环境自检")
    ap.add_argument("--model-set", default="paper", help="paper | paper-70b | scaling")
    ap.add_argument("--smoke", action="store_true", help="额外做一次 GPU 前向")
    ap.add_argument("--proxy-port", type=int, default=7890)
    ap.add_argument("--require-proxy", action="store_true",
                    help="把代理不可用视为错误（需要联网下载时用）")
    args = ap.parse_args()

    print("=" * 66)
    print("DuoDecoding 环境自检")
    print("=" * 66)

    total_ok = total_bad = 0
    for title, fn in [
        ("Python 与依赖", lambda r: (check_python(r), check_packages(r))),
        ("GPU", lambda r: check_gpu(r, args.smoke)),
        ("资产（模型/检查点/数据）", lambda r: check_assets(r, args.model_set)),
        ("网络", lambda r: check_network(r, args)),
    ]:
        rep = Report()
        fn(rep)
        ok, bad = rep.dump(title)
        total_ok += ok
        total_bad += bad

    print("\n" + "=" * 66)
    if total_bad == 0:
        print(f"全部通过（{total_ok} 项）—— 可以开始跑实验了。")
        print("示例：bash scripts/run_baseline_table.sh 0")
    else:
        print(f"通过 {total_ok} 项，**失败 {total_bad} 项** —— 见上面的 ✗ 与修复建议。")
        print("一键补齐外部资产：python scripts/setup/download_assets.py --only paper")
    print("=" * 66)
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
