#!/usr/bin/env python3
"""按清单下载迁移所需的全部外部资产（模型 / acc_head 检查点 / 数据集）。

设计原则
--------
* **幂等**：已下好的资产默认跳过（可用 --force 重下）；中断后重跑即续传。
* **不硬编码任何密钥**：token 一律从环境变量 HF_TOKEN 读取。
* **镜像/代理可选**：国内环境常用 `--hf-endpoint https://hf-mirror.com` 或走本地代理；
  但 **gated 模型（meta-llama/*）只能从官方站点拉**，镜像不支持。
* **路径与代码一致**：模型落到 assets.json 里的 local 路径（llama/<name>），
  与 src/model_loading.py 的解析方式对齐，装完即可直接跑。

用法
----
    python scripts/setup/download_assets.py --only paper          # 论文最小集 ~52GB
    python scripts/setup/download_assets.py --only all            # 全部（约 770GB）
    python scripts/setup/download_assets.py --only paper --hf-endpoint https://hf-mirror.com
    python scripts/setup/download_assets.py --list                # 只看清单
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import time
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

# 判定"模型已下好"的必要文件：config.json + 至少一个权重分片
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def probe(host: str, port: int, timeout: float = 0.6) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def setup_network(args) -> None:
    """配置 HF 端点与代理（在 import huggingface_hub 之前设置才生效）。"""
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        log(f"HF 端点: {args.hf_endpoint}")
    if args.proxy == "none":
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(k, None)
        log("代理: 已禁用")
    else:
        url = None
        if args.proxy == "auto":
            # 本机常见的 clash 端口；探测到才启用，避免把直连也拖进代理
            for port in (7890, 7891, 7897):
                if probe("127.0.0.1", port):
                    url = f"http://127.0.0.1:{port}"
                    break
        else:
            url = args.proxy
        if url:
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                os.environ[k] = url
            log(f"代理: {url}")
        else:
            log("代理: 未探测到本地代理，将直连")


def model_complete(local: Path) -> bool:
    if not (local / "config.json").exists():
        return False
    return any(p.suffix in WEIGHT_SUFFIXES for p in local.rglob("*") if p.is_file())


def download_models(names: list[str], args) -> list[tuple[str, str]]:
    from huggingface_hub import snapshot_download

    results = []
    for name in names:
        spec = ASSETS["models"][name]
        local = ROOT / spec["local"]
        if model_complete(local) and not args.force:
            log(f"跳过（已存在）: {name}  ({spec['local']})")
            results.append((name, "已存在"))
            continue
        if spec.get("gated") and not os.environ.get("HF_TOKEN"):
            log(f"✗ 跳过 {name}: 该模型需要 HF_TOKEN（且在 HF 页面接受许可）—— {spec.get('note','')}")
            results.append((name, "缺 HF_TOKEN"))
            continue
        log(f"下载模型: {name}  ({spec['repo']} → {spec['local']}, 约 {spec['approx']})")
        local.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=spec["repo"],
                local_dir=str(local),
                token=os.environ.get("HF_TOKEN"),
                max_workers=args.workers,
                # 不拉 .bin/.pth/gguf 之外的历史格式，省流量；需要时去掉 ignore
                ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            )
            results.append((name, "✓"))
        except Exception as exc:  # noqa: BLE001 - 网络/许可问题都归为可重试失败
            log(f"✗ 失败 {name}: {type(exc).__name__}: {str(exc)[:160]}")
            results.append((name, f"失败({type(exc).__name__})"))
    return results


def download_acc_heads(args) -> list[tuple[str, str]]:
    """按 acc_head_registry.json 下载投机接受率预测头（方法必需，且不在 git 里）。"""
    from huggingface_hub import snapshot_download

    registry_path = ROOT / ASSETS["acc_head_registry"]
    if not registry_path.exists():
        log(f"✗ 找不到注册表 {ASSETS['acc_head_registry']}")
        return []
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    # 默认只下论文必需的那几对（assets.json 的 acc_head_required），
    # 想全下（12 对，约 600MB）用 --all-acc-heads。
    if args.acc_head_pairs:
        wanted = {p.strip() for p in args.acc_head_pairs if p.strip()}
    elif args.all_acc_heads:
        wanted = None
    else:
        wanted = {p.split("/")[0] for p in ASSETS["acc_head_required"]}

    results = []
    for entry in registry:
        local = ROOT / entry["local_path"]
        key = f"{entry['source']}--to--{entry['target']}"
        if wanted and key not in wanted:
            continue
        if model_complete(local) and not args.force:
            results.append((key, "已存在"))
            continue
        sub = entry.get("hf_subpath") or key
        log(f"下载 acc_head: {key}")
        local.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                snapshot_download(
                    repo_id=entry["hf_repo"],
                    local_dir=tmp,
                    allow_patterns=[f"{sub}/*"],
                    token=os.environ.get("HF_TOKEN"),
                    max_workers=args.workers,
                )
                src = Path(tmp) / sub
                if not src.exists():
                    raise FileNotFoundError(f"远端缺少子目录 {sub}")
                for item in src.iterdir():
                    dst = local / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dst)
            results.append((key, "✓"))
        except Exception as exc:  # noqa: BLE001
            log(f"✗ 失败 {key}: {type(exc).__name__}: {str(exc)[:160]}")
            results.append((key, f"失败({type(exc).__name__})"))
    return results


def download_datasets(args) -> list[tuple[str, str]]:
    """预热 HF datasets 缓存。评测脚本用 load_dataset()，缓存好即可离线跑。"""
    from datasets import load_dataset

    results = []
    for spec in ASSETS["datasets"]:
        name, config = spec["name"], spec.get("config")
        log(f"下载数据集: {name}{'/' + config if config else ''} ({spec['split']})")
        try:
            if config:
                load_dataset(name, config, split=spec["split"])
            else:
                load_dataset(name, split=spec["split"])
            results.append((name, "✓"))
        except Exception as exc:  # noqa: BLE001
            log(f"✗ 失败 {name}: {type(exc).__name__}: {str(exc)[:160]}")
            results.append((name, f"失败({type(exc).__name__})"))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 DuoDecoding 迁移所需资产")
    ap.add_argument("--only", default="paper",
                    help="paper | paper-70b | scaling | all | 逗号分隔的模型名")
    ap.add_argument("--proxy", default="auto",
                    help="auto | none | http://host:port")
    ap.add_argument("--hf-endpoint", default=None,
                    help="例如 https://hf-mirror.com（gated 模型不支持镜像）")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-datasets", action="store_true")
    ap.add_argument("--skip-accheads", action="store_true")
    ap.add_argument("--acc-head-pairs", nargs="*", default=None,
                    help="只下这几对 acc_head（默认只下 assets.json 里论文必需的那几对）")
    ap.add_argument("--all-acc-heads", action="store_true",
                    help="下载注册表里全部 12 对 acc_head（约 600MB）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="已存在的也重下")
    ap.add_argument("--list", action="store_true", help="只打印清单后退出")
    args = ap.parse_args()

    if args.list:
        print("可用集合:")
        for key, s in ASSETS["sets"].items():
            print(f"  {key:12} {s['desc']}")
            for m in s["models"]:
                spec = ASSETS["models"][m]
                print(f"      - {m:18} {spec['approx']:>6}  {spec['repo']}")
        return 0

    # 解析 --only
    if args.only == "all":
        names = list(ASSETS["models"])
    elif args.only in ASSETS["sets"]:
        names = list(ASSETS["sets"][args.only]["models"])
    else:
        names = [n.strip() for n in args.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in ASSETS["models"]]
    if unknown:
        log(f"✗ 未知模型名: {unknown}（用 --list 查看）")
        return 2

    setup_network(args)
    log(f"目标模型: {names}")

    summary: dict[str, list[tuple[str, str]]] = {}
    if not args.skip_models:
        summary["模型"] = download_models(names, args)
    if not args.skip_accheads:
        summary["acc_head 检查点"] = download_acc_heads(args)
    if not args.skip_datasets:
        summary["数据集"] = download_datasets(args)

    print("\n" + "=" * 66)
    ok = fail = 0
    for section, rows in summary.items():
        print(f"{section}:")
        for name, status in rows:
            print(f"  {'✓' if status in ('✓', '已存在') else '✗'} {name:34} {status}")
            ok += status in ("✓", "已存在")
            fail += status not in ("✓", "已存在")
    print("=" * 66)
    print(f"完成 {ok} 项，失败 {fail} 项")
    if fail:
        print("提示：gated 模型需要 HF_TOKEN 并在 HF 页面接受许可；网络问题可直接重跑（幂等续传）。")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
