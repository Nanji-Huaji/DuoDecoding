"""运行时 checkpoint provenance：manifest 校验 + 记录。

- 生产路径（`checkpoints/rl/...`、`src/SpecDec_pp/checkpoints/acc_head/...`）
  必须在 `checkpoints/manifest.json` 里有条目且 sha256 一致，否则 fail closed；
- staging / 临时路径 / 旧 clone 没有条目时标 `source="untracked"`，不阻断；
- 校验结果挂到 ``args._checkpoint_provenance``，由 eval 层写入 metrics。
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_REL = Path("checkpoints") / "manifest.json"


class CheckpointIntegrityError(RuntimeError):
    """checkpoint 与 manifest 的 sha256 不一致。"""


_SHA_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path) -> str:
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _SHA_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA_CACHE[key] = digest
    return digest


@lru_cache(maxsize=8)
def _manifest(manifest_path: str) -> dict:
    path = Path(manifest_path)
    if not path.is_file():
        return {"entries": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def describe_path(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """校验单个文件；返回可写入 metrics 的 provenance 字典。"""
    p = Path(path)
    rel = _rel(p, root)
    if not p.is_file():
        return {"path": rel or str(path), "source": "missing"}
    digest = sha256_file(p)
    entry = _manifest(str(root / MANIFEST_REL))["entries"].get(rel or "")
    if entry is None:
        return {
            "path": rel or str(path),
            "sha256": digest,
            "source": "untracked",
        }
    if digest != entry["sha256"]:
        raise CheckpointIntegrityError(
            f"{rel}: sha256 不匹配（磁盘 {digest[:16]}… != manifest "
            f"{entry['sha256'][:16]}…）。请运行 `python tools/fetch_checkpoints.py` "
            "或重新晋升该 checkpoint。"
        )
    return {
        "path": rel,
        "sha256": digest,
        "role": entry.get("role"),
        "pair": entry.get("pair"),
        "source": "manifest",
    }


def describe_dir(path: str | Path, *, root: Path = ROOT) -> dict[str, Any]:
    """校验目录内所有文件（acc head 是目录）；任一文件 sha 不符即失败。"""
    p = Path(path)
    rel = _rel(p, root)
    if not p.is_dir():
        return {"dir": rel or str(path), "source": "missing"}
    files: dict[str, dict] = {}
    for f in sorted(p.rglob("*")):
        if not f.is_file() or ".cache" in f.parts:
            continue
        files[str(f.relative_to(p))] = describe_path(f, root=root)
    source = (
        "manifest"
        if any(v.get("source") == "manifest" for v in files.values())
        else "untracked"
    )
    return {"dir": rel or str(path), "source": source, "files": files}


def record(args: Any, key: str, info: dict) -> None:
    """把 provenance 挂到 args，供 eval 层写入 metrics。"""
    bucket = getattr(args, "_checkpoint_provenance", None)
    if bucket is None:
        bucket = {}
        setattr(args, "_checkpoint_provenance", bucket)
    bucket[key] = info


def summary_line(key: str, info: dict) -> str:
    if info.get("source") == "missing":
        return f"[checkpoint] {key}: missing"
    if info.get("source") == "untracked":
        return f"[checkpoint] {key}: {info.get('path') or info.get('dir')} (untracked)"
    sha = info.get("sha256") or next(
        (v.get("sha256") for v in (info.get("files") or {}).values()), ""
    )
    return f"[checkpoint] {key}: {info.get('path') or info.get('dir')} sha={sha[:16]}… (manifest)"
