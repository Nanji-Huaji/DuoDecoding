import json
import time
from pathlib import Path

import torch


class AdaptiveDecodingDebugger:
    def __init__(self, log_path: str | None, *, enabled: bool = False):
        self.enabled = enabled and bool(log_path)
        self.log_path = Path(log_path) if log_path else None
        if self.enabled and self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict) -> None:
        if not self.enabled or self.log_path is None:
            return
        record = {"ts": time.time(), **payload}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def event(self, label: str, **fields) -> None:
        self._write({"type": "event", "label": label, **fields})

    def tensor(self, label: str, tensor: torch.Tensor) -> None:
        if not self.enabled:
            return
        record = {
            "type": "tensor",
            "label": label,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
        }
        if tensor.numel() > 0:
            values = tensor.detach()
            if values.dtype != torch.long:
                values = values.to(torch.long)
            flat = values.reshape(-1)
            record["min"] = int(flat.min().item())
            record["max"] = int(flat.max().item())
            record["head"] = flat[:16].tolist()
        self._write(record)

    def invalid_tokens(
        self,
        label: str,
        tensor: torch.Tensor,
        *,
        vocab_size: int,
    ) -> None:
        if not self.enabled:
            return
        values = tensor.detach()
        if values.dtype != torch.long:
            values = values.to(torch.long)
        flat = values.reshape(-1)
        self._write(
            {
                "type": "invalid_tokens",
                "label": label,
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "device": str(values.device),
                "vocab_size": vocab_size,
                "min": int(flat.min().item()) if flat.numel() else None,
                "max": int(flat.max().item()) if flat.numel() else None,
                "head": flat[:16].tolist(),
            }
        )
