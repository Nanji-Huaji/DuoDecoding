"""带宽轨迹 IO（D3：从 utils 拆分，两函数重复块单点化）。

轨迹文件格式：`###...###` 分隔的块，每块一行 `Run <id>` + 一行逗号分隔
的带宽序列（Mbps）。
"""

# 带宽下限：轨迹清洗口径（尾部 <5.0 弹除、其余 clamp ≥5.0）。
# 与 --min_bandwidth_mbps 参数同量纲，历史实现即硬编码 5.0。
BANDWIDTH_FLOOR_MBPS = 5.0


def _parse_runs(trace_file: str) -> dict[int, str]:
    """解析轨迹文件为 {run_id: data_line}（原两函数 ~30 行逐字重复的块）"""
    with open(trace_file, "r") as f:
        content = f.read()

    runs: dict[int, str] = {}
    for block in content.split("###############################"):
        block = block.strip()
        if not block:
            continue

        run_id, data_line = -1, ""
        for line in block.split("\n"):
            line = line.strip()
            if line.startswith("Run"):
                try:
                    run_id = int(line.split()[1])
                except (ValueError, IndexError):
                    # 原实现为裸 except；收窄为可预期异常（KeyboardInterrupt
                    # 等不再被吞）
                    pass
            elif line:
                data_line = line

        if run_id != -1 and data_line:
            runs[run_id] = data_line
    return runs


def _clean_trace_values(data: list[float]) -> list[float]:
    """历史口径：先弹掉尾部 < 下限的值，再对剩余整体 clamp 到下限"""
    while data and data[-1] < BANDWIDTH_FLOOR_MBPS:
        data.pop()
    return [max(BANDWIDTH_FLOOR_MBPS, x) for x in data]


def read_trace_file(trace_file: str, read_idx: int = 1) -> list:
    runs = _parse_runs(trace_file)
    if read_idx in runs:
        return _clean_trace_values(
            [float(x) for x in runs[read_idx].split(",")]
        )
    raise ValueError(f"Run ID {read_idx} not found in trace file.")


def return_closest_mean_index(trace_file: str, mean_value: float | None = None) -> int:
    """返回均值最接近目标值的 Run ID；mean_value 为 None 时取全体均值。"""
    run_means: dict[int, float] = {}
    for run_id, data_line in _parse_runs(trace_file).items():
        try:
            processed = _clean_trace_values(
                [float(x) for x in data_line.split(",")]
            )
            if processed:
                run_means[run_id] = sum(processed) / len(processed)
        except ValueError:
            pass

    if not run_means:
        return -1

    if mean_value is None:
        mean_value = sum(run_means.values()) / len(run_means)

    closest_run_id, min_diff = -1, float("inf")
    for run_id, r_mean in run_means.items():
        diff = abs(r_mean - mean_value)
        if diff < min_diff:
            min_diff = diff
            closest_run_id = run_id
    return closest_run_id
