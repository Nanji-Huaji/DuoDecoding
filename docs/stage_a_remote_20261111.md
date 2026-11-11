# 阶段 A 远端验收记录（2026-11-11）

## 当前结果

**GPU 实验尚未执行。** 五次限时窗口都没有满足空闲条件的卡，累计等待约
181 分钟（下列均为 2026-11-11 新加坡时间）：

| 窗口 | 时间 | 实际等待 | 已运行实验 |
|---|---|---|---|
| a1 | 04:29:08–05:29:22 | 3613.9 秒 | 0/22 |
| a2 | 06:11:31–06:41:38 | 1806.6 秒 | 0/22 |
| a3 | 07:09:55–07:40:02 | 1806.9 秒 | 0/22 |
| a4 | 08:21:17–08:51:24 | 1806.9 秒 | 0/22 |
| a5 | 09:25:08–09:55:15 | 1807.4 秒 | 0/22 |

五次返回码 3 均表示等待期结束，不是算法测试失败。没有停止、重置或占用
他人的 GPU 进程；五次都使用固定阶段 A 源码，没有切到后续数值修复版本。
第五轮 60 次占用观察均无满足空闲条件的设备，未启动任何模型进程。
总等待时间为 10841.6695 秒。

原始记录已收集到本地忽略目录：[首轮](<../.refactor/stage-a-gpu-20261111/runs-a1/summary.json>)、
[第二轮](<../.refactor/stage-a-gpu-20261111/runs-a2/summary.json>)、
[第三轮](<../.refactor/stage-a-gpu-20261111/runs-a3/summary.json>)、
[第四轮](<../.refactor/stage-a-gpu-20261111/runs-a4/summary.json>)、
[第五轮](<../.refactor/stage-a-gpu-20261111/runs-a5/summary.json>)；
它们不是已提交的 GPU 验收工件。

## 已准备的实验

- 主机：`tiantianyi@10.160.8.194`，4 张 RTX A6000（每张约 48 GiB）。
- 独立源码副本：`/home/tiantianyi/code/DuoDecoding-stage-a-20261111-58034cd`，
  固定提交 `58034cd7ce8063ca1046c41c79fcb6c1456e07ef`（PR #31 合并版本）。
- 已核对本机已有的 Qwen2.5 1.5B/7B、Llama 68M/1.1B/13B、所需 ARP/RL
  权重及 tokenizer 配对，清单无缺项；未下载模型或改写权重。
- 22 个配置：SD/DSD/DSSD × eager/graph × 温度 0/.7，共 12 个；
  tridecoding、adaptive_decoding、adaptive_tridecoding、cee_sd_no_arp、
  cee_sd_opportunistic × eager/graph，共 10 个。
- 每个配置 2 个样本、最多 32 个新 token，seed `20261111`，cache 容量 512；
  ARP 配置均显式使用 `cumulative`。使用单个确认空闲的 GPU UUID。

等待器要求连续两次观察均无计算进程、显存占用低于 256 MiB、利用率为零；
启动时还经过 smoke 入口的设备和资产检查。有剩余显存但正在运行他人进程的卡
不算空闲。队列有等待期限和单案例超时，首次案例失败即停，不并发抢卡。

## CPU 与安装证据

首次干净副本预检的两种导入模式各有 18 项资源/路径相关失败，原结果保留。
随后通过 [PR #33](https://github.com/Nanji-Huaji/DuoDecoding/pull/33) 修复测试
可移植性，并在另一个干净副本和新建依赖环境取得双模式各 1386 passed、
21 skipped、163 subtests；本机完整权重严格模式各 1400 passed、7 skipped、
163 subtests。详见[可移植性与隔离安装记录](<cpu_portability_20261111.md>)。
原有 Python 环境未重装，固定 GPU 副本没有切换到后续 CUHLM 代码。

下一次取得空闲设备后可继续使用已准备的配置；仍须重新检查占用和资源指纹。
当前不能据此声称真实模型接线、CUDA capture/replay 或跨设备验收通过。
