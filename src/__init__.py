import sys
import os

# 添加gpt-fast模块路径
gpt_fast_path = os.path.join(os.path.dirname(__file__), "gpt-fast")
if gpt_fast_path not in sys.path:
    sys.path.insert(0, gpt_fast_path)

# 可选：直接导入gpt-fast的主要模块
try:
    import gpt_fast
except ImportError:
    pass  # 如果gpt-fast模块不可用，忽略错误
