import unittest
import torch
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# 设置项目根目录
project_root = "/home/tiantianyi/code/DuoDecoding"
sys.path.insert(0, project_root)

# 导入模块
try:
    from src.engine import Decoding

    print(f"Successfully imported Decoding from src.engine")
except ImportError as e:
    print(f"Failed to import: {e}")
    try:
        src_path = os.path.join(project_root, "src")
        sys.path.insert(0, src_path)
        import model_gpu
        import engine
        from engine import Decoding

        print(f"Successfully imported using module pre-loading")
    except ImportError as e2:
        import importlib.util

        model_gpu_path = os.path.join(project_root, "src", "model_gpu.py")
        spec_model_gpu = importlib.util.spec_from_file_location("model_gpu", model_gpu_path)
        model_gpu_module = importlib.util.module_from_spec(spec_model_gpu)
        sys.modules["model_gpu"] = model_gpu_module
        spec_model_gpu.loader.exec_module(model_gpu_module)

        engine_path = os.path.join(project_root, "src", "engine.py")
        spec_engine = importlib.util.spec_from_file_location("engine", engine_path)
        engine_module = importlib.util.module_from_spec(spec_engine)
        sys.modules["engine"] = engine_module
        engine_module.KVCacheModel = model_gpu_module.KVCacheModel
        spec_engine.loader.exec_module(engine_module)

        Decoding = engine_module.Decoding
        print(f"Successfully imported using importlib manual loading")


class TestCUDATimingAccuracy(unittest.TestCase):
    """专门测试CUDA计时准确性的测试类"""

    def setUp(self):
        """设置测试环境"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        self.mock_args = Mock()
        self.mock_args.eval_mode = "large"
        self.mock_args.temp = 0.0
        self.mock_args.top_k = 50
        self.mock_args.top_p = 0.9
        self.mock_args.max_tokens = 128
        self.mock_args.seed = 42

        self.mock_decoding = Mock()
        self.mock_decoding.args = self.mock_args
        self.mock_decoding.vocab_size = 32000

        self.mock_target_model = Mock()
        self.mock_target_model.device = torch.device("cuda")
        self.mock_target_model.parameters.return_value = [torch.randn(100, 100, requires_grad=True)]

        self.mock_decoding.target_model = self.mock_target_model
        self.mock_decoding.color_print = Mock()

        # 绑定真实的计时函数
        self.mock_decoding.autoregressive_sampling = Decoding.autoregressive_sampling.__get__(
            self.mock_decoding, type(self.mock_decoding)
        )

    @patch("src.engine.KVCacheModel")
    def test_cuda_timing_accuracy(self, mock_kv_cache):
        """测试CUDA计时的准确性"""
        mock_model_instance = Mock()
        mock_kv_cache.return_value = mock_model_instance

        initial_tokens = torch.randint(0, 1000, (1, 3))

        # 🔧 适合128 tokens的延迟设置
        prefill_delay = 0.1  # 100ms - 不应计入decode_time
        decode_delay = 0.02  # 20ms per token - 应计入decode_time

        call_count = 0

        def mock_generate_with_known_delay(x, num_tokens):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                print(f"  Prefill phase: sleeping {prefill_delay}s")
                time.sleep(prefill_delay)
            else:
                print(f"  Decode phase {call_count-1}: sleeping {decode_delay}s")
                time.sleep(decode_delay)

            device = x.device
            new_tokens = torch.randint(0, 1000, (x.shape[0], num_tokens), device=device)
            return torch.cat([x, new_tokens], dim=1)

        mock_model_instance.generate = mock_generate_with_known_delay
        mock_model_instance.vocab_size = 32000

        print(f"\n🔄 Starting CUDA timing test with {self.mock_args.max_tokens} tokens...")
        start_time = time.time()
        result, decode_time = self.mock_decoding.autoregressive_sampling(initial_tokens, enable_timing=True)
        total_time = time.time() - start_time

        # 计算期望的decode时间
        expected_decode_time = decode_delay * (self.mock_args.max_tokens - 1)

        print(f"\n📊 CUDA Timing Results:")
        print(f"Total execution time: {total_time:.4f}s")
        print(f"Measured decode time: {decode_time:.4f}s")
        print(f"Expected decode time: {expected_decode_time:.4f}s")
        print(f"Prefill delay (excluded): {prefill_delay:.4f}s")
        print(f"Timing accuracy: {(decode_time/expected_decode_time)*100:.1f}%")

        # 关键验证：decode_time应该排除prefill时间
        self.assertLess(decode_time, total_time, "Decode time should be less than total time")

        # 验证计时精度：允许±5%误差（更严格，因为延迟时间更短）
        tolerance = 0.05
        lower_bound = expected_decode_time * (1 - tolerance)
        upper_bound = expected_decode_time * (1 + tolerance)

        self.assertGreater(decode_time, lower_bound, f"Decode time {decode_time:.4f}s should be ≥ {lower_bound:.4f}s")
        self.assertLess(decode_time, upper_bound, f"Decode time {decode_time:.4f}s should be ≤ {upper_bound:.4f}s")

        print("✅ CUDA timing accuracy test passed!")

    @patch("src.engine.KVCacheModel")
    def test_cuda_prefill_decode_separation(self, mock_kv_cache):
        """测试CUDA环境下prefill和decode阶段的正确分离"""
        mock_model_instance = Mock()
        mock_kv_cache.return_value = mock_model_instance

        initial_tokens = torch.randint(0, 1000, (1, 3))

        # 🔧 适合128 tokens的延迟设置
        prefill_delay = 0.05  # 50ms
        decode_delay = 0.01  # 10ms per token

        call_count = 0
        phase_times = []

        def mock_generate_with_phase_tracking(x, num_tokens):
            nonlocal call_count
            call_count += 1

            phase_start = time.time()

            if call_count == 1:
                print(f"  Prefill: {prefill_delay}s delay")
                time.sleep(prefill_delay)
                phase_times.append(("prefill", time.time() - phase_start))
            else:
                if call_count <= 10:  # 只打印前几个，避免输出过多
                    print(f"  Decode {call_count-1}: {decode_delay}s delay")
                elif call_count == 11:
                    print(f"  ... continuing decode phases ...")
                time.sleep(decode_delay)
                phase_times.append(("decode", time.time() - phase_start))

            device = x.device
            new_tokens = torch.randint(0, 1000, (x.shape[0], num_tokens), device=device)
            return torch.cat([x, new_tokens], dim=1)

        mock_model_instance.generate = mock_generate_with_phase_tracking
        mock_model_instance.vocab_size = 32000

        print(f"\n🔄 Testing prefill/decode separation with {self.mock_args.max_tokens} tokens...")
        result, decode_time = self.mock_decoding.autoregressive_sampling(initial_tokens, enable_timing=True)

        # 分析阶段时间
        prefill_phases = [t for phase, t in phase_times if phase == "prefill"]
        decode_phases = [t for phase, t in phase_times if phase == "decode"]

        total_prefill_time = sum(prefill_phases)
        total_decode_time_actual = sum(decode_phases)

        print(f"\n📊 Phase Separation Results:")
        print(f"Prefill phases: {len(prefill_phases)} (total: {total_prefill_time:.4f}s)")
        print(f"Decode phases: {len(decode_phases)} (total: {total_decode_time_actual:.4f}s)")
        print(f"Measured decode_time: {decode_time:.4f}s")
        print(f"Accuracy ratio: {decode_time/total_decode_time_actual:.4f}")

        # 🔧 修正验证逻辑：
        # 1. decode_time应该接近实际decode阶段总时间
        self.assertAlmostEqual(
            decode_time, total_decode_time_actual, delta=0.2, msg="decode_time should match actual decode phases time"
        )

        # 2. decode_time应该远小于包含prefill的总时间
        total_all_phases = total_prefill_time + total_decode_time_actual
        self.assertLess(
            decode_time,
            total_all_phases,
            f"decode_time ({decode_time:.4f}s) should be much less than total time ({total_all_phases:.4f}s)",
        )

        # 3. decode_time不应该接近prefill时间（说明没有错误地包含prefill）
        self.assertGreater(
            decode_time,
            total_prefill_time * 10,
            f"decode_time ({decode_time:.4f}s) should be much larger than prefill time ({total_prefill_time:.4f}s)",
        )

        print("✅ Prefill/decode separation test passed!")
        print(
            f"✨ Your timing function is working perfectly! Accuracy: {(decode_time/total_decode_time_actual)*100:.2f}%"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
