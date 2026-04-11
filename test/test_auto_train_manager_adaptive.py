import json
import os
import tempfile
import unittest
from pathlib import Path

from auto_train_manager_adaptive import TrainingManager


class AdaptiveTrainingManagerTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.tempdir.name)

    def tearDown(self):
        os.chdir(self.original_cwd)
        self.tempdir.cleanup()

    def test_main_rl_checkpoint_uses_adaptive_decoding_root(self):
        manager = TrainingManager(model_series_name="llama")

        self.assertEqual(manager.main_rl_spec.method, "adaptive_decoding")
        self.assertIn(
            "checkpoints/rl_agents/adaptive_decoding/main/",
            manager.main_rl_spec.latest_path,
        )
        self.assertEqual(
            manager.status_file,
            Path("checkpoints/rl_agents/adaptive_decoding/llama/training_status.json"),
        )

    def test_save_training_status_only_persists_main_metrics(self):
        manager = TrainingManager(model_series_name="llama")
        manager.best_tps = 1.5
        manager.tps_history = [1.0, 1.5]
        manager.loss_history_main = [0.2]
        manager.reward_history_main = [0.8]

        manager.save_training_status()

        with open(manager.status_file, "r") as f:
            status = json.load(f)

        self.assertEqual(status["loss_history_main"], [0.2])
        self.assertEqual(status["reward_history_main"], [0.8])
        self.assertNotIn("loss_history_little", status)
        self.assertNotIn("reward_history_little", status)

    def test_prepare_checkpoints_migrates_legacy_single_agent_checkpoint(self):
        manager = TrainingManager(model_series_name="llama")
        legacy_dir = Path("checkpoints/llama")
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_pth = legacy_dir / "rl_adapter.pth"
        legacy_buffer = legacy_dir / "rl_adapter.pth.buffer"
        legacy_pth.write_bytes(b"legacy-main")
        legacy_buffer.write_bytes(b"legacy-buffer")

        manager.prepare_checkpoints()

        self.assertEqual(
            Path(manager.main_rl_spec.latest_path).read_bytes(), b"legacy-main"
        )
        self.assertEqual(
            Path(manager.main_rl_spec.latest_path + ".buffer").read_bytes(),
            b"legacy-buffer",
        )

    def test_save_best_checkpoint_copies_only_main_agent_files(self):
        manager = TrainingManager(model_series_name="llama")
        main_path = Path(manager.main_rl_spec.latest_path)
        main_path.parent.mkdir(parents=True, exist_ok=True)
        main_path.write_bytes(b"main-agent")
        Path(manager.main_rl_spec.latest_path + ".buffer").write_bytes(b"main-buffer")
        manager.save_training_status()

        manager.save_best_checkpoint(2.0)

        self.assertEqual(len(manager.top_checkpoints), 1)
        _, best_dir = manager.top_checkpoints[0]
        saved_files = {path.name for path in best_dir.iterdir()}

        self.assertEqual(
            saved_files, {"latest.pth", "latest.pth.buffer", "training_status.json"}
        )


if __name__ == "__main__":
    unittest.main()
