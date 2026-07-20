import pickle
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.rl_adapter import DDQNAgent, RLNetworkAdapter


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


def _opportunistic_args(checkpoint_root: Path) -> Namespace:
    return Namespace(
        eval_mode="cee_sd_opportunistic",
        use_rl_adapter=True,
        little_model="llama-68m",
        draft_model="tiny-llama-1.1b",
        target_model="llama-2-13b",
        main_rl_path=None,
        main_rl_best_path=None,
        little_rl_path=None,
        little_rl_best_path=None,
        rl_checkpoint_root=str(checkpoint_root),
        rl_init_seed=73,
        rl_epsilon_decay=None,
        rl_reward_scale=None,
        rl_batch_size=None,
    )


def test_ddqn_initialization_is_reproducible_for_same_seed():
    first = DDQNAgent(feature_dim=10, action_dim=4, device="cpu", init_seed=73)
    second = DDQNAgent(feature_dim=10, action_dim=4, device="cpu", init_seed=73)

    first_state = first.policy_net.state_dict()
    second_state = second.policy_net.state_dict()

    assert first_state.keys() == second_state.keys()
    assert all(torch.equal(first_state[key], second_state[key]) for key in first_state)


def test_opportunistic_mode_uses_isolated_root_and_threshold_only_little_agent(
    tmp_path,
):
    args = _opportunistic_args(tmp_path / "cee_sd_opportunistic")

    with (
        patch("src.baselines.Decoding.__init__", return_value=None),
        patch("src.baselines.RLNetworkAdapter") as adapter,
        patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
    ):
        _TestBaselines(args)

    main_call, little_call = adapter.call_args_list
    assert main_call.kwargs["model_path"].startswith(args.rl_checkpoint_root)
    assert little_call.kwargs["model_path"].startswith(args.rl_checkpoint_root)
    assert main_call.kwargs["init_seed"] == args.rl_init_seed
    assert little_call.kwargs["init_seed"] == args.rl_init_seed + 1
    assert little_call.kwargs["k_candidates"] == [1]


def test_prefer_latest_is_reserved_for_trainable_opportunistic_little(tmp_path):
    # Given: opportunistic training, evaluation, and standard RL arguments.
    opportunistic_training = _opportunistic_args(tmp_path / "opportunistic-training")
    opportunistic_training.disable_rl_update = False
    opportunistic_evaluation = _opportunistic_args(
        tmp_path / "opportunistic-evaluation"
    )
    opportunistic_evaluation.disable_rl_update = True
    standard = _opportunistic_args(tmp_path / "standard")
    standard.eval_mode = "cee_sd"

    # When: Baselines constructs adapters for each training policy.
    calls_by_args = []
    for args in (opportunistic_training, opportunistic_evaluation, standard):
        with (
            patch("src.baselines.Decoding.__init__", return_value=None),
            patch("src.baselines.RLNetworkAdapter") as adapter,
            patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
        ):
            _TestBaselines(args)
            calls_by_args.append(adapter.call_args_list)

    # Then: only trainable opportunistic Little prefers the latest checkpoint.
    training_main, training_little = calls_by_args[0]
    evaluation_main, evaluation_little = calls_by_args[1]
    standard_main, standard_little = calls_by_args[2]
    assert training_main.kwargs["prefer_latest"] is False
    assert training_little.kwargs["prefer_latest"] is True
    assert evaluation_main.kwargs["prefer_latest"] is False
    assert evaluation_little.kwargs["prefer_latest"] is False
    assert standard_main.kwargs["prefer_latest"] is False
    assert standard_little.kwargs["prefer_latest"] is False


def test_fresh_initialization_refuses_to_overwrite_existing_checkpoint(tmp_path):
    checkpoint = tmp_path / "latest.pth"
    checkpoint.write_bytes(b"existing")

    try:
        RLNetworkAdapter(
            Namespace(),
            model_path=str(checkpoint),
            best_model_path=str(tmp_path / "best.pth"),
            device="cpu",
            init_seed=73,
            init_strategy="fresh",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("fresh initialization overwrote an existing checkpoint")

    assert checkpoint.read_bytes() == b"existing"


def test_resume_restores_best_tps_and_replay_buffer(tmp_path):
    checkpoint = tmp_path / "latest.pth"
    first = RLNetworkAdapter(
        Namespace(),
        model_path=str(checkpoint),
        best_model_path=str(checkpoint),
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
    )
    first.best_tps = 2.5
    first.agent.store_transition([[0.0] * 10] * 8, 0, 1.0, [[0.0] * 10] * 8, False)
    first.save(2.5)

    resumed = RLNetworkAdapter(
        Namespace(),
        model_path=str(checkpoint),
        best_model_path=str(checkpoint),
        device="cpu",
        init_seed=73,
        init_strategy="resume",
    )

    assert resumed.best_tps == 2.5
    assert len(resumed.agent.memory) == 1
    with open(str(checkpoint) + ".buffer", "rb") as buffer_file:
        assert len(pickle.load(buffer_file)) == 1


def test_resume_prefers_latest_only_when_requested(tmp_path):
    # Given: a best checkpoint followed by a newer latest checkpoint and replay.
    latest = tmp_path / "latest.pth"
    best = tmp_path / "best.pth"
    first = RLNetworkAdapter(
        Namespace(),
        model_path=str(latest),
        best_model_path=str(best),
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
        batch_size=1,
    )
    state = [[0.0] * 10] * 8
    first.agent.store_transition(state, 0, 1.0, state, True)
    first.agent.update()
    first.save(2.0)
    first.agent.store_transition(state, 1, 2.0, state, True)
    first.agent.update()
    first.save(1.0)

    # When: evaluation resumes normally and training explicitly prefers latest.
    evaluation = RLNetworkAdapter(
        Namespace(),
        model_path=str(latest),
        best_model_path=str(best),
        device="cpu",
        init_seed=73,
        init_strategy="resume",
    )
    training = RLNetworkAdapter(
        Namespace(),
        model_path=str(latest),
        best_model_path=str(best),
        device="cpu",
        init_seed=73,
        init_strategy="resume",
        prefer_latest=True,
    )

    # Then: evaluation remains best-first while training restores the newer state.
    assert evaluation.agent.update_count == 1
    assert evaluation.agent.epsilon > training.agent.epsilon
    assert len(evaluation.agent.memory) == 1
    assert training.agent.update_count == 2
    assert len(training.agent.memory) == 2


def test_save_flushes_one_pending_terminal_transition_and_persists_it(tmp_path):
    # Given: a trainable CPU adapter with one pending state, action, and reward.
    latest = tmp_path / "latest.pth"
    adapter = RLNetworkAdapter(
        Namespace(),
        model_path=str(latest),
        best_model_path=str(tmp_path / "best.pth"),
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
        batch_size=1,
    )
    adapter.select_config(
        bandwidth_mbps=10.0,
        latency_ms=10.0,
        acc_probs=[0.5],
        entropy=1.0,
        training=False,
    )
    pending_state = adapter.last_state_seq
    pending_action = adapter.last_action
    adapter.step(2.0)

    # When: the adapter saves twice at the terminal boundary.
    adapter.save(1.0)
    adapter.save(1.0)

    # Then: exactly one terminal transition updates, clears, and persists.
    assert adapter.agent.update_count == 1
    assert len(adapter.agent.memory) == 1
    transition = adapter.agent.memory[0]
    assert transition[0] is pending_state
    assert transition[1:] == (pending_action, 2.0, pending_state, True)
    assert adapter.last_state_seq is None
    assert adapter.last_action is None
    assert adapter.last_reward is None
    with open(str(latest) + ".buffer", "rb") as buffer_file:
        persisted = pickle.load(buffer_file)
    assert len(persisted) == 1
    assert persisted[0][1:] == (pending_action, 2.0, persisted[0][0], True)


def test_frozen_evaluation_does_not_save_rl_checkpoints():
    class _UnexpectedSave:
        def save(self, throughput):
            raise AssertionError("frozen evaluation must not write checkpoints")

    instance = object.__new__(_TestBaselines)
    instance.args = Namespace(disable_rl_update=True)
    instance.rl_adapter = _UnexpectedSave()
    instance.little_rl_adapter = _UnexpectedSave()

    instance._save_adaptive_rl_checkpoints(1.5)
