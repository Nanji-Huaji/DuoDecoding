import math
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

import src.baselines as baselines
import src.utils as utils
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


@pytest.mark.parametrize(
    ("pilot_flags", "expected_values"),
    [
        ([], (None, None, None)),
        (
            [
                "--rl_epsilon_decay",
                "0.8",
                "--rl_reward_scale",
                "0.25",
                "--rl_batch_size",
                "7",
            ],
            (0.8, 0.25, 7),
        ),
    ],
)
def test_parse_arguments_exposes_rl_pilot_controls(
    monkeypatch, tmp_path, pilot_flags, expected_values
):
    # Given: an isolated CLI invocation without model loading or path resolution.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utils, "model_zoo", lambda args: None)
    monkeypatch.setattr(utils, "resolve_acc_head_path", lambda *_: "acceptance.pt")
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate.py",
            "--exp_name",
            "parser-test",
            "--acc_head_path",
            "acceptance.pt",
            "--small_draft_acc_head_path",
            "acceptance.pt",
            "--draft_target_acc_head_path",
            "acceptance.pt",
            "--main_rl_path",
            "main-latest.pt",
            "--main_rl_best_path",
            "main-best.pt",
            "--little_rl_path",
            "little-latest.pt",
            "--little_rl_best_path",
            "little-best.pt",
            *pilot_flags,
        ],
    )

    # When: the common CLI parser receives omitted or explicit pilot controls.
    args = utils.parse_arguments()

    # Then: omitted values remain None and explicit values preserve their types.
    assert (
        args.rl_epsilon_decay,
        args.rl_reward_scale,
        args.rl_batch_size,
    ) == expected_values


def test_frozen_adapter_selects_greedy_action_and_does_not_arm_transition(tmp_path):
    # Given: a frozen adapter whose greedy action differs from its exploratory action.
    adapter = RLNetworkAdapter(
        Namespace(),
        model_path=str(tmp_path / "latest.pth"),
        best_model_path=str(tmp_path / "best.pth"),
        device="cpu",
        init_seed=73,
        init_strategy="fresh",
        frozen=True,
    )
    for parameter in adapter.agent.policy_net.parameters():
        parameter.data.zero_()
    adapter.agent.policy_net.adv_fc[-1].bias.data[3] = 1.0
    adapter.agent.epsilon = 1.0

    # When: inference selects a configuration and records a stage reward.
    selected_topk, selected_threshold = adapter.select_config(
        bandwidth_mbps=10.0,
        latency_ms=10.0,
        acc_probs=[0.5],
        entropy=1.0,
        training=True,
    )
    adapter.step(7.5)
    adapter.save(1.5)

    # Then: the greedy action is used and no learning state or checkpoint is written.
    assert (selected_topk, selected_threshold) == (1, 0.6)
    assert adapter.last_state_seq is None
    assert adapter.last_action is None
    assert adapter.last_reward is None
    assert len(adapter.agent.memory) == 0
    assert not (tmp_path / "latest.pth").exists()
    assert not (tmp_path / "best.pth").exists()


def test_ddqn_agent_scales_rewards_before_updating():
    # Given: a CPU agent with a configurable reward scale and one replay transition.
    agent = DDQNAgent(
        feature_dim=10,
        action_dim=4,
        device="cpu",
        batch_size=1,
        reward_scale=0.25,
    )
    state = [[0.0] * 10] * 8
    agent.store_transition(state, 0, 8.0, state, True)
    # When: the agent consumes the transition.
    with patch.object(agent, "loss_fn", wraps=agent.loss_fn) as loss_fn:
        agent.update()

    # Then: the target uses the configured reward scale rather than a fixed constant.
    assert loss_fn.call_args.args[1].item() == pytest.approx(2.0)


def test_adapter_forwards_pilot_configuration_to_ddqn_agent(tmp_path):
    # Given: adapter-level pilot configuration and a CPU DDQN constructor seam.
    with patch("src.rl_adapter.DDQNAgent") as ddqn_agent:
        ddqn_agent.return_value.best_tps = -1.0

        # When: an adapter is initialized for a fresh CPU run.
        RLNetworkAdapter(
            Namespace(),
            model_path=str(tmp_path / "latest.pth"),
            best_model_path=str(tmp_path / "best.pth"),
            device="cpu",
            init_seed=73,
            init_strategy="fresh",
            epsilon_decay=0.8,
            reward_scale=0.25,
            batch_size=7,
        )

    # Then: the DDQN agent receives every pilot hyperparameter unchanged.
    assert ddqn_agent.call_args.kwargs["epsilon_decay"] == 0.8
    assert ddqn_agent.call_args.kwargs["reward_scale"] == 0.25
    assert ddqn_agent.call_args.kwargs["batch_size"] == 7


def test_opportunistic_mode_forwards_pilot_agent_configuration(tmp_path):
    # Given: opportunistic mode with explicit pilot configuration overrides.
    args = _opportunistic_args(tmp_path / "cee_sd_opportunistic")
    args.rl_epsilon_decay = 0.8
    args.rl_reward_scale = 0.25
    args.rl_batch_size = 7

    # When: Baselines constructs its main and Little adapters.
    with (
        patch("src.baselines.Decoding.__init__", return_value=None),
        patch("src.baselines.RLNetworkAdapter") as adapter,
        patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
    ):
        _TestBaselines(args)

    # Then: both adapters receive all pilot configuration values.
    for call in adapter.call_args_list:
        assert call.kwargs["epsilon_decay"] == 0.8
        assert call.kwargs["reward_scale"] == 0.25
        assert call.kwargs["batch_size"] == 7


def test_opportunistic_mode_uses_frozen_main_and_trainable_little_pilot_defaults(
    tmp_path,
):
    # Given: opportunistic mode without pilot overrides.
    args = _opportunistic_args(tmp_path / "cee_sd_opportunistic")

    # When: Baselines constructs its adapters.
    with (
        patch("src.baselines.Decoding.__init__", return_value=None),
        patch("src.baselines.RLNetworkAdapter") as adapter,
        patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
    ):
        _TestBaselines(args)

    main_call, little_call = adapter.call_args_list

    # Then: Main is frozen while Little trains with opportunistic pilot defaults.
    assert main_call.kwargs["frozen"] is True
    assert little_call.kwargs["frozen"] is False
    for call in (main_call, little_call):
        assert call.kwargs["epsilon_decay"] == 0.95
        assert call.kwargs["reward_scale"] == 1.0
        assert call.kwargs["batch_size"] == 16


def test_other_rl_modes_retain_historical_agent_defaults(tmp_path):
    # Given: an existing non-opportunistic RL mode without pilot overrides.
    args = _opportunistic_args(tmp_path / "cee_sd")
    args.eval_mode = "cee_sd"

    # When: Baselines constructs its adapters.
    with (
        patch("src.baselines.Decoding.__init__", return_value=None),
        patch("src.baselines.RLNetworkAdapter") as adapter,
        patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
    ):
        _TestBaselines(args)

    # Then: the pilot-specific controls are left at the adapter's historical defaults.
    for call in adapter.call_args_list:
        assert call.kwargs["frozen"] is False
        assert call.kwargs["epsilon_decay"] == 0.9995
        assert call.kwargs["reward_scale"] == 0.01
        assert call.kwargs["batch_size"] == 32


def test_stage_reward_uses_raw_tps_for_opportunistic_mode():
    # Given: a stage that has an incomplete acceptance rate.
    tps_part = 12.5

    # When: opportunistic reward shaping is requested.
    reward = baselines.compute_stage_reward(
        tps_part=tps_part,
        generated_tokens=4,
        accepted_tokens=1,
        opportunistic=True,
    )

    # Then: the incremental TPS is passed through without acceptance shaping.
    assert reward == tps_part


def test_stage_reward_preserves_historical_shaping_outside_opportunistic_mode():
    # Given: a non-opportunistic stage with partial acceptance.
    tps_part = 12.5

    # When: the historical reward path is requested.
    reward = baselines.compute_stage_reward(
        tps_part=tps_part,
        generated_tokens=4,
        accepted_tokens=1,
        opportunistic=False,
    )

    # Then: exponential TPS shaping and the squared acceptance penalty remain intact.
    assert reward == math.exp(tps_part / 20.0) * 0.25**2
