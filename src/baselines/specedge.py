"""Single-request, single-chain SpecEdge (§4.1–4.2, arXiv:2505.17052).

Only IDs cross the modeled link. The two models execute serially here; the
latency model overlaps measured proactive work with the corresponding reply wait.
"""

import time

import torch

from ..dual_decoding import DualStats, truncate_confirmed_eos, validate_dual_prefix
from ..metrics import get_empty_metrics
from ..mode_features import specedge_candidate_limit
from ..register import Register
from ..sampling import sample
from .dual import acquire_dual_caches, distributed_metrics
from .simulators import _make_symmetric_comm_simulator


def _synchronize(tokens):
    if tokens.is_cuda:
        torch.cuda.synchronize(tokens.device)


def _draft(cache, prefix, count):
    if not count:
        return prefix, 0.0
    start = time.perf_counter()
    tokens = cache.generate(prefix, count)
    _synchronize(tokens)
    return tokens, time.perf_counter() - start


def _verify(cache, inputs, base, greedy):
    """One target forward; reply through the first mismatch, or through the bonus."""
    cache.forward_context(inputs)
    rows = cache.prob_history[0, base - 1 : inputs.shape[1]]
    chosen = rows.argmax(-1) if greedy else sample(rows).squeeze(-1)
    candidates = inputs[0, base:].to(chosen.device)
    mismatch = (chosen[:-1] != candidates).nonzero()
    matched = int(mismatch[0, 0]) if mismatch.numel() else candidates.numel()
    return chosen[None, : matched + 1], matched


class SpecEdgeMixin:
    @Register.register_decoding("specedge")
    @torch.inference_mode()
    def specedge(
        self,
        prefix,
        use_precise_comm_sim=False,
        use_stochastic_comm=False,
        ntt_ms_edge_cloud=200,
        ntt_ms_edge_end=20,
        use_early_stopping=False,
        stop_sequences=None,
        **kwargs,
    ):
        validate_dual_prefix(prefix, self.vocab_size)
        gamma = int(self.args.gamma)
        lookahead = specedge_candidate_limit(
            gamma, getattr(self.args, "specedge_lookahead", None)
        )
        extra = dict(
            proactive_tokens=0,
            reused_tokens=0,
            proposal_rounds=0,
            proactive_time=0.0,
            overlap_time=0.0,
        )
        wire_messages = []
        if self.args.max_tokens <= 0:
            return prefix, {**get_empty_metrics(), "specedge": extra}

        sim = _make_symmetric_comm_simulator(
            self.comm_env,
            precise=use_precise_comm_sim,
            use_stochastic_comm=use_stochastic_comm,
            ntt_ms_edge_cloud=ntt_ms_edge_cloud,
            ntt_ms_edge_end=ntt_ms_edge_end,
        )
        sim.coalesce_rounds = self.comm_config.comm_round_trip_mode == "per_round"
        draft, target = acquire_dual_caches(
            self,
            "_specedge_caches",
            self.args.top_k,
            self.args.top_k,
            self.args.top_p,
        )
        prompt_len = prefix.shape[1]
        limit = prompt_len + self.args.max_tokens
        prefix = prefix.to(draft.device)  # confirmed edge output
        draft_prefix = prefix  # may also contain next round's candidates
        server_prefix = None  # the server keeps its own confirmed IDs
        stats = DualStats()
        draft_calls = 0
        tokenizer = getattr(self, "tokenizer", None)
        delay = self.comm_env.batch_delay
        start = time.perf_counter()
        while prefix.shape[1] < limit:
            base = prefix.shape[1]
            count = min(gamma, limit - base - 1)  # leave room for a correction/bonus
            carried = draft_prefix.shape[1] - base
            sent, seconds = _draft(draft, draft_prefix, count - carried)
            stats.proposal(count, 0)
            stats.draft_seconds += seconds
            draft_calls += count - carried
            extra["reused_tokens"] += carried

            sim.set_round(stats.rounds)
            first_msg = len(sim.stats["edge_cloud"])
            uplink = sent if server_prefix is None else sent[:, base:]
            sim.simulate_transfer(uplink.nbytes, "edge_cloud", draft_len=count)
            wire_messages.append([uplink.nbytes, stats.rounds])

            # Predict without inspecting the reply. The first prediction is the
            # bonus root, not a next-round candidate. Never draft past the budget.
            ahead = min(lookahead, max(0, limit - sent.shape[1] - 2))
            steps = 1 + ahead if ahead else 0
            predicted, proactive = _draft(draft, sent, steps)
            draft_calls += steps
            stats.draft_seconds += proactive
            extra["proactive_tokens"] += steps
            extra["proactive_time"] += proactive

            t0 = time.perf_counter()
            incoming = uplink.to(target.device)
            inputs = (
                incoming
                if server_prefix is None
                else torch.cat((server_prefix, incoming), dim=1)
            )
            reply, matched = _verify(target, inputs, base, self.args.temp == 0)
            server_prefix = torch.cat(
                (inputs[:, :base], reply.to(inputs.device)), dim=1
            )
            _synchronize(server_prefix)
            verify_seconds = time.perf_counter() - t0
            stats.target_seconds += verify_seconds
            sim.simulate_transfer(reply.nbytes, "edge_cloud")
            wire_messages.append([reply.nbytes, stats.rounds])
            sim.flush_round()
            network = sum(
                msg["transfer_time"] for msg in sim.stats["edge_cloud"][first_msg:]
            )
            extra["overlap_time"] += min(proactive, network + delay + verify_seconds)

            reply = reply.to(draft.device)
            prefix = torch.cat((prefix, reply), dim=1)
            prefix, stop = truncate_confirmed_eos(
                prefix,
                base,
                getattr(tokenizer, "eos_token_id", None),
                getattr(self.args, "disable_eos_stop", False),
            )
            if use_early_stopping and stop_sequences and tokenizer is not None:
                for end in range(base + 1, prefix.shape[1] + 1):
                    # A stop such as '\nprint' can end inside the token 'printf'.
                    # Inspect only generated text, never a marker in the prompt.
                    text = tokenizer.decode(
                        prefix[0, prompt_len:end], skip_special_tokens=True
                    )
                    if any(marker in text for marker in stop_sequences):
                        prefix, stop = prefix[:, :end], True
                        break
            kept = min(matched, prefix.shape[1] - base)
            self.num_acc_tokens.append(kept)
            stats.accepted += kept
            target.rollback(base + kept)  # correction/bonus not yet forwarded
            if (
                not stop
                and steps
                and matched == count
                and torch.equal(predicted[:, sent.shape[1]], reply[:, -1])
            ):
                draft_prefix = predicted  # aligned: preserve both the suffix and its KV
            else:
                draft.rollback(min(draft.current_length, base + kept))
                draft_prefix = prefix
            if stop:
                break

        _synchronize(prefix)
        serial = time.perf_counter() - start
        metrics = distributed_metrics(
            stats,
            prefix.shape[1] - prompt_len,
            serial - extra["overlap_time"],
            delay,
            sim,
            self.args,
        )
        extra["proposal_rounds"] = stats.draft_rounds
        # Keep both message sizes even when the simulator settles one combined RTT.
        metrics.update(
            specedge=extra,
            draft_forward_times=draft_calls,
            computation_time=serial,
            comm_trace_edge_cloud=wire_messages,
        )
        if self.accelerator.is_main_process:
            self.draft_forward_times += draft_calls
            self.target_forward_times += stats.rounds
        return prefix, metrics
