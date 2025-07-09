import torch
import json
import torch.distributed as dist
import numpy as np
import os
import transformers
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
import llama_cpp
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .model_gpu import KVCacheModel
from .model_cpu import KVCacheCppModel
from .utils import seed_everything, norm_logits, sample, max_fn
import time

from transformers import StoppingCriteriaList, MaxLengthCriteria
from .model.pld.pld import greedy_search_pld

from .model.rest.rest.model.utils import *
from .model.rest.rest.model.rest_model import RestModel
from .model.rest.rest.model.kv_cache import initialize_past_key_values
import draftretriever


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            size = int(os.environ["WORLD_SIZE"])
            if "duodec" in args.eval_mode:
                dist.init_process_group("gloo", init_method="env://", rank=rank, world_size=size)
            else:
                dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=size)
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.prob_with_flag = []

    def load_model(self):
        # * load models according to different evaluation methods.
        if self.args.draft_model is None or self.args.draft_model == "":
            self.draft_model = "llama/llama-160m"
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        self.color_print(f"draft model: {self.args.draft_model}", 3)
        self.color_print(f"target model: {self.args.target_model}", 3)
        if self.args.smallest_model is not None and self.args.eval_mode == "tridec":
            self.color_print(f"smallest model: {self.args.smallest_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
        elif self.args.eval_mode == "sd":
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="balanced_low_0",
                # device_map="cuda:1",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()

        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="balanced_low_0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()

        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map=f"cuda:{torch.cuda.device_count()-1}",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()

        elif self.args.eval_mode == "duodec":
            if self.accelerator.is_main_process:
                print(
                    f"duodec mode, loading draft model {self.args.draft_model} on rank {self.accelerator.process_index}"
                )
                self.draft_model = llama_cpp.Llama(
                    model_path=self.args.draft_model,
                    n_ctx=4096,
                    verbose=False,
                    n_threads=16,
                )
            else:
                print(
                    f"duodec mode, loading target model {self.args.target_model} on rank {self.accelerator.process_index}"
                )
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()

        elif self.args.eval_mode == "pld":
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
            self.target_model.greedy_search_pld = greedy_search_pld.__get__(self.target_model, type(self.target_model))
        elif self.args.eval_mode == "lade":
            from .model.lade.utils import augment_all, config_lade
            from .model.lade.decoding import CONFIG_MAP

            if int(os.environ.get("USE_LADE", 0)):
                augment_all()
                config_lade(
                    LEVEL=self.args.level,
                    WINDOW_SIZE=self.args.window,
                    GUESS_SET_SIZE=self.args.guess,
                    DEBUG=0,
                    USE_FLASH=0,
                    DIST_WORKERS=len(os.environ.get("CUDA_VISIBLE_DEVICES").split(",")),
                )
                print("lade activated config: ", CONFIG_MAP)
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
        elif self.args.eval_mode == "rest":
            self.model = RestModel.from_pretrained(
                self.args.target_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.token_spans = list(range(2, self.args.max_token_span + 1))[::-1]
            self.datastore = draftretriever.Reader(
                index_file_path=self.args.datastore_path,
            )
        elif self.args.eval_mode == "tridec":
            self.smallest_model = AutoModelForCausalLM.from_pretrained(
                self.args.smallest_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="balanced_low_0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="balanced_low_0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()

        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model,
            trust_remote_code=True,
            cache_dir="llama/.cache/huggingface",
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"

        # for llama models
        self.tokenizer.pad_token_id = 2

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def autoregressive_sampling(self, prefix, enable_timing=True):
        if self.args.eval_mode == "small":
            model = self.draft_model
            self.color_print("--- Running in SMALL mode ---", 3)
        elif self.args.eval_mode == "large":
            model = self.target_model
            self.color_print("--- Running in LARGE mode ---", 3)
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.color_print(f"Model: {type(model).__name__}, Parameters: {num_params / 1e9:.2f}B", 3)
        #  初始化模型和 KV 缓存
        device = model.device
        prefix = prefix.to(device)
        model = KVCacheModel(model, self.args.temp, self.args.top_k, self.args.top_p)
        model.vocab_size = self.vocab_size
        #  设置计时器
        use_cuda_events = enable_timing and device.type == "cuda"

        if use_cuda_events:
            decode_start_event = torch.cuda.Event(enable_timing=True)
            decode_end_event = torch.cuda.Event(enable_timing=True)
        decode_time = 0.0
        #  生成循环
        x = prefix
        max_len = x.shape[1] + self.args.max_tokens
        #  Prefill
        x = model.generate(x, 1)
        #  Decode
        if x.shape[1] < max_len:
            if use_cuda_events:
                decode_start_event.record()
            elif enable_timing:
                decode_start_time = time.time()
            while x.shape[1] < max_len:
                x = model.generate(x, 1)
            #  结束计时
            if use_cuda_events:
                decode_end_event.record()
                torch.cuda.synchronize()
                decode_time = decode_start_event.elapsed_time(decode_end_event) / 1000.0
            elif enable_timing:
                decode_time = time.time() - decode_start_time
        else:
            # 如果只生成一个 token，decode_time 为 0
            if use_cuda_events:
                torch.cuda.synchronize()
            decode_time = 0.0

        x = x.to("cpu")
        return x, decode_time

    @torch.no_grad()
    def speculative_decoding(self, prefix, enable_timing=True):
        local_accepted_tokens = 0
        local_total_tokens = 0
        local_draft_forwards = 0
        local_target_forwards = 0
        max_tokens = prefix.shape[1] + self.args.max_tokens
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        # 计时器设置
        use_cuda_events = enable_timing and draft_device.type == "cuda" and target_device.type == "cuda"
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        prefill_time = 0.0
        decode_time = 0.0
        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size
        # 获取模型计算时使用的数据类型
        model_dtype = next(self.target_model.parameters()).dtype

        # 创建常量张量
        ONE_TENSOR = torch.tensor(1.0, device=draft_device, dtype=model_dtype)

        loop_index = 0
        while prefix.shape[1] < max_tokens:
            # 计时循环
            if use_cuda_events:
                start_event.record()
            elif enable_timing:
                loop_begin_time = time.time()

            prefix_len = prefix.shape[1]

            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)

            if self.accelerator.is_main_process:
                local_draft_forwards += self.args.gamma
                local_target_forwards += 1

            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                r = torch.rand(1, device=draft_device, dtype=model_dtype)  # 确保随机数类型也匹配
                j = x[:, prefix_len + i]

                target_prob = target_model_cache._prob_history.to(draft_device)[:, prefix_len + i - 1, j]
                approx_prob = approx_model_cache._prob_history[:, prefix_len + i - 1, j]
                # 避免除以零
                if approx_prob < 1e-9:
                    n = prefix_len + i - 1
                    break
                ratio = target_prob / approx_prob

                if r > torch.min(ONE_TENSOR, ratio):
                    n = prefix_len + i - 1
                    break

            accepted_count = n - prefix_len + 1
            self.num_acc_tokens.append(accepted_count)
            local_total_tokens += self.args.gamma
            local_accepted_tokens += accepted_count

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

            prefix = x[:, : n + 1]
            approx_model_cache.rollback(n + 1)
            if n < prefix_len + self.args.gamma - 1:
                target_dist = target_model_cache._prob_history[:, n, : self.vocab_size].to(draft_device)
                approx_dist = approx_model_cache._prob_history[:, n, : self.vocab_size]

                corrected_dist = max_fn(target_dist - approx_dist)

                if corrected_dist.sum() > 1e-6:
                    t = sample(corrected_dist)
                else:
                    t = sample(target_dist)

                target_model_cache.rollback(n + 1)
            else:
                t = sample(target_model_cache._prob_history[:, -1, : self.vocab_size]).to(draft_device)
                target_model_cache.rollback(n + 2)

            prefix = torch.cat((prefix, t), dim=1)

            # 计时结束
            if use_cuda_events:
                end_event.record()
                torch.cuda.synchronize()
                loop_duration_s = start_event.elapsed_time(end_event) / 1000.0
            elif enable_timing:
                loop_duration_s = time.time() - loop_begin_time
            if enable_timing:
                if loop_index == 0:
                    prefill_time += loop_duration_s
                else:
                    decode_time += loop_duration_s
            loop_index += 1

        prefix = prefix.to("cpu")
        return (
            prefix,
            local_accepted_tokens,
            local_total_tokens,
            local_draft_forwards,
            local_target_forwards,
            decode_time,
        )

    @torch.no_grad()
    def pld_forward(self, prefix):
        input_ids = prefix.cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        max_tokens = prefix.shape[1] + self.args.max_tokens
        output_ids, idx, accept_length_list = self.target_model.greedy_search_pld(
            input_ids,
            attention_mask=attention_mask,
            # stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=len(input_ids[0]) + max_new_tokens)]),
            stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_tokens)]),
            draft_matching_window_size=3,
            draft_num_candidate_tokens=10,
            use_cache=True,
            pad_token_id=2,
            eos_token_id=2,
            return_dict_in_generate=False,
        )
        input_len = len(input_ids[0])
        new_token = len(output_ids[0][input_len:])
        if 2 in output_ids[0, input_len:].tolist():
            for i, id in enumerate(output_ids[0, input_len:]):
                if id == 2:
                    eos_token_ids_index = i
            invalid_len = len(output_ids[0, input_len:]) - eos_token_ids_index - 1
            if invalid_len > 0:
                accept_length_list[-1] -= invalid_len
                new_token -= invalid_len
        return output_ids

    @torch.no_grad()
    def lookahead_forward(self, prefix):
        input_ids = prefix.cuda()

        output_ids, idx, accept_length_list = self.target_model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.args.max_tokens,
            num_assistant_tokens_schedule="constant",
        )
        new_token = len(output_ids[0][len(input_ids[0]) :])
        return output_ids

    @torch.no_grad()
    def rest_forward(self, prefix):
        temperature = self.args.temp
        top_p = self.args.top_p
        num_draft = self.args.num_draft
        token_spans = self.token_spans
        max_steps = 512
        datastore = self.datastore
        max_new_tokens = self.args.max_tokens
        model = self.model
        tokenizer = self.tokenizer

        input_ids = prefix.cuda()
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        accept_length_list = []

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        cur_length = input_len
        model.base_model.model.draft_mask = None
        logits = initialize_logits(input_ids, model, past_key_values)
        new_token = 0

        for idx in range(max_steps):
            candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                logits,
                input_ids,
                datastore,
                token_spans,
                top_p,
                temperature,
                max_num_draft=num_draft,
                device=model.base_model.device,
            )
            model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                draft_buffers["draft_position_ids"],
                input_ids,
                draft_buffers["retrieve_indices"],
            )
            best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, top_p)
            input_ids, logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                draft_buffers["retrieve_indices"],
                outputs,
                logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_length_list.append(accept_length_tree)
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
        return input_ids

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0
        start_time = time.time()
        while prefix.shape[1] < max_tokens:
            # if self.accelerator.is_main_process:
            #     print(f"rank 0 verify time {time.time() - start_time}")
            # else:
            #     print(f"rank 1 verify time {time.time() - start_time}")
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1 : prefix_len, : self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1 : self.args.gamma * 2] = x[
                    :, prefix_len - self.args.gamma + 1 : prefix_len + self.args.gamma
                ]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - self.args.gamma - 1 : prefix_len, : self.vocab_size]
                prob = prob.to("cuda:1")
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()
            start_time = time.time()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            # dist.barrier()
            # if self.accelerator.is_main_process:
            #     print(f"rank 0 gather time {time.time() - start_time}")
            # else:
            #     print(f"rank 1 gather time {time.time() - start_time}")
            # with self.accelerator.main_process_first():
            #     print(f"rank {self.accelerator.process_index} {all_prob.shape}")
            #     print(prob)
            draft_ids = all_prob[0, [0], 1 : self.args.gamma * 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)

                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma :]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma :]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, : prefix_len - self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)
            # if self.accelerator.is_main_process:
            #     print(prefix.shape[1])
        return prefix

    @torch.no_grad()
    def duodecoding(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheCppModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = "cpu"
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device
            stop_signal = torch.ones(1, device="cpu")

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0
        total_time = 0

        comm_tensor = torch.tensor(
            [1, 1, 15000], dtype=torch.int, device="cpu"
        )  # cure_mode, accept_len, new_sampled_token_id

        draft_prob_comm = torch.zeros(self.args.gamma + 1, device="cpu")

        prev_k = 1
        prev_size = self.args.gamma

        prefilling = True
        gamma = self.args.gamma

        while prefix.shape[1] < max_tokens:
            if prefilling:
                self.args.gamma = 1
                prev_size = 1
                prefilling = False
            else:
                self.args.gamma = gamma

            prefix_len = prefix.shape[1]

            if self.accelerator.is_main_process:  # draft model
                input_ids = prefix.tolist()[0]

                flatten_draft_k_seq_ids, draft_k_seq_prob, cur_k = model.generate_k_seq(input_ids, self.args.gamma)
                # draft_prob = model._prob_history[:, prefix_len - prev_size:prefix_len, :self.vocab_size]

                model.stop_signal[0] = 0.0

                # draft_prob_comm[0, 0, 0] = cur_k
                # draft_prob_comm[0, 0, 1:self.args.gamma + 1] = torch.tensor(flatten_draft_k_seq_ids)
                # draft_prob_comm[0, 1:prev_size+1, :] = torch.from_numpy(draft_prob)

                draft_prob_comm[0] = cur_k
                draft_prob_comm[1 : self.args.gamma + 1] = torch.tensor(flatten_draft_k_seq_ids)

                self.draft_forward_times += self.args.gamma
                input_ids = prefix

            else:  # target model
                input_ids = prefix.to(device)
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len - prev_size : prefix_len, : self.vocab_size]
                prob = prob.to(device)
                self.target_forward_times += 1

            if self.accelerator.is_main_process:
                dist.send(draft_prob_comm, dst=1)
            else:
                dist.recv(draft_prob_comm, src=0)

            if not self.accelerator.is_main_process:  # target model

                cur_k = int(draft_prob_comm[0].item())
                flatten_draft_k_seq_ids = draft_prob_comm[1 : self.args.gamma + 1]

                # prev_size = self.args.gamma / prev_k
                prev_ids_draft = prefix[:, prefix_len - prev_size + 1 : prefix_len]  # 两个进程的输入

                prev_prob_draft = []

                cur_size = int(self.args.gamma / cur_k)
                cur_ids_k_seq_draft = (
                    torch.tensor(flatten_draft_k_seq_ids, dtype=torch.int).reshape(cur_k, -1).to(device)
                )  # [k, cur_size]

                prob_target = prob

            else:
                cur_ids_k_seq_draft = (
                    torch.tensor(flatten_draft_k_seq_ids, dtype=torch.int).reshape(cur_k, -1).to(device)
                )
                cur_size = int(self.args.gamma / cur_k)
            if cur_mode:
                if not self.accelerator.is_main_process:

                    flag, resampled_token_id, chosen_draft_tokens_seq_idx = self.verify_first_token_for_k_seq(
                        cur_ids_k_seq_draft,
                        prev_prob_draft,
                        prob_target[:, [-1], :],
                    )

                    if flag == False:
                        # reject the first token in all k seq
                        prefix = torch.cat((input_ids, resampled_token_id), dim=1)

                        # record the number of accepted tokens
                        self.num_acc_tokens.append(num_acc_token)
                        num_acc_token = 0

                        comm_tensor[0] = 1
                        comm_tensor[1] = 0
                        comm_tensor[2] = resampled_token_id
                    else:
                        # accept the first token, change the mode
                        cur_mode = False
                        prefix = torch.cat(
                            (
                                input_ids,
                                cur_ids_k_seq_draft[chosen_draft_tokens_seq_idx].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )
                        num_acc_token += 1
                        comm_tensor[0] = 0
                        comm_tensor[1] = chosen_draft_tokens_seq_idx
                    dist.send(comm_tensor, dst=0)
                else:  # draft model
                    dist.recv(comm_tensor, src=1)
                    cur_mode = comm_tensor[0].item()
                    if cur_mode:
                        prefix = torch.cat((input_ids, comm_tensor[2:3].unsqueeze(dim=0)), dim=1)
                        # model.rollback(prefix_len)
                    else:
                        chosen_draft_tokens_seq_idx = comm_tensor[1].item()
                        prefix = torch.cat(
                            (
                                input_ids,
                                cur_ids_k_seq_draft[chosen_draft_tokens_seq_idx].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )

                        if cur_k != 1:
                            llama_cpp.llama_state_set_data(
                                model._model.ctx,
                                model.kv_cache[chosen_draft_tokens_seq_idx][0],
                                model.kv_cache[chosen_draft_tokens_seq_idx][3],
                            )
                            model._model.input_ids[: model.kv_cache[chosen_draft_tokens_seq_idx][2]] = model.kv_cache[
                                chosen_draft_tokens_seq_idx
                            ][1]

            else:
                if not self.accelerator.is_main_process:
                    n = prev_size - 1
                    for i in range(prev_size - 1):
                        token = prev_ids_draft[:, i]
                        r = torch.rand(1, device=device)

                        if r > prob_target[:, i, token]:
                            n = i
                            break
                    if n == prev_size - 1:
                        flag, resampled_token_id, chosen_draft_tokens_seq_idx = self.verify_first_token_for_k_seq(
                            cur_ids_k_seq_draft,
                            prev_prob_draft,
                            prob_target[:, [-1], :],
                        )

                        if flag == False:
                            cur_mode = True
                            comm_tensor[0] = 1
                            comm_tensor[1] = n
                            comm_tensor[2] = resampled_token_id
                            dist.send(comm_tensor, dst=0)
                            prefix = torch.cat((input_ids, resampled_token_id), dim=1)
                            model.rollback(prefix_len)
                        else:
                            comm_tensor[0] = 0
                            comm_tensor[1] = chosen_draft_tokens_seq_idx
                            dist.send(comm_tensor, dst=0)
                            prefix = torch.cat(
                                (
                                    input_ids,
                                    cur_ids_k_seq_draft[chosen_draft_tokens_seq_idx].unsqueeze(dim=0),
                                ),
                                dim=1,
                            )
                    else:
                        cur_mode = True
                        t = torch.where(prob_target[0, n, :] == 1)[0].unsqueeze(0)

                        comm_tensor[0] = 1
                        comm_tensor[1] = n
                        comm_tensor[2] = t
                        dist.send(comm_tensor, dst=0)
                        prefix = torch.cat((input_ids[:, : prefix_len - prev_size + n + 1], t), dim=1)
                        model.rollback(prefix_len - prev_size + n + 1)

                else:  # draft model
                    dist.recv(comm_tensor, src=1)
                    cur_mode = comm_tensor[0].item()
                    n = comm_tensor[1].item()
                    t = comm_tensor[2].item()
                    if cur_mode == 0:
                        prefix = torch.cat((input_ids, cur_ids_k_seq_draft[n].unsqueeze(dim=0)), dim=1)

                        if cur_k != 1:
                            llama_cpp.llama_state_set_data(
                                model._model.ctx,
                                model.kv_cache[n][0],
                                model.kv_cache[n][3],
                            )
                            model._model.input_ids[: model.kv_cache[n][2]] = model.kv_cache[n][1]

                    else:  # reject someone
                        prefix = torch.cat(
                            (
                                input_ids[:, : prefix_len - prev_size + n + 1],
                                comm_tensor[2:3].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )

            prev_k = cur_k
            prev_size = cur_size

        return prefix

    @torch.no_grad()
    def verify_first_token_for_k_seq(self, draft_tokens_k_seq, draft_prob_k_seq, target_prob):
        flag = False  # if any accepted
        resampled_token_id = 0
        chosen_draft_tokens_seq_idx = 0

        first_token_k_seq = draft_tokens_k_seq[:, 0]

        r = torch.rand(1, device=target_prob.device)

        if r > target_prob[:, 0, first_token_k_seq[0]]:

            t = torch.where(target_prob[0, 0, :] == 1)[0].unsqueeze(0)
            resampled_token_id = t
            idx = 0
            for idx, first_token in enumerate(first_token_k_seq[1:], 1):
                if t == first_token:
                    flag = True
                    chosen_draft_tokens_seq_idx = idx
                    break
        else:
            flag = True
            chosen_draft_tokens_seq_idx = 0

        return flag, resampled_token_id, chosen_draft_tokens_seq_idx

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")

    @torch.no_grad()
    def tridecoding(self, prefix):
        """
        Three-layer speculative decoding
        """
        local_accepted_tokens = 0
        local_total_tokens = 0
        local_draft_forwards = 0
        local_target_forwards = 0

        max_tokens = prefix.shape[1] + self.args.max_tokens
        smallest_device = self.smallest_model.device
        small_device = self.draft_model.device
        target_device = self.target_model.device
        smallest_model_cache = KVCacheModel(self.smallest_model, self.args.temp, self.args.top_k, self.args.top_p)
        smallest_model_cache.vocab_size = self.vocab_size
        small_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        small_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size
        local_decode_time = 0
        loop_index = 0
        gamma1 = self.args.gamma
        gamma2 = self.args.gamma

        while prefix.shape[1] < max_tokens:
            begin_time = time.time()
            prefix_len = prefix.shape[1]

            # --- Generation ---
            draft1_tokens = smallest_model_cache.generate(prefix.to(smallest_device), gamma1)
            local_draft_forwards += gamma1
            draft2_tokens = small_model_cache.generate(draft1_tokens.to(small_device), gamma2)
            local_draft_forwards += gamma2
            _ = target_model_cache.generate(draft2_tokens.to(target_device), 1)
            local_target_forwards += 1
            # --- Verification ---
            accepted_len = 0
            p_smallest = smallest_model_cache._prob_history
            p_small = small_model_cache._prob_history
            p_target = target_model_cache._prob_history

            final_prefix = prefix
            rejection_handled = False
            for i in range(gamma1 + gamma2):
                current_pos = prefix_len + i

                if i < gamma1:
                    verifier_probs = p_small.to(smallest_device)[:, current_pos - 1, :]
                    drafter_probs = p_smallest[:, current_pos - 1, :]
                    j = draft1_tokens[:, current_pos]
                else:
                    verifier_probs = p_target.to(small_device)[:, current_pos - 1, :]
                    drafter_probs = p_small[:, current_pos - 1, :]
                    j = draft2_tokens[:, current_pos]
                p_verifier_j = verifier_probs[:, j].squeeze()
                p_drafter_j = drafter_probs[:, j].squeeze()
                r = torch.rand(1, device=j.device).squeeze()
                if r * p_drafter_j > p_verifier_j:
                    residual_prob = max_fn(verifier_probs - drafter_probs)

                    if residual_prob.sum() < 1e-9:
                        next_token = sample(verifier_probs)
                    else:
                        next_token = sample(residual_prob)

                    final_prefix = torch.cat([draft2_tokens[:, :current_pos].to(next_token.device), next_token], dim=1)
                    rejection_handled = True
                    break
                else:
                    accepted_len += 1

            if not rejection_handled:
                last_pos_probs = p_target[:, -2, :]
                next_token = sample(last_pos_probs)
                final_prefix = torch.cat([draft2_tokens.to(next_token.device), next_token], dim=1)
                accepted_len += 1
            prefix = final_prefix

            local_total_tokens += gamma1 + gamma2
            local_accepted_tokens += accepted_len
            self.num_acc_tokens.append(accepted_len)

            # --- Rollback ---
            # 对三个模型应用不同的回滚策略
            new_len = prefix.shape[1]

            # 对于 small 和 target 模型，回滚到新的前缀长度是安全的
            small_model_cache.rollback(new_len)
            target_model_cache.rollback(new_len)

            # 少回滚一个 token，让其缓存长度为 new_len - 1，避免 IndexError
            if new_len > 0:
                smallest_model_cache.rollback(new_len - 1)
            else:
                smallest_model_cache.rollback(0)  # 如果 prefix 是空的，回滚到0

            end_time = time.time()
            local_decode_time += end_time - begin_time
            loop_index += 1
        return (
            prefix,
            local_accepted_tokens,
            local_total_tokens,
            local_draft_forwards,
            local_target_forwards,
            local_decode_time,
        )
