import transformers
import openai
import torch
from typing import List, Optional, Union


class VLLMModelWrapper:
    def __init__(self):
        self.device = "cpu"


class VLLMWrapper:
    def __init__(self, model_name: str, tokenizer: transformers.PreTrainedTokenizer, base_url: str, **sampling_kwargs):
        """
        Initializes the VLLMWrapper with the model name, tokenizer, base URL, and sampling parameters.
        sampling_kwargs: Additional keyword arguments for sampling, such as temperature, top_p, etc.
        - temperature: float
        - top_p: float
        - top_k: int
        """
        model_dict = {
            "vicuna/tiny-vicuna-1b": "tiny-vicuna-1b",
            "vicuna/vicuna-13b-v1.5": "vicuna-13b-v1.5",
            "vicuna/vicuna-68m": "vicuna-68m",
        }
        self.model_name = model_dict.get(model_name, model_name)
        self.tokenizer = tokenizer
        self.sampling_kwargs = sampling_kwargs
        self.sampling_kwargs.pop("top_k", None)

        self.client = openai.Client(base_url=base_url, api_key="not_needed_but_required")

        self.model = VLLMModelWrapper()

    @property
    def device(self):
        """
        Returns the device of the model, which is always 'cpu' for this wrapper.
        """
        return "cpu"

    def generate(self, prompt: Union[List[int], torch.Tensor], max_new_tokens: int = 1) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            prompt_list = prompt.flatten().to(torch.int64).cpu().tolist()
        else:
            # 输入本身就是列表
            prompt_list = prompt
        response = self.client.completions.create(
            model=self.model_name, prompt=prompt_list, max_tokens=max_new_tokens, logprobs=1, **self.sampling_kwargs
        )
        generated_token_strings = response.choices[0].logprobs.tokens
        newly_generated_ids = self.tokenizer.convert_tokens_to_ids(generated_token_strings)
        if isinstance(newly_generated_ids, int):
            newly_generated_ids = [newly_generated_ids]
        full_sequence_list = prompt_list + newly_generated_ids
        return torch.tensor(full_sequence_list, dtype=torch.int64).unsqueeze(0)

    __call__ = generate
