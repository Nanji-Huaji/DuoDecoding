import torch
import json
import torch.distributed as dist
import numpy as np
import os
import transformers
import warnings

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


from .model.rest.rest.model.utils import *
from .model.rest.rest.model.rest_model import RestModel
from .model.rest.rest.model.kv_cache import initialize_past_key_values
import draftretriever

from typing import List, Tuple, Dict, Any, TypedDict, Union, Optional

from engine import Decoding

class Baseline(Decoding):
    def __init__(self, args):
        super().__init__(args)

    def load_model(self):
        super().load_model()

    