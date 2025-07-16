import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
import shortuuid
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from fastchat.model import get_conversation_template
from typing import List, Tuple

from src.engine import DecodingMetrics


class TestDecoding(Decoding):
    def __init__(self, args):
        super().__init__(args)
