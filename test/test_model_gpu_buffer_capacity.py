from types import SimpleNamespace

import torch

from src.model_gpu import KVCacheModel


class _EmbeddingModel:
    config = SimpleNamespace(vocab_size=7)

    def get_input_embeddings(self) -> SimpleNamespace:
        return SimpleNamespace(weight=torch.empty((7, 2)))


def test_explicit_max_length_sets_initial_buffer_capacity() -> None:
    given = KVCacheModel(_EmbeddingModel(), max_length=5)

    given._ensure_buffer_size(1, 2, torch.device("cpu"), torch.float32)

    assert given.max_length == 5
    assert given._prob_buffer is not None
    assert given._logits_buffer is not None
    assert given._prob_buffer.shape == (1, 5, 7)
    assert given._logits_buffer.shape == (1, 5, 7)


def test_default_max_length_keeps_existing_initial_buffer_capacity() -> None:
    given = KVCacheModel(_EmbeddingModel())

    given._ensure_buffer_size(1, 2, torch.device("cpu"), torch.float32)

    assert given.max_length == 2048


def test_explicit_max_length_expands_when_sequence_exceeds_capacity() -> None:
    given = KVCacheModel(_EmbeddingModel(), max_length=2)

    given._ensure_buffer_size(1, 3, torch.device("cpu"), torch.float32)

    assert given.max_length == 1027
    assert given._prob_buffer is not None
    assert given._prob_buffer.shape == (1, 1027, 7)
