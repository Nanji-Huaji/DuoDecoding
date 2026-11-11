"""Private RNG context for the draws a decoding loop makes outside the model.

A per-decode-invocation context owns one private ``torch.Generator`` per actual
sampling device. Draft, acceptance, residual and bonus draws reuse the stream for
their device; different devices use distinct keys instead of cloned default CUDA
streams. This removes random-word reuse caused by identical default seeds and
offsets; it is not a claim of mathematical independence for every PRNG output.

Entropy is allocated lazily from the default CPU generator at the first real
draw: one scalar int64 draw becomes the base seed. That draw is metadata for
seeding private generators, not an extra sampled token, so it does not consume
vocabulary randomness. Construction allocates no generators or tensors and
consumes no entropy. Reseeding via ``torch.manual_seed`` before the first draw
reproduces seed allocation; a fresh context without reseeding does not restart
from a constant.

Private device generators do not touch the default CUDA streams. Weight
preprocessing and generator construction precede seed allocation; failures in
those operations consume no entropy. Callers must validate sizes and raw rows
before drawing; this helper does not make every invalid Torch argument RNG-free.

This module does not validate raw probability rows or model inputs: ``sample``
composes the existing ``sampling_weights`` preprocessing (including its
invalid-row fallback) exactly once, and callers own validation. It has no model
or callback dependencies.
"""

import torch

from .sampling import sampling_weights

_MODULUS = 1 << 64


def _device_for_generator(device: torch.device) -> torch.device:
    """Normalize the device identity used for generator lookup.

    ``cpu:0`` and ``cpu`` are the same physical device. An unresolved ``cuda``
    resolves to the current index; an explicit ``cuda:N`` never queries CUDA.
    """
    if device.type == "cpu":
        return torch.device("cpu")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class DecodingRng:
    """One private RNG context per decode invocation."""

    def __init__(self) -> None:
        self._base: int | None = None
        self._generators: dict[torch.device, torch.Generator] = {}

    def _base_entropy(self) -> int:
        if self._base is None:
            # Metadata entropy from the default CPU generator; not a token draw.
            entropy = torch.empty((), dtype=torch.int64, device="cpu").random_()
            self._base = int(entropy.item()) % _MODULUS
        return self._base

    def _generator_for(self, device: torch.device) -> torch.Generator:
        normalized = _device_for_generator(device)
        generator = self._generators.get(normalized)
        if generator is not None:
            return generator

        # Construct first so an unsupported device fails before entropy is used.
        generator = torch.Generator(device=normalized)
        seed = (self._base_entropy() + len(self._generators)) % _MODULUS
        generator.manual_seed(seed)
        self._generators[normalized] = generator
        return generator

    def sample(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Draw one categorical token per row with the private device generator.

        Applies the existing ``sampling_weights`` preprocessing exactly once,
        then calls ``torch.multinomial(..., num_samples=1)`` with an explicit
        generator. Raw probability validation belongs to the caller.
        """
        weights = sampling_weights(probabilities)
        generator = self._generator_for(weights.device)
        return torch.multinomial(weights, num_samples=1, generator=generator)

    def rand(self, shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        """Like ``torch.rand(shape, device=device)`` with the private generator.

        Default dtype semantics are preserved. Callers pass positive,
        already-validated sizes.
        """
        normalized = _device_for_generator(device)
        generator = self._generator_for(normalized)
        return torch.rand(shape, device=normalized, generator=generator)
