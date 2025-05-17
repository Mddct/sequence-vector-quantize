from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from codebook_utils import (SimilarityMetric, _add_codebook_summaries,
                            _apply_paddings, _ids_to_onehots,
                            quantize_by_nearest_neighbor)


@dataclass
class QuantizerOutputs:
    # [..., num_codebooks].
    ids: torch.Tensor
    # [..., num_codebooks, codebook_dim].
    quantized_vectors: torch.Tensor
    # Scalar of quantizer loss.
    loss: Optional[torch.Tensor] = None
    summary: Optional[Dict] = None


class RandomVectorQuantizer(nn.Module):
    """Random-projection quantizer (frozen codebook)."""

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        normalize_codebook: bool = True,
        normalize_inputs: bool = True,
        seed=2025,
    ):
        super().__init__()
        proj_g = torch.Generator()
        proj_g.manual_seed(seed)

        codebook_g = torch.Generator()
        codebook_g.manual_seed(seed + 1)
        self.normalize_codebook = normalize_codebook
        self.normalize_inputs = normalize_inputs
        self.proj = nn.Parameter(torch.rand(input_dim,
                                            num_codebooks * codebook_dim,
                                            generator=proj_g),
                                 requires_grad=False)
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        # [G, V, D]
        self.codebook = nn.Parameter(torch.randn(codebook_size,
                                                 num_codebooks,
                                                 codebook_dim,
                                                 generator=codebook_g),
                                     requires_grad=False)
        if normalize_codebook:
            with torch.no_grad():
                self.codebook.data = F.normalize(self.codebook.data, dim=-1)

    @torch.no_grad()
    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape
        x = torch.matmul(inputs, self.proj).view(B, T, self.num_codebooks,
                                                 self.codebook_dim)
        if self.normalize_inputs:
            x = F.normalize(x, dim=-1)
        metric = SimilarityMetric.DOT_PRODUCT if self.normalize_codebook \
            else SimilarityMetric.L2_DISTANCE
        ids, quantized = quantize_by_nearest_neighbor(x, self.codebook, metric)
        ids, quantized = _apply_paddings(ids, quantized, paddings)

        onehots = _ids_to_onehots(ids, self.codebook_size)
        infos = _add_codebook_summaries(onehots, paddings)
        return QuantizerOutputs(
            ids=ids,
            quantized_vectors=quantized,
            summary=infos,
        )
