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


class KmeansVectorQuantizer(nn.Module):
    """Trainable VQ-VAE style quantizer with MSE + commitment losses."""

    def __init__(self,
                 num_codebooks: int,
                 codebook_size: int,
                 codebook_dim: int,
                 beta: float = 1.0):
        super().__init__(num_codebooks, codebook_size, codebook_dim)
        self.beta = beta

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape
        G, D = self.num_codebooks, self.codebook_dim
        x = inputs.view(B, T, G, D)
        ids, quantized, onehots = quantize_by_nearest_neighbor(
            x, self.codebook, SimilarityMetric.L2)
        mask = (1 - paddings).view(B, T, 1, 1).float()
        kmeans_loss = ((quantized - x.detach())**2 * mask).sum() / mask.sum()
        commitment_loss = (
            (x - quantized.detach())**2 * mask).sum() / mask.sum()
        total_loss = kmeans_loss + self.beta * commitment_loss
        quantized_st = (x + (quantized - x).detach()) * mask
        quantized_st = quantized_st.view(B, T, -1)
        ids = ids.masked_fill(mask.squeeze(-1).squeeze(-1) == 0, -1)
        return ids, quantized_st, {
            'kmeans_loss': kmeans_loss,
            'commitment_loss': commitment_loss,
            'total_loss': total_loss
        }


class GumbelSoftmaxVectorQuantizer(BaseQuantizer):
    """Vector quantizer with Gumbel-Softmax straight-through estimator."""

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        temperature_fn: Callable,
    ):
        super().__init__(num_codebooks, codebook_size, codebook_dim)
        self.temperature_fn = temperature_fn
        self.input_proj = nn.Linear(input_dim, num_codebooks * codebook_size)

    def forward(self,
                inputs: torch.Tensor,
                paddings: torch.Tensor,
                step: int = 0):
        B, T, _ = inputs.shape
        logits = self.input_proj(inputs).view(B, T, self.num_codebooks,
                                              self.codebook_size)
        if self.training:
            tau = self.temperature_fn(step)
            gumbel = -torch.empty_like(logits).exponential_().log()
            logits = (logits + gumbel) / tau
        ids = logits.argmax(dim=-1)
        pad = paddings.unsqueeze(-1)
        ids = ids.masked_fill(pad.bool(), -1)
        probs = F.softmax(logits, dim=-1)
        probs = probs * (1 - pad).unsqueeze(-1)
        onehots = F.one_hot(ids.clamp(min=0),
                            num_classes=self.codebook_size).float()
        if self.training:
            onehots = probs + (onehots - probs).detach()
        oh = onehots.permute(0, 1, 3, 2)
        quantized = torch.einsum("btvg,vgd->btgd", oh, self.codebook)
        quantized = quantized * (1 - pad).unsqueeze(-1)
        quantized_flat = quantized.view(B, T, -1)
        return (ids, quantized_flat), {
            'probs': probs,
            'temperature': tau if self.training else None
        }
