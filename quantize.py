import math
from dataclasses import dataclass
from re import match
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from codebook_utils import (SimilarityMetric, _add_codebook_summaries,
                            _apply_paddings, _einsum_dims, _ids_to_onehots,
                            _lookup, quantize_by_nearest_neighbor)


@dataclass
class QuantizerOutputs:
    # [..., num_codebooks].
    ids: torch.Tensor
    # [..., num_codebooks, codebook_dim].
    quantized_vectors: torch.Tensor
    # Scalar of quantizer loss.
    loss: Optional[Dict[str, torch.Tensor]] = None
    summary: Optional[Dict] = None


class RandomVectorQuantizer(nn.Module):
    """Random-projection quantizer (frozen codebook).
    https://arxiv.org/pdf/2202.01855
    """

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
        """forward for training"""
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


class VectorQuantizer(nn.Module):
    """Trainable VQ-VAE style quantizer with MSE + commitment losses.
    https://arxiv.org/pdf/1711.00937.pdf
    https://arxiv.org/pdf/1910.05453.pdf. (vq-wav2vec)
    """

    def __init__(self,
                 num_codebooks: int,
                 codebook_size: int,
                 codebook_dim: int,
                 normalize_codebook: bool = False,
                 normalize_inputs: bool = True,
                 beta: float = 1.0):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.normalize_codebook = normalize_codebook
        self.normalize_inputs = normalize_inputs
        # [G, V, D]
        self.codebook = nn.Parameter(
            torch.rand(codebook_size, num_codebooks, codebook_dim))
        self.beta = beta

        if self.normalize_codebook:
            with torch.no_grad():
                self.codebook.data = F.normalize(self.codebook.data, dim=-1)

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        """forward for training"""
        inputs = inputs * (1 - paddings)[:, :, None]
        B, T, _ = inputs.shape
        G, D = self.num_codebooks, self.codebook_dim
        x = inputs.view(B, T, G, D)
        ids, quantized = quantize_by_nearest_neighbor(
            x, self.codebook, SimilarityMetric.L2_DISTANCE)
        ids = ids.view(B, T, G)
        quantized = quantized.view(B, T, G, D)
        ids, quantized = _apply_paddings(ids, quantized, self.codebook)

        q_vec = quantized.view([B, T, -1])
        num_frames = (1 - paddings).sum()
        denominator = (num_frames * G * D).clamp(min=1)

        inputs_to_loss = inputs
        if self.normalize_codebook:
            inputs_to_loss = F.normalize(inputs, dim=-1)
        kmeans_loss = ((q_vec - inputs_to_loss.detach())**2 *
                       (1 - paddings)[:, :, None]).sum() / denominator
        commitment_loss = ((inputs_to_loss - q_vec.detach())**2 *
                           (1 - paddings)[:, :, None]).sum() / denominator
        total_loss = kmeans_loss + self.beta * commitment_loss

        # Straight-through estimator such that dL/inputs = dL/q_outputs.
        # Note that gradient on quantized_vectors is not propagated to the codebook.
        quantized_vectors = inputs + (q_vec - inputs).detach()
        # We need this to stop gradients on the padded inputs.
        quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None]

        onehots = _ids_to_onehots(
            ids * (1 - paddings)[:, :, None],
            codebook_size=self.codebook_size,
        )
        infos = _add_codebook_summaries(onehots, paddings)
        return QuantizerOutputs(
            ids=ids,
            quantized_vectors=quantized_vectors,
            loss={
                "total_loss": total_loss,
                "kmeans_loss": kmeans_loss,
                "commitment_loss": commitment_loss,
            },
            summary=infos,
        )

    def quantize(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape
        G, D = self.num_codebooks, self.codebook_dim
        x = inputs.view(B, T, G, D)
        ids, quantized = quantize_by_nearest_neighbor(
            x, self.codebook, SimilarityMetric.L2_DISTANCE)
        # [B,T,G], [B,T,d]
        return ids, quantized


class VectorQuantizerEMA(nn.Module):
    """Trainable VQ-VAE style quantizer with EMA updates for the codebook."""

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        normalize_codebook: bool = False,
        normalize_inputs: bool = False,
        beta: float = 1.0,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.normalize_codebook = normalize_codebook
        self.normalize_inputs = normalize_inputs
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        codebook_init = torch.randn(codebook_size, num_codebooks, codebook_dim)
        if self.normalize_codebook:
            codebook_init = F.normalize(codebook_init, p=2, dim=-1)

        self.register_buffer('codebook', codebook_init)
        # EMA buffers
        self.register_buffer('ema_cluster_size',
                             torch.zeros(codebook_size, num_codebooks))
        self.register_buffer('ema_dw', codebook_init.clone())

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        inputs = inputs * (1 - paddings)[:, :, None]
        B, T, _ = inputs.shape
        G, D = self.num_codebooks, self.codebook_dim
        x = inputs.view(B, T, G, D)
        ids, quantized = quantize_by_nearest_neighbor(
            x, self.codebook, SimilarityMetric.L2_DISTANCE)
        ids = ids.view(B, T, G)
        quantized = quantized.view(B, T, G, D)
        ids, quantized = _apply_paddings(ids, quantized, self.codebook)
        q_vec = quantized.view([B, T, -1])

        num_frames = (1 - paddings).sum()
        denominator = (num_frames * G * D).clamp(min=1)
        inputs_to_loss = inputs
        if self.normalize_inputs:
            inputs_to_loss = F.normalize(inputs, dim=-1)
        commitment_loss = ((inputs_to_loss - q_vec.detach())**2 *
                           (1 - paddings)[:, :, None]).sum() / denominator

        total_loss = self.beta * commitment_loss

        onehots = _ids_to_onehots(
            ids * (1 - paddings)[:, :, None],
            codebook_size=self.codebook_size,
        )  # [B,T,G, C]
        onehots = onehots * (1 - paddings).to(onehots.dtype)[:, :, None]

        if self.training:
            current_cluster_size = onehots.sum(dim=(0,
                                                    1)).transpose(0,
                                                                  1)  # [C,G]

            current_dw = torch.einsum('btgd,btgv->vgd', x, onehots.clone())
            if dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(current_cluster_size, op=dist.ReduceOp.SUM)
                dist.all_reduce(current_dw, op=dist.ReduceOp.SUM)

            self.ema_cluster_size.data.mul_(self.decay).add_(
                current_cluster_size, alpha=1 - self.decay)
            self.ema_dw.data.mul_(self.decay).add_(current_dw,
                                                   alpha=1 - self.decay)

            n_sum_cluster_size = self.ema_cluster_size.sum(
                dim=0, keepdim=True)  # (1, G)
            smoothed_cluster_size = (
                (self.ema_cluster_size + self.epsilon) /
                (n_sum_cluster_size + self.codebook_size * self.epsilon) *
                n_sum_cluster_size)

            updated_codebook = self.ema_dw / (
                smoothed_cluster_size.unsqueeze(-1) + self.epsilon
            )  # (V,G,D_sub)

            if self.normalize_codebook:
                updated_codebook = F.normalize(updated_codebook, p=2, dim=-1)

            self.codebook.data.copy_(updated_codebook)

        # Straight-through estimator such that dL/inputs = dL/q_outputs.
        # Note that gradient on quantized_vectors is not propagated to the codebook.
        quantized_vectors = inputs + (q_vec - inputs).detach()
        # We need this to stop gradients on the padded inputs.
        quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None]
        infos = _add_codebook_summaries(onehots, paddings)

        return QuantizerOutputs(
            ids=ids,
            quantized_vectors=quantized_vectors,
            loss={
                "total_loss": total_loss,
                "commitment_loss": commitment_loss,
            },
            summary=infos,
        )

    def quantize(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape
        G, D = self.num_codebooks, self.codebook_dim
        x = inputs.view(B, T, G, D)
        ids, quantized = quantize_by_nearest_neighbor(
            x, self.codebook, SimilarityMetric.L2_DISTANCE)
        # [B,T,G], [B,T,d]
        return ids, quantized


class GumbelSoftmaxVectorQuantizer(nn.Module):
    """Vector quantizer with Gumbel-Softmax straight-through estimator.
    https://arxiv.org/pdf/2006.11477
    """

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        temperature_fn: Callable,
        normalize_codebook: bool = False,
        normalize_inputs: bool = False,
    ):
        super().__init__()
        self.temperature_fn = temperature_fn
        self.input_proj = nn.Linear(input_dim, num_codebooks * codebook_size)
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.normalize_codebook = normalize_codebook
        self.normalize_inputs = normalize_inputs

        self.codebook = nn.Parameter(
            torch.randn(
                codebook_size,
                num_codebooks,
                codebook_dim,
            ))

    def forward(self,
                inputs: torch.Tensor,
                paddings: torch.Tensor,
                step: int = 0):
        """forward for training"""
        B, T, _ = inputs.shape
        logits = self.input_proj(inputs).view(B, T, self.num_codebooks,
                                              self.codebook_size)
        # [batch_size, seq_len, num_codebooks].
        ids = torch.argmax(logits, dim=-1)
        tau = self.temperature_fn(step)
        gumbel = -torch.empty_like(logits).exponential_().log()
        logits = (logits + gumbel) / tau

        # [batch_size, seq_len, 1].
        mask = (1 - paddings)[:, :, None].to(ids.dtype)
        ids = ids * mask + (-1) * (1 - mask)
        onehots = _ids_to_onehots(ids * mask, codebook_size=self.codebook_size)
        # We need this to stop gradients on the padded frames.
        mask = mask.to(inputs.dtype)
        onehots = onehots * mask[:, :, :, None]
        # [batch_size, seq_len, num_codebooks, vocab_size].
        y_soft = torch.nn.functional.softmax(logits, dim=-1)
        y_soft = y_soft * mask[:, :, :, None]

        # Straight-through estimator such that dL/y_soft = dL/onehots.
        onehots = y_soft + (onehots - y_soft).detach()
        batch_dims = _einsum_dims[:onehots.ndim - 2]
        quantized_vectors = torch.einsum(f"{batch_dims}gv,vgh->{batch_dims}gh",
                                         onehots, self.codebook)
        quantized_vectors = quantized_vectors * mask[:, :, :, None]
        infos = _add_codebook_summaries(onehots=onehots, paddings=paddings)
        return QuantizerOutputs(
            # [batch_size, seq_len, num_codebooks].
            ids=ids,
            # [batch_size, seq_len, num_codebooks, codebook_dim].
            quantized_vectors=quantized_vectors,
            summary=infos,
        )

    def quantize(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape

        logits = self.input_proj(inputs).view(B, T, self.num_codebooks,
                                              self.codebook_size)
        # [batch_size, seq_len, num_codebooks].
        ids = torch.argmax(logits, dim=-1)

        return _lookup(ids, self.codebook)


class LookupFreeQuantizer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
    ) -> None:
        super().__init__()

        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # NOTE: log2(codebook_size) per group
        self.proj = torch.nn.Linear(input_dim, num_codebooks * codebook_dim)

    def _get_indices(self, dim_indices: torch.Tensor,
                     grouped_base: torch.Tensor):
        """Convert from per-dimension indices into global indices in groups.

          The number of dimensions does not need to be divisible by group_size.

          Args:
              dim_indices: per-dimension indices, e.g. [0, 1, 0, 1, 0]
              grouped_base: grouped per-dimension bases, e.g. [1, 2, 4, 1, 2]
          Returns:
              global indices, e.g., [2, 1] shape [B,T,G]
        """
        group_size = self.num_codebooks * self.codebook_dim
        indices = dim_indices * grouped_base
        pad_len = (group_size - indices.shape[-1] % group_size) % group_size
        indices = torch.cat([
            indices,
            torch.zeros((*indices.shape[:-1], pad_len),
                        device=dim_indices.device,
                        dtype=torch.uint32)
        ],
                            dim=-1)
        indices = indices.reshape(*indices.shape[:-1], -1,
                                  group_size).sum(dim=-1)
        if indices.shape[-1] == 1:
            indices = indices[..., 0]
        return indices

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        """forward for training"""
        inputs = self.proj(inputs)  # [B,T,G*D]
        G, D = self.num_codebooks, self.codebook_dim
        base = torch.pow(
            2,
            torch.arange(self.codebook_dim * self.num_codebooks,
                         dtype=torch.uint32,
                         device=inputs.device) % self.num_codebooks)
        samples = inputs >= 0
        quantized = torch.where(samples, 1.0, -1.0)
        ids = self._get_indices(samples, base)

        inputs_to_loss = inputs * (1-paddings)[:,;,None]
        q_vec = quantized
        num_frames = (1 - paddings).sum()
        denominator = (num_frames * G * D).clamp(min=1)
        commitment_loss = 0.0
        # Commitment loss
        commitment_loss = ((inputs_to_loss - q_vec.detach())**2 *
                               (1 - paddings)[:, :, None]).sum() / denominator

        # TODO: Entropy loss
        total_loss = commitment_loss
        quantized_vectors = inputs + (q_vec - inputs).detach()

        onehots = _ids_to_onehots(ids * (1-paddings)[:,:,None], codebook_size=self.codebook_size)
        infos = _add_codebook_summaries(onehots, paddings)
        return QuantizerOutputs(
            ids=ids,
            quantized_vectors=quantized_vectors,
            loss={
                "total_loss": total_loss,
                "commitment_loss": commitment_loss,
            },
            summary=infos,
        )

    def quantize(self, inputs: torch.Tensor, paddings: torch.Tensor):
        inputs = self.proj(inputs)  # [B,T,G*D]
        G, D = self.num_codebooks, self.codebook_dim
        base = torch.pow(
            2,
            torch.arange(self.codebook_dim * self.num_codebooks,
                         dtype=torch.uint32,
                         device=inputs.device) % self.num_codebooks)
        samples = inputs >= 0
        quantized = torch.where(samples, 1.0, -1.0)
        ids = self._get_indices(samples, base)
        return ids, quantized


