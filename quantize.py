from enum import Enum, unique
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def compute_code_histogram(onehots: torch.Tensor,
                           paddings: torch.Tensor) -> torch.Tensor:
    """Compute histogram of quantized codes, aggregated across distributed processes."""
    mask = (1 - paddings).unsqueeze(-1).unsqueeze(-1).float()
    local_hist = (onehots * mask).sum(dim=tuple(range(onehots.dim() - 2)))
    if dist.is_initialized():
        world_size = dist.get_world_size()
        hist_list = [torch.zeros_like(local_hist) for _ in range(world_size)]
        dist.all_gather(hist_list, local_hist)
        return sum(hist_list)
    return local_hist


def compute_code_pplx(onehots: torch.Tensor, paddings: torch.Tensor):
    """Compute perplexity and entropy over code assignments across processes."""
    hist = compute_code_histogram(onehots, paddings)
    local_count = (1 - paddings).sum().float()
    if dist.is_initialized():
        total_count = local_count.clone()
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    else:
        total_count = torch.clamp(local_count, min=1.0)
    probs = hist / torch.clamp(total_count, min=1.0)
    log_probs = torch.log(torch.clamp(probs, min=1e-30))
    sum_plogp = (probs * log_probs).sum(dim=-1)
    pplx = torch.exp(-sum_plogp).mean()
    entropy = torch.log(pplx)
    return pplx, entropy


def compute_code_coverage(onehots: torch.Tensor,
                          paddings: torch.Tensor) -> torch.Tensor:
    """Compute average codebook coverage aggregated across processes."""
    hist = compute_code_histogram(onehots, paddings)
    codebook_size = onehots.size(-1)
    covered = (hist > 0).float().sum(dim=-1)
    local_cov = covered / codebook_size
    if dist.is_initialized():
        total_cov = local_cov.clone()
        dist.all_reduce(total_cov, op=dist.ReduceOp.SUM)
        return total_cov / dist.get_world_size()
    return local_cov


@unique
class SimilarityMetric(Enum):
    L2 = 0
    DOT = 1


@unique
class BaseQuantizer(nn.Module):
    """Abstract vector quantizer base class."""

    def __init__(self, num_codebooks: int, codebook_size: int,
                 codebook_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        # [G, V, D]
        self.codebook = nn.Parameter(
            torch.randn(num_codebooks, codebook_size, codebook_dim))

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        raise NotImplementedError

    def lookup(self, ids: torch.LongTensor) -> torch.Tensor:
        expanded = self.codebook.unsqueeze(0)
        for _ in range(ids.dim() - 1):
            expanded = expanded.unsqueeze(0)
        idx = ids.unsqueeze(-1).expand(*ids.shape, self.codebook_dim)
        return torch.gather(expanded, dim=-2, index=idx)


def quantize_by_nearest_neighbor(inputs: torch.Tensor, codebook: torch.Tensor,
                                 metric: SimilarityMetric):
    batch_shape = inputs.shape[:-2]
    G, D = inputs.shape[-2:]
    V = codebook.size(1)
    inp = inputs.unsqueeze(-3)  # [...,1,G,D]
    cb = codebook.unsqueeze(0).expand(*batch_shape, V, G, D)
    if metric == SimilarityMetric.L2:
        dist_mat = ((inp - cb)**2).sum(-1)
    else:
        dist_mat = -(inp * cb).sum(-1)
    ids = dist_mat.argmin(dim=-2)  # [...,G]
    onehots = F.one_hot(ids, num_classes=V).float()
    oh = onehots.permute(*range(len(batch_shape)), -1, -2)
    quantized = torch.einsum(f"{''.join(['...']):s}vg, vgd -> {'...'}gd", oh,
                             codebook)
    return ids, quantized, onehots


class RandomVectorQuantizer(BaseQuantizer):
    """Random-projection quantizer (frozen codebook)."""

    def __init__(
        self,
        input_dim: int,
        num_codebooks: int,
        codebook_size: int,
        codebook_dim: int,
        normalize_codebook: bool = True,
        normalize_inputs: bool = True,
    ):
        super().__init__(num_codebooks, codebook_size, codebook_dim)
        self.normalize_codebook = normalize_codebook
        self.normalize_inputs = normalize_inputs
        self.proj = nn.Linear(input_dim,
                              num_codebooks * codebook_dim,
                              bias=False)
        if normalize_codebook:
            with torch.no_grad():
                self.codebook.data = F.normalize(self.codebook.data, dim=-1)

    def forward(self, inputs: torch.Tensor, paddings: torch.Tensor):
        B, T, _ = inputs.shape
        x = self.proj(inputs).view(B, T, self.num_codebooks, self.codebook_dim)
        if self.normalize_inputs:
            x = F.normalize(x, dim=-1)
        metric = SimilarityMetric.DOT if self.normalize_codebook else SimilarityMetric.L2
        ids, quantized, _ = quantize_by_nearest_neighbor(
            x, self.codebook, metric)
        pad = paddings.unsqueeze(-1)
        ids = ids.masked_fill(pad.bool(), -1)
        quantized = quantized * (1 - pad).unsqueeze(-1)
        return ids, quantized


class KmeansVectorQuantizer(BaseQuantizer):
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
