from dataclasses import dataclass
from enum import Enum, unique
from typing import Union

import torch
import torch.distributed as dist


@unique
class SimilarityMetric(Enum):
    L2_DISTANCE = 0
    DOT_PRODUCT = 1


def compute_code_histogram_allgather(onehots: torch.Tensor,
                                     paddings: torch.Tensor) -> torch.Tensor:
    """Compute histogram of quantized codes, aggregated across distributed processes using AllGather."""
    # paddings: [B, T], onehots: [B, T, G, K]
    # G: number of code groups, K: number of codes per group (codebook_size)
    # B: batch_size, T: sequence_length

    # Create a mask to ignore padded positions
    # mask shape: [B, T, 1, 1] to broadcast correctly with onehots
    mask = (~paddings.bool()).unsqueeze(-1).unsqueeze(-1).float()

    # Apply mask to onehots
    # masked_onehots shape: [B, T, G, K]
    masked_onehots = onehots * mask

    # Compute local histogram by summing over batch and time dimensions
    # local_hist shape: [G, K]
    local_hist = masked_onehots.sum(dim=(0, 1))

    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()

        # Create a tensor to store all gathered histograms from all processes.
        # Its shape will be [world_size, G, K].
        gathered_hists_tensor = torch.empty(
            (world_size, ) + local_hist.shape,  # e.g., (N, G, K)
            dtype=local_hist.dtype,
            device=local_hist.device)

        # Perform AllGather.
        # Each process's local_hist will be gathered into the corresponding
        # slice of gathered_hists_tensor on all processes.
        # For example, on rank 0, gathered_hists_tensor[0] will be its own local_hist,
        # gathered_hists_tensor[1] will be rank 1's local_hist, and so on.
        dist.all_gather_into_tensor(gathered_hists_tensor, local_hist)

        # Now, gathered_hists_tensor on each process contains all local histograms.
        # To get the global sum (equivalent to what all_reduce would have given),
        # sum along the dimension that represents the different processes (dim=0).
        # global_hist shape: [G, K]
        global_hist = gathered_hists_tensor.sum(dim=0)

        return global_hist
    else:
        # If not distributed or only one process, the local histogram is the global one.
        return local_hist


def compute_code_pplx(onehots: torch.Tensor, paddings: torch.Tensor):
    """
    Compute perplexity and entropy over code assignments across processes.

    Args:
        onehots (torch.Tensor): A tensor of one-hot encoded codes.
                                Expected shape: [B, T, G, K].
        paddings (torch.Tensor): A boolean or integer tensor indicating padded elements.
                                 Expected shape: [B, T]. True or 1 for padded.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pplx (torch.Tensor): The computed average perplexity (scalar).
            - entropy (torch.Tensor): The computed entropy corresponding to pplx (scalar).
    """
    # Compute the global histogram of code occurrences.
    # hist shape: [G, K]
    hist = compute_code_histogram_allgather(onehots, paddings)

    # Calculate the number of non-padded elements in the local batch.
    # (1 - paddings) creates a mask of non-padded elements (0 for padded, 1 for non-padded).
    local_count = (
        1 - paddings.float()).sum()  # Ensure paddings is float for subtraction

    # Aggregate the counts from all processes to get the total number of non-padded elements.
    if dist.is_initialized() and dist.get_world_size() > 1:
        total_count = local_count.clone()
        # Sum local_count from all processes.
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
    else:
        # If not distributed, the local count is the total count.
        # Clamp to minimum 1.0 to prevent division by zero later.
        total_count = local_count

    # Ensure total_count is at least 1.0 to prevent division by zero.
    # This is especially important if all inputs were padded.
    clamped_total_count = torch.clamp(total_count, min=1.0)

    # Calculate probabilities by dividing the histogram counts by the total count.
    # probs shape: [G, K]
    probs = hist / clamped_total_count

    # Calculate log probabilities, clamping probabilities to a small positive number
    # to avoid log(0) which results in -inf.
    log_probs = torch.log(torch.clamp(probs, min=1e-30))

    # Calculate sum(p * log(p)) for each code group.
    # This is related to the negative entropy of each group's code distribution.
    # sum_plogp shape: [G]
    sum_plogp = (probs * log_probs).sum(dim=-1)  # Sum over the K dimension

    # Perplexity for each group is exp(-sum(p*log(p))).
    # Then, compute the mean perplexity across all G groups.
    pplx = torch.exp(-sum_plogp).mean()

    # Entropy is log(perplexity).
    entropy = torch.log(torch.clamp(
        pplx, min=1e-30))  # Clamp pplx if it could be zero/negative

    return pplx, entropy


def compute_code_coverage(onehots: torch.Tensor,
                          paddings: torch.Tensor) -> torch.Tensor:
    """
    Computes codebook coverage as the average fraction of codes used per codebook.

    Args:
        onehots (torch.Tensor): A tensor of one-hot encoded codes.
                                Expected shape: [B, T, G, K].
        paddings (torch.Tensor): A boolean or integer tensor indicating padded elements.
                                 Expected shape: [B, T]. True or 1 for padded.

    Returns:
        torch.Tensor: The average codebook coverage (scalar, between 0 and 1).
    """
    # K (number of codes per group, i.e., codebook_size)
    codebook_size = onehots.shape[-1]

    # Get the global histogram of code occurrences.
    # histogram shape: [G, K] (num_codebooks, codebook_size)
    histogram = compute_code_histogram_allgather(onehots, paddings)

    # For each codebook (group G), count how many codes were used at least once.
    # (histogram > 0) creates a boolean tensor where True means the code was used.
    # .float() converts True/False to 1.0/0.0.
    # .sum(dim=-1) sums these 1.0s along the K dimension for each group G.
    # num_covered_words_per_group shape: [G]
    num_covered_words_per_group = (histogram > 0).float().sum(dim=-1)

    # Calculate the average number of covered words across all code groups.
    avg_num_covered_words = torch.mean(num_covered_words_per_group)

    # Coverage is the average number of covered words divided by the codebook size.
    # This gives a fraction representing the average portion of each codebook that was utilized.
    coverage = avg_num_covered_words / codebook_size
    return coverage


def _lookup(ids: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Codebook look up with ids.

    Args:
        ids: integer tensor of shape [..., num_codebooks] with values
            in range [0, codebook_size).
        codebook: Tensor of shape [codebook_size, num_codebooks, codebook_dim].

    Returns:
        * quantized_vectors: Tensor [..., num_codebooks, codebook_dim].

    """
    g_index = torch.arange(
        ids.shape[-1],
        dtype=ids.dtype,
        device=ids.device,
    )
    for _ in range(ids.dim() - 1):
        g_index = g_index.unsqueeze(0)
    return codebook[ids, g_index]


def quantize_by_nearest_neighbor(inputs: torch.Tensor, codebook: torch.Tensor,
                                 metric: SimilarityMetric):
    """Quantizes inputs by the nearest neighbor look-up in the codebook.

    This is used in both RandomVectorQuantizer and KmeansVectorQuantizer.

    Args:
        inputs: Tensor of shape [..., num_codebooks, codebook_dim].
        codebook: Tensor of shape [codebook_size, num_codebooks, codebook_dim].
        metric: similarity metric to rank the codebook. Choose from
            L2_DISTANCE or DOT_PRODUCT.

    Returns:
        ids: Tensor of shape [..., num_codebooks]
        quantized: Tensor of shape [..., num_codebooks, codebook_dim]
        onehots

    """
    batch_dims = inputs.shape[:-2]
    distance = -2 * torch.einsum(f"{batch_dims}gh,vgh->{batch_dims}vg", inputs,
                                 codebook)
    # l2_dist = (inputs - codebook) ** 2 = inputs ** 2 - 2 * input * codebook + codebook ** 2.
    # Since we do not compare distances across input vectors, we can drop the `input ** 2` term.
    # [..., vocab_size, num_codebooks].
    if metric == SimilarityMetric.L2_DISTANCE:
        distance += torch.sum(codebook**2, dim=-1)
    ids = torch.argmin(distance, dim=-2)
    # Note if the codebook is normalized, quantized_vectors is also normalized.
    return ids, _lookup(ids, codebook)


def _apply_paddings(ids: torch.Tensor, quantized_vectors: torch.Tensor,
                    paddings: torch.Tensor):
    """Applies paddings to quantizer outputs.

    ids are padded with -1. onehots and quantized_vectors are padded with 0s.

    Args:
        ids: [B, T, num_codebooks]
        quantized_vectors:[B, T, num_codebooks, codebook_dim]
        padding: [B, T]

    Returns:
        ids:
        quantized_vectors
    """

    # ids are padded with -1.
    ids_paddings = paddings[:, :, None].to(ids.dtype)
    ids = ids * (1 - ids_paddings)  # + (-1) * ids_paddings
    quantized_vectors = quantized_vectors * (1 - paddings)[:, :, None, None]
    return ids, quantized_vectors


def _ids_to_onehots(ids: torch.Tensor, codebook_size: int) -> torch.Tensor:
    # [..., num_codebooks, codebook_size].
    return torch.nn.functional.one_hot(ids, num_classes=codebook_size)


@dataclass
class WeightedScalar:
    """A weighted scalar represents a weighted Summable value.

    Weight should be a scalar and is assumed to be non-negative.
    A weight of zero corresponds to zero mean.
    """
    mean: Union[torch.Tensor, int, float]
    weight: Union[torch.Tensor, int, float]

    def __add__(self, other: "WeightedScalar") -> "WeightedScalar":
        weight = self.weight + other.weight
        if isinstance(weight, int) or isinstance(weight, float):
            if weight > 0:
                mean = (self.mean * self.weight +
                        other.mean * other.weight) / weight
            else:
                mean = 0.0
        else:
            mean = torch.where(
                weight > 0,
                (self.mean * self.weight + other.mean * other.weight) /
                torch.where(weight > 0, weight, 1),
                0.0,
            )
        return WeightedScalar(mean, weight)

    def accumulate(self, other):
        # TODO: overlap
        if not isinstance(other, WeightedScalar):
            raise TypeError(f"Expected WeightedScalar, got {type(other)}.")
        return self + other


def _add_codebook_summaries(onehots: torch.Tensor, paddings: torch.Tensor):
    """Helper function to compute codebook distribution statistics and add to summaries.

    The statistics are from all frames, not only on those masked frames in self-supervised training.

    Args:
        onehots: onehot of BaseQuantizer.Output.ids.
        paddings: 0/1 tensor of shape [batch_size, seq_len], where 0 is valid position.
    """
    coverage = compute_code_coverage(onehots=onehots, paddings=paddings)
    pplx, entropy = compute_code_pplx(onehots=onehots, paddings=paddings)
    batch_size = paddings.shape[0]

    num_frames = (torch.sum(1 - paddings)).clamp(min=1)
    return {
        "codebook/num_frames":
        WeightedScalar(num_frames.to(torch.float32) / batch_size, batch_size),
        "codebook/coverage":
        WeightedScalar(coverage, num_frames),
        "codebook/pplx":
        WeightedScalar(pplx, num_frames),
        "codebook/entropy":
        WeightedScalar(entropy, num_frames),
    }
