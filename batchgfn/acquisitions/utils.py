import numpy as np
from toma import toma

import torch
import torch.nn as nn

from batchbald_redux import joint_entropy, batchbald


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    """Same as batchbald_redux.compute_conditional_entropy, but without tqdm progress bar."""
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)

    return entropies_N


def joint_mutual_information(F, variance=1.0, task="reg"):
    # F: (S, C, N) or (S, N)
    if task == "reg":
        K = torch.cov(F.T) / variance
        K += torch.eye(K.shape[0])
        return 0.5 * torch.slogdet(K)[-1]
    elif task == "cls":
        # assuming tensor
        S, C, N = F.shape
        F = F.permute(2, 0, 1)  # (N, S, C)
        logits = nn.LogSoftmax(dim=-1)(F)
        batch_joint_entropy = joint_entropy.DynamicJointEntropy(
            10000, N, S, C, dtype=F.dtype, device=F.device
        )
        batch_joint_entropy.add_variables(logits)
        batch_entropy = batch_joint_entropy.compute()
        conditional_entropy = compute_conditional_entropy(logits).sum()
        batch_mi = batch_entropy - conditional_entropy
        return batch_mi.cpu().float()
    else:
        raise NotImplementedError(f"Task {task} not implemented.")
