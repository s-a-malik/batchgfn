import numpy as np
import torch
import torch.nn as nn

from batchbald_redux import batchbald


def random(F, batch_size, task="reg"):
    return np.random.choice(len(F), batch_size, replace=False)


def bald(F, batch_size, task="reg"):
    """F: (K, N, C)"""
    if task == "reg":
        return list(np.argsort(F.std(0))[-batch_size:])
    elif task == "cls":
        F = F.permute(1, 0, 2)  # (N, K, C)
        logits = nn.LogSoftmax(dim=-1)(F)
        candidate_batch = batchbald.get_bald_batch(
            logits, batch_size, dtype=torch.float64, device=F.device
        )
        return candidate_batch.indices
    else:
        raise ValueError("Task must be either 'reg' or 'cls'.")


def bald_gp(x_pool, batch_size, model, task="reg"):
    F = model.predict(x_pool)
    if task == "reg":
        return list(np.argsort(F.variance.detach().cpu().numpy())[-batch_size:])
    elif task == "cls":
        return list(np.argsort(F.variance.detach().cpu().numpy().sum(-1))[-batch_size:])
    else:
        raise ValueError("Task must be either 'reg' or 'cls'.")
