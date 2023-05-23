import time

import numpy as np
import torch
from numpy.linalg import cholesky, norm
from scipy.linalg import solve_triangular

from batchbald_redux import batchbald


def batch_bald_cov(
    F, batch_size, variance=1.0, num_batchbald_samples=10000, task="reg"
):
    """
    F: np.ndarray (K,N,(C))
    batch_size: int
    eps: floa
    """
    if task == "reg":
        d = variance * (F.shape[0] - 1)
        F -= F.mean(0, keepdimså=True)
        D = ((F**2).sum(0) / d) + 1.0
        F = F.astype(np.float64)
        D = D.astype(np.float64)
        idx = [np.argmax(D)]
        A = np.array([[D[idx[-1]]]])
        C = F.T @ F[:, idx] / d
        for _ in range(1, batch_size):
            L = cholesky(A)
            score = D - norm(solve_triangular(L, C.T, lower=True), ord=2, axis=0) ** 2
            score[idx] = -np.inf
            idx.append(np.argmax(score))
            A = np.pad(A, ((0, 1), (0, 1)), "constant", constant_values=0)
            A[-1, :-1] = C[idx[-1]]
            A[:-1, -1] = C[idx[-1]]
            A[-1, -1] = D[idx[-1]]
            C = np.hstack([C, F.T @ F[:, idx[-1:]] / d])
    elif task == "cls":
        # using batchbald redux
        F = F.permute(1, 0, 2)  # (N, K, C)
        logprobs = torch.nn.LogSoftmax(dim=-1)(F)
        candidate_batch = batchbald.get_batchbald_batch(
            logprobs,
            batch_size,
            num_samples=num_batchbald_samples,
            dtype=torch.float64,
            device=logprobs.device,
        )
        idx = candidate_batch.indices
        scores = candidate_batch.scores

    return idx


def batch_bald_cov_gp(x_pool, batch_size, model, task="reg"):
    """
    x_pool: torch tensors
    batch_size: int
    eps: float
    """
    F = model.predict(x_pool)
    if task == "reg":
        cov = (
            F.covariance_matrix.detach().cpu().numpy()  # (N, N)
            / model.model.likelihood.noise.detach().cpu().numpy()
        )
        D = np.diag(cov) + 1.0  # (N,)
        idx = [np.argmax(D)]  # (1,)
        A = np.array([[D[idx[-1]]]])  # (1, 1)
        C = cov[:, idx]  # (N, 1)
        for _ in range(1, batch_size):
            L = cholesky(A)  # (1, 1)
            score = (
                D - norm(solve_triangular(L, C.T, lower=True), ord=2, axis=0) ** 2
            )  # (N,)
            score[idx] = -np.inf
            idx.append(np.argmax(score))  # (i+1,)
            A = np.pad(A, ((0, 1), (0, 1)), "constant", constant_values=0)  # (i+1, i+1)
            A[-1, :-1] = C[idx[-1]]
            A[:-1, -1] = C[idx[-1]]
            A[-1, -1] = D[idx[-1]]
            C = cov[:, idx]  # (N, i+1)
        return idx
    else:
        raise ValueError(
            "task should be 'reg'. Exact GP version not implemented for classification."
        )
