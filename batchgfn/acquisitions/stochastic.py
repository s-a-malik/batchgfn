import time

import torch
import numpy as np
from numpy.linalg import cholesky, norm
from scipy.linalg import solve_triangular

from batchgfn.acquisitions.utils import (
    joint_mutual_information,
    compute_conditional_entropy,
)

from batchbald_redux import batchbald


def stochastic_batch_bald_gp(
    x_pool, batch_size, model, reps=100, temp=None, return_scores=False, task="reg"
):
    """
    x_pool: torch.Tensor (N, D)
    batch_size: int
    model: gpytorch model
    TODO currently mixing torch/numpy for reg/cls
    """
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    F = model.predict(x_pool)
    if task == "reg":
        cov = (
            F.covariance_matrix.detach().cpu().numpy()
            / model.model.likelihood.noise.detach().cpu().numpy()
        )
        D = np.diag(cov) + 1.0
        for _ in range(reps):
            idx = [np.argmax(D)]
            A = np.array([[D[idx[-1]]]])
            C = cov[:, idx]
            for _ in range(1, batch_size):
                L = cholesky(A)
                score = (
                    D - norm(solve_triangular(L, C.T, lower=True), ord=2, axis=0) ** 2
                )
                score[idx] = 0
                # sample proportional to score
                if temp is not None:
                    next_idx = np.random.choice(
                        len(score), p=np.exp(score / temp) / np.exp(score / temp).sum()
                    )
                else:
                    next_idx = np.random.choice(len(score), p=score / score.sum())
                idx.append(next_idx)
                A = np.pad(A, ((0, 1), (0, 1)), "constant", constant_values=0)
                A[-1, :-1] = C[idx[-1]]
                A[:-1, -1] = C[idx[-1]]
                A[-1, -1] = D[idx[-1]]
                C = cov[:, idx]
            # compute JMI
            jmi = model.mutual_information(x_pool[idx]).detach().cpu().numpy()
            if jmi > best_jmi:
                best_jmi = jmi
                best_idx = idx
            times.append(time.time() - start_time)
            jmis.append(best_jmi)
    elif task == "cls":
        raise NotImplementedError(
            "Stochastic batch BALD for classification not correct right now."
        )
    else:
        raise ValueError("Task must be either 'reg' or 'cls'.")

    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx


def stochastic_batch_bald(
    F, batch_size, variance=1.0, reps=100, temp=None, return_scores=False, task="reg"
):
    """
    Compute the batch BALD acquisition function using the covariance matrix.
    Stochastically acquire the batch by sampling from the acquisition function.
    Repeat the acquisition function sampling `reps` times and return the best.

    Parameters:
    -----------
    F: np.ndarray (n_samples, n_points) or (n_samples, n_points, n_classes)
    batch_size: int
    variance: float
    temp: float (temperature for softmax)
    return_scores: bool

    Returns:
    --------
    idx: list of int indices into F
    max_jmi: list[float] of the max JMI for each number of reps (if return_scores)
    times: list[float] of the time taken for each number of reps (if return_scores)
    """
    d = variance * (F.shape[0] - 1)
    F = F.astype(np.float64)
    F -= F.mean(0, keepdims=True)
    D = ((F**2).sum(0) / d) + 1.0
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    for _ in range(reps):

        if task == "reg":
            idx = [np.argmax(D)]
            A = np.array([[D[idx[-1]]]])
            C = F.T @ F[:, idx] / d
            for _ in range(1, batch_size):
                L = cholesky(A)
                score = (
                    D - norm(solve_triangular(L, C.T, lower=True), ord=2, axis=0) ** 2
                )
                score[idx] = 0
                # sample proportional to score
                if temp is not None:
                    next_idx = np.random.choice(
                        len(score), p=np.exp(score / temp) / np.exp(score / temp).sum()
                    )
                else:
                    next_idx = np.random.choice(len(score), p=score / score.sum())
                idx.append(next_idx)
                A = np.pad(A, ((0, 1), (0, 1)), "constant", constant_values=0)
                A[-1, :-1] = C[idx[-1]]
                A[:-1, -1] = C[idx[-1]]
                A[-1, -1] = D[idx[-1]]
                C = np.hstack([C, F.T @ F[:, idx[-1:]] / d])
        elif task == "cls":
            raise NotImplementedError(
                "Stochastic batch BALD for classification not correct right now."
            )
        # compute JMI
        jmi = joint_mutual_information(F[:, idx], variance, task=task)

        if jmi > best_jmi:
            best_jmi = jmi
            best_idx = idx
        times.append(time.time() - start_time)
        jmis.append(best_jmi)

    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx


def stochastic_bald(
    F, batch_size, variance=1.0, reps=100, temp=None, return_scores=False, task="reg"
):
    # F (n_samples, n_points, (n_classes)))
    if task == "reg":
        scores = np.sqrt(1 + F.var(0) / variance)
    elif task == "cls":
        F = F.permute(1, 0, 2)  # (n_points, n_samples, n_classes)
        logits = torch.nn.LogSoftmax(dim=-1)(F)
        scores = -compute_conditional_entropy(logits)
        scores += batchbald.compute_entropy(logits)
    else:
        raise ValueError("task must be 'reg' or 'cls'")
    if temp is not None:
        p = np.exp(scores / temp) / np.exp(scores / temp).sum()
    else:
        p = scores / scores.sum()
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    for _ in range(reps):
        idx = list(np.random.choice(len(p), replace=False, p=p, size=batch_size))
        if task == "reg":
            jmi = joint_mutual_information(F[..., idx], variance, task=task)
        else:
            jmi = joint_mutual_information(F[idx], variance, task=task)
        if jmi > best_jmi:
            best_jmi = jmi
            best_idx = idx
        times.append(time.time() - start_time)
        jmis.append(best_jmi)

    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx


def stochastic_bald_gp(
    x_pool, batch_size, model, reps=100, temp=None, return_scores=False, task="reg"
):
    F = model.predict(x_pool)
    if task == "reg":
        scores = F.variance.detach().cpu().numpy()
    elif task == "cls":
        scores = (
            F.variance.detach().cpu().numpy().sum(-1)
        )  # assuming independent classes
    else:
        raise ValueError("task must be 'reg' or 'cls'")

    if temp is not None:
        p = np.exp(scores / temp) / np.exp(scores / temp).sum()
    else:
        p = scores / scores.sum()
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    for _ in range(reps):
        idx = list(np.random.choice(len(p), replace=False, p=p, size=batch_size))
        jmi = model.mutual_information(x_pool[idx]).detach().cpu().numpy()
        if jmi > best_jmi:
            best_jmi = jmi
            best_idx = idx
        times.append(time.time() - start_time)
        jmis.append(best_jmi)

    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx


def random_multi_gp(
    x_pool, labelled_idx, model, batch_size, reps=100, return_scores=False, task="reg"
):
    unlabelled_idx = [i for i in range(len(x_pool)) if i not in labelled_idx]
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    for _ in range(reps):
        idx = list(np.random.choice(unlabelled_idx, size=batch_size, replace=False))
        jmi = model.mutual_information(x_pool[idx]).detach().cpu().numpy()
        if jmi > best_jmi:
            best_jmi = jmi
            best_idx = idx
        times.append(time.time() - start_time)
        jmis.append(best_jmi)
    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx


def random_multi(
    F, labelled_idx, batch_size, variance=1.0, reps=100, return_scores=False, task="reg"
):
    unlabelled_idx = [i for i in range(F.shape[-1]) if i not in labelled_idx]
    best_idx = None
    best_jmi = -np.inf
    jmis = []
    start_time = time.time()
    times = []
    for _ in range(reps):
        idx = list(np.random.choice(unlabelled_idx, size=batch_size, replace=False))
        jmi = joint_mutual_information(F[..., idx], variance, task=task)
        if jmi > best_jmi:
            best_jmi = jmi
            best_idx = idx
        times.append(time.time() - start_time)
        jmis.append(best_jmi)
    if return_scores:
        return best_idx, jmis, times
    else:
        return best_idx
