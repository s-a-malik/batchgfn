import numpy as np
import torch
from functools import partial
from pathlib import Path
from numpy.random import Generator
from scipy.stats import logistic, multivariate_t

from typing import Callable, Union


class TwoBells(torch.utils.data.Dataset):
    """
    Synthetic dataset with two-dimensional inputs and binary class labels.
    Ref: https://github.com/fbickfordsmith/epig/tree/main/src/datasets
    """

    def __init__(
        self,
        n_train: int = 1000,
        n_test: int = 2000,
        input_scale: float = 2.0,
        latent_scale: float = 1.0,
        label_scale: float = 0.05,
        seed: int = 0,
        shift: bool = False,
        train: bool = True,
    ) -> None:
        def f(x: np.ndarray, input_scale: float, latent_scale: float) -> np.ndarray:
            return latent_scale * np.tanh(input_scale * x)

        def classify(
            x: np.ndarray, f: Callable, label_scale: float, rng: Generator
        ) -> np.ndarray:
            probs = logistic.cdf((x[:, 1] - f(x[:, 0])) / label_scale)
            return rng.binomial(n=np.ones_like(probs, dtype=int), p=probs)

        rng = np.random.default_rng(seed=seed)

        f = partial(f, input_scale=input_scale, latent_scale=latent_scale)

        if train or (not shift):
            x = 0
            input_dist = multivariate_t(loc=[x, f(x)], shape=0.8, df=5)
        else:
            x = np.pi / (2 * input_scale)
            input_dist = multivariate_t(loc=[x, f(x)], shape=0.2, df=5)

        n_samples = n_train if train else n_test

        inputs = input_dist.rvs(size=n_samples, random_state=rng)
        inputs = inputs.astype(np.float32)

        labels = classify(inputs, f, label_scale, rng)

        self.input_dist = input_dist
        self.data = torch.tensor(inputs)
        self.targets = torch.tensor(labels)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.targets[index]

    def __len__(self) -> int:
        return len(self.data)
