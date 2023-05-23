import torch
import numpy as np


class Simulated(torch.utils.data.Dataset):
    """
    A 1D simulated dataset of samples from a function f(x) with parameters given by
    w, a, b, c, d, sigma.
    """

    def __init__(
        self,
        n=1000,
        sigma=0.1,
        w=[-0.6667, -0.6012, -1.0172, -0.7687, 1.4680, -0.1678],
        a=np.pi,
        b=0,
        c=0.5,
        d=0,
        seed=1331,
    ):
        torch.manual_seed(seed)
        self.data = torch.randn(n, 1).float()
        self.targets = f(self.data, w=w, a=a, b=b, c=c, d=d, sigma=sigma).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class TensorDataset(torch.utils.data.Dataset):
    """
    Base class for datasets.
    """

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class SimulatedMeta(torch.utils.data.Dataset):
    """
    Dataset of *functions* f(x), where each function is sampled from a
    distribution over functions. Each member of the dataset is a set of samples from
    this function.
    num_samples : int
        Number of samples of the function contained in dataset.
    num_points : int
        Number of points at which to evaluate f(x) for each sample.
    TODO: add log scale option for ranges
    """

    def __init__(
        self,
        num_samples=1000,
        num_points=100,
        seed=1331,
        w_dim_range=(3, 6),
        w_range=(-1.0, 1.0),
        a_range=(0.5, 1.0),
        b_range=(-1.0, 1.0),
        c_range=(0.3, 0.7),
        d_range=(-1.0, 1.0),
        sigma_range=(0.0005, 0.005),
    ):
        torch.manual_seed(seed)
        self.num_samples = num_samples
        self.num_points = num_points
        self.a_range = a_range
        self.b_range = b_range
        self.c_range = c_range
        self.d_range = d_range
        self.sigma_range = sigma_range

        # Generate data
        self.data = []
        self.params = []

        # use the default function if num_samples == 1
        if num_samples == 1:
            x = torch.randn(num_points, 1)
            self.params = [
                {
                    "w": [-0.6667, -0.6012, -1.0172, -0.7687, 1.4680, -0.1678],
                    "a": np.pi,
                    "b": 0,
                    "c": 0.5,
                    "d": 0,
                    "sigma": 0.1,
                }
            ]
            self.data = [(x, f(x))]
        else:
            w_dim_min, w_dim_max = w_dim_range
            w_min, w_max = w_range
            a_min, a_max = a_range
            b_min, b_max = b_range
            c_min, c_max = c_range
            d_min, d_max = d_range
            sigma_min, sigma_max = sigma_range
            # Sample the function parameters
            for i in range(num_samples):
                a = (a_max - a_min) * torch.rand(1) + a_min
                b = (b_max - b_min) * torch.rand(1) + b_min
                c = (c_max - c_min) * torch.rand(1) + c_min
                d = (d_max - d_min) * torch.rand(1) + d_min
                sigma = (sigma_max - sigma_min) * torch.rand(1) + sigma_min
                w_dim = torch.randint(w_dim_min, w_dim_max + 1, (1,))
                w = (w_max - w_min) * torch.rand(w_dim) + w_min
                x = torch.randn(num_points, 1)
                y = f(x, w=w, a=a, b=b, c=c, d=d, sigma=sigma)
                self.params.append(
                    {"w": w, "a": a, "b": b, "c": c, "d": d, "sigma": sigma}
                )
                self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


def f(
    x,
    w=[-0.6667, -0.6012, -1.0172, -0.7687, 1.4680, -0.1678],
    a=np.pi,
    b=0,
    c=0.5,
    d=0,
    sigma=0.1,
):
    """
    Function of the form:
    f(x) = (w_0 * x^0 + w_1 * x^1 + ... + w_n * x^n)*sin(a*x - b)*exp(-c*(x-d)^2) + N(0, sigma)
    """
    w = torch.tensor(w)
    fx = 0
    for i in range(len(w)):
        fx += w[i] * (x**i)
    fx *= np.sin(a * (x - b))
    fx *= np.sqrt(c) * np.exp(-c * ((x - d) ** 2)) / np.sqrt(2 * np.pi)
    return (
        fx.squeeze(-1)
        + torch.randn(
            len(x),
        )
        * sigma
    )
