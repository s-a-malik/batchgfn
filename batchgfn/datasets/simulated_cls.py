import torch
import numpy as np


class SimulatedCls(torch.utils.data.Dataset):
    """
    Simulated dataset for classification
    """

    def __init__(
        self,
        num_classes=2,
        num_samples_per_class=1000,
        minority_class_ratio=1.0,
        seed=2019,
    ):
        """
        Args:
            num_classes (int): number of classes
            num_samples_per_class (int): number of samples per class
            minority_class_ratio (float): ratio of minority class samples to total samples
            seed (int): random seed
        """
        self.num_classes = num_classes
        self.num_samples_per_class = num_samples_per_class
        self.minority_class_ratio = minority_class_ratio
        self.seed = seed

        self.mean_list = []
        self.covariance_list = []

        # generate mean and covariance for each class (fixed seed for reproducibility)
        torch.manual_seed(4)
        for i in range(num_classes):
            mean = torch.rand(size=(2,)) * 10 - 5
            covariance = torch.diag((0.1 - 0.5) + torch.rand(size=(2,)) + 0.5)
            self.mean_list.append(mean)
            self.covariance_list.append(covariance)

        minority_samples = int(minority_class_ratio * num_samples_per_class)

        self.data = []
        self.targets = []

        # generate samples for each class
        torch.manual_seed(seed)  # use given seed to generate samples
        for i in range(num_classes - 1):
            m = torch.distributions.MultivariateNormal(
                self.mean_list[i], self.covariance_list[i]
            )
            self.data.append(m.sample((num_samples_per_class,)))
            self.targets.append(torch.ones(num_samples_per_class, dtype=torch.long) * i)

        # generate samples for minority class
        m = torch.distributions.MultivariateNormal(
            self.mean_list[-1], self.covariance_list[-1]
        )
        self.data.append(m.sample((minority_samples,)))
        self.targets.append(
            torch.ones(minority_samples, dtype=torch.long) * (num_classes - 1)
        )
        self.data = torch.cat(self.data)
        self.targets = torch.cat(self.targets).reshape(-1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        data = self.data[idx]
        targets = self.targets[idx]
        return data, targets
