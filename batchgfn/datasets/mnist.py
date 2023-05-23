"""Reference:
https://github.com/BlackHC/batchbald_redux/blob/master/batchbald_redux/repeated_mnist.py
"""

import os

import numpy as np

import torch
from torchvision import datasets, transforms


class TransformedDataset(torch.utils.data.Dataset):
    """
    Transforms a dataset.

    Arguments:
        dataset (Dataset): The whole Dataset
        transformer (LambdaType): (idx, sample) -> transformed_sample
    """

    def __init__(self, dataset, *, transformer=None, vision_transformer=None):
        self.dataset = dataset
        assert not transformer or not vision_transformer
        if transformer:
            self.transformer = transformer
        else:
            self.transformer = lambda _, data_label: (
                vision_transformer(data_label[0]),
                data_label[1],
            )

    def __getitem__(self, idx):
        return self.transformer(idx, self.dataset[idx])

    def __len__(self):
        return len(self.dataset)


def create_modified_MNIST_dataset(
    *,
    num_classes: int = 10,
    num_samples_per_class: int = -1,
    minority_class_ratio: float = 1.0,
    num_repetitions: int = 3,
    add_noise: bool = True
):
    """
    Create MNIST dataset with num_classes classes, num_samples_per_class samples per class (-1, leave as is), and
    minority_class_ratio*num_samples_per_class samples in the 0th class.
    Repeat each sample num_repetitions times.
    If add_noise, add Gaussian noise to each sample.
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    if num_classes < 10:

        def restrict_classes(dataset, num_classes):
            """
            Reduce dataset to only include labels 0-(num_classes-1)
            """
            inds_to_keep = np.flatnonzero(
                [label in range(num_classes) for label in dataset.targets]
            )
            dataset.data = dataset.data[inds_to_keep]
            dataset.targets = dataset.targets[inds_to_keep]
            return dataset

        train_dataset = restrict_classes(train_dataset, num_classes)
        # also restrict test dataset
        test_dataset = restrict_classes(test_dataset, num_classes)

    if num_samples_per_class != -1:

        def subsample_classes(dataset, num_samples_per_class, minority_class_ratio):
            """
            Subsample dataset to only include num_samples_per_class samples per class, randomly
            and have the 0th class have minority_class_ratio*num_samples_per_class samples
            """
            inds_to_keep = []
            for i in range(num_classes):
                inds = np.flatnonzero([label == i for label in dataset.targets])
                np.random.shuffle(inds)
                if i == 0:
                    inds = inds[: int(minority_class_ratio * num_samples_per_class)]
                else:
                    inds = inds[:num_samples_per_class]
                inds_to_keep += inds.tolist()
            dataset.data = dataset.data[inds_to_keep]
            dataset.targets = dataset.targets[inds_to_keep]
            return dataset

        train_dataset = subsample_classes(
            train_dataset, num_samples_per_class, minority_class_ratio
        )

    if num_repetitions > 1:
        train_dataset = torch.utils.data.ConcatDataset(
            [train_dataset] * num_repetitions
        )

    if add_noise:
        dataset_noise = torch.empty(
            (len(train_dataset), 28, 28), dtype=torch.float32
        ).normal_(0.0, 0.1)

        def apply_noise(idx, sample):
            data, target = sample
            return data + dataset_noise[idx], target

        train_dataset = TransformedDataset(train_dataset, transformer=apply_noise)

    return train_dataset, test_dataset


def create_MNIST_dataset():
    return create_modified_MNIST_dataset(num_repetitions=1, add_noise=False)


def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.

    This supports subsets and other derivative datasets."""
    if isinstance(dataset, TransformedDataset):
        return get_targets(dataset.dataset)
    if isinstance(dataset, torch.utils.data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)


def get_data(dataset):
    """Get all the tensors of a dataset, with the transforms applied."""
    data = []
    targets = []
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)
    for batch in dataloader:
        data.append(batch[0])
        targets.append(batch[1])
    return torch.cat(data), torch.cat(targets)
