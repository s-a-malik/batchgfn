"""
Ref: https://github.com/fbickfordsmith/epig/blob/main/src/models/convolutional_nn.py
"""

from batchbald_redux.consistent_mc_dropout import (
    BayesianModule,
    ConsistentMCDropout,
    ConsistentMCDropout2d,
)
from batchbald_redux import joint_entropy
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm.autonotebook import trange
from typing import Sequence
import math

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from batchgfn.acquisitions.utils import compute_conditional_entropy


class ConvBlockMC(BayesianModule):
    def __init__(
        self, dropout_rate: float, n_in: int, n_out: int, kernel_size: int = 3
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=kernel_size
        )
        self.dropout = ConsistentMCDropout2d(p=dropout_rate)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.activation_fn = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch_in, H_in, W_in]
        Returns:
            Tensor[float], [N, Ch_out, H_out, W_out]
        """
        x = self.conv(x)
        x = self.dropout(x)
        x = self.maxpool(x)
        x = self.activation_fn(x)
        return x


class FullyConnectedBlockMC(BayesianModule):
    def __init__(self, dropout_rate: float, n_in: int, n_out: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=n_in, out_features=n_out)
        self.dropout = ConsistentMCDropout(p=dropout_rate)
        self.activation_fn = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, F_in]
        Returns:
            Tensor[float], [N, F_out]
        """
        x = self.fc(x)
        x = self.dropout(x)
        x = self.activation_fn(x)
        return x


class ConvolutionalNeuralNetwork(BayesianModule):
    """
    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/03_consistent_mc_dropout.ipynb
    """

    def __init__(
        self, input_shape: Sequence[int], output_size: int, dropout_rate: float
    ) -> None:
        n_input_channels, _, image_width = input_shape
        fc1_size = compute_conv_output_size(
            image_width,
            kernel_sizes=(2 * (5, 2)),
            strides=(2 * (1, 2)),
            n_output_channels=64,
        )
        super().__init__()
        self.block1 = ConvBlockMC(
            dropout_rate, n_in=n_input_channels, n_out=32, kernel_size=5
        )
        self.block2 = ConvBlockMC(dropout_rate, n_in=32, n_out=64, kernel_size=5)
        self.block3 = FullyConnectedBlockMC(dropout_rate, n_in=fc1_size, n_out=128)
        self.fc = nn.Linear(in_features=128, out_features=output_size)

    def mc_forward_impl(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor[float], [N, Ch, H, W]
        Returns:
            Tensor[float], [N, O]
        """

        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.block3(x)
        x = self.fc(x)
        return x

    def mutual_information(self, x, sample=True, num_samples=100):
        """Compute the joint mutual information
        of the samples in x.
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 4:
                x = x.unsqueeze(0)
            B, N, C, H, W = x.shape
            mi = torch.zeros(B, device=x.device)
            for batch in range(B):
                logits = self(x[batch], num_samples)  # (N, S, C)
                N, S, C = logits.shape
                logprobs = F.log_softmax(logits, dim=-1)
                batch_joint_entropy = joint_entropy.DynamicJointEntropy(
                    10000, N, S, C, dtype=x.dtype, device=x.device
                )
                batch_joint_entropy.add_variables(logprobs)
                batch_entropy = batch_joint_entropy.compute()
                conditional_entropy = compute_conditional_entropy(logprobs).sum()
                batch_mi = batch_entropy - conditional_entropy
                mi[batch] = batch_mi
            return mi

    def evaluate(
        self,
        generator,
        optimizer=None,
        scheduler=None,
        criterion=None,
        device="cpu",
        num_inference_samples=100,
        task="train",
    ):
        """Generic evaluation function for model. Iterates through generator once (1 epoch)
        Params:
        - model (nn.Module): model to eval
        - generator (DataLoader): data to eval on
        - optimizer (optim): torch optimizer for model
        - scheduler: LR scheduler
        - criterion: loss function
        - device: Cuda/CPU
        - num_inference_samples (int): number of samples to take for inference
        - task (str) = [train, val, test]: determines whether to take gradient steps etc.
        Returns:
        if test: results (test_pred, test_true, test_f1, test_prec, test_rec)
        if train/val: average loss and accuracy over batch
        """

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        if task == "test":
            self.eval()
            test_true = []
            test_pred = []
        elif task == "val":
            self.eval()
        elif task == "train":
            self.train()
        else:
            raise NameError("Only train, val or test is allowed as task")
        disable = task == "train"
        with trange(len(generator), desc="Epoch", disable=disable) as t:
            for batch in generator:
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                if task == "train":
                    preds = self(x, 1).squeeze(1)  # (batch_size, num_classes)
                    # compute loss
                    loss = criterion(preds, y)
                    loss_meter.update(loss.data.cpu().item(), y.size(0))
                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                else:
                    logits = self(
                        x, num_inference_samples
                    )  # (batch_size, num_samples, num_classes)
                    preds = F.log_softmax(
                        logits, dim=-1
                    )  # (batch_size, num_samples, num_classes)
                    preds = torch.logsumexp(preds, dim=1) - math.log(
                        num_inference_samples
                    )  # (batch_size, num_classes)
                    loss = F.nll_loss(preds, y)
                    loss_meter.update(loss.data.cpu().item(), y.size(0))
                # predictions
                y_preds = preds.max(1)[1]
                acc = torch.mean(y_preds.eq(y).float())
                acc_meter.update(acc.data.cpu().item(), y.size(0))

                if task == "test":
                    # collect the model outputs
                    test_true += y.detach().cpu().tolist()
                    test_pred += y_preds.detach().cpu().tolist()

                t.update()

        if task == "test":
            # per class metrics
            prec, rec, f1, sup = precision_recall_fscore_support(
                test_true, test_pred, average=None
            )
            return (
                loss_meter.avg,
                acc_meter.avg,
                prec,
                rec,
                f1,
                sup,
                test_pred,
                test_true,
            )
        else:
            return loss_meter.avg, acc_meter.avg


def compute_conv_output_size(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    n_output_channels: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    width = compute_conv_output_width(
        input_width, kernel_sizes, strides, padding, dilation
    )
    return n_output_channels * (width**2)


def compute_conv_output_width(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    References:
        https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    width = input_width
    for kernel_size, stride in zip(kernel_sizes, strides):
        width = width + (2 * padding) - (dilation * (kernel_size - 1)) - 1
        width = math.floor((width / stride) + 1)
    return width


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


## Autoencoder to learn latent representation of MNIST images for GFN


class ConvEncoder(nn.Module):
    """Convolutional Neural Network Encoder. Encodes image into a latent representation"""

    def __init__(self, latent_dim, kernel_size=3, strides=1):
        """Params:
        latent_dim - Dimensions of encoded representation
        kernel_size - kernel size for convolutions, default = 3
        strides - The stride for the convolutions, default = 1
        """
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=4,
            kernel_size=kernel_size,
            stride=strides,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=4,
            out_channels=4,
            kernel_size=kernel_size,
            stride=strides,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=4,
            out_channels=8,
            kernel_size=kernel_size,
            stride=strides,
            padding=1,
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(392, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # B, 4, 28, 28
        x = F.relu(self.conv2(x))  # B, 8, 28, 28
        x = self.pool(x)  # B, 4, 14, 14
        x = F.relu(self.conv3(x))  # B, 8, 14, 14
        x = self.pool(x)  # B, 8, 7, 7
        x = self.flatten(x)  # B, 392
        out = self.fc(x)  # B, latent
        return out


class ConvDecoder(nn.Module):
    """Convolutional Decoder
    Reconstructs an image from latent representation
    """

    def __init__(self, latent_dim, kernel_size=3, strides=1):
        """Params:
        latent_dim - Dimensions of encoded representation
        kernel_size - kernel size for convolutions, default = 3
        strides - The stride for the convolutions, default = 1
        """
        super(ConvDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.strides = strides

        self.fc = nn.Linear(latent_dim, 392)
        self.t_conv1 = nn.ConvTranspose2d(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            stride=strides * 2,
            padding=1,
            output_padding=1,
        )
        self.t_conv2 = nn.ConvTranspose2d(
            in_channels=4, out_channels=4, kernel_size=3, stride=strides, padding=1
        )
        self.t_conv3 = nn.ConvTranspose2d(
            in_channels=4,
            out_channels=1,
            kernel_size=3,
            stride=strides * 2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):
        x = self.fc(x)  # B, 392
        x = x.view(-1, 8, 7, 7)  # B, 8, 7, 7
        x = F.relu(self.t_conv1(x))  # B, 4, 14, 14
        x = F.relu(self.t_conv2(x))  # B, 4, 14, 14
        # sigmoid output so image pixels are in range (0,1)
        x = torch.sigmoid(self.t_conv3(x))  # B, 1, 28, 28
        return x


class CnnAutoencoder(pl.LightningModule):
    """
    Ref: https://lightning.ai/docs/pytorch/latest/notebooks/course_UvA-DL/08-deep-autoencoders.html
    """

    def __init__(
        self,
        latent_dim,
        kernel_size,
        strides,
        encoder_class: object = ConvEncoder,
        decoder_class: object = ConvDecoder,
        num_input_channels: int = 1,
        width: int = 28,
        height: int = 28,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(latent_dim, kernel_size, strides)
        self.decoder = decoder_class(latent_dim, kernel_size, strides)

    def forward(self, x, embedding=False):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        if embedding:
            return x_hat, z
        return x_hat

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        with torch.no_grad():
            x, _ = batch
            return self(x, embedding=True)

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        return {"optimizer": optimizer, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)
