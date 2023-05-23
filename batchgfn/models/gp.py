import torch
import torch.nn as nn
import gpytorch
import pytorch_lightning as pl

from batchbald_redux import joint_entropy, batchbald

from batchgfn.acquisitions import compute_conditional_entropy

# create a gpytorch gaussian process model
class GPModel(gpytorch.models.ExactGP):
    """
    if len(batch shape) == 0, then this is a single task GP, else
    this is a multitask GP (with independent tasks)
    """

    def __init__(self, train_x, train_y, likelihood, kernel, batch_shape=[]):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.batch_shape = batch_shape
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            kernel, batch_shape=batch_shape
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if len(self.batch_shape) == 0:
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        else:
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )


# create a gpytoch gaussian process model as a pytorch lightning module
class GP(pl.LightningModule):
    def __init__(self, train_x, train_y, task="reg", num_classes=1, kernel_str="rbf"):
        super().__init__()
        self.task = task
        self.num_classes = num_classes
        self.kernel_str = kernel_str
        batch_shape = torch.Size((num_classes,)) if task == "cls" else []
        kernel_dict = {
            "rbf": gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.shape[-1], batch_shape=batch_shape
            ),
            "matern": gpytorch.kernels.MaternKernel(nu=2.5, batch_shape=batch_shape),
            "periodic": gpytorch.kernels.PeriodicKernel(batch_shape=batch_shape),
            "linear": gpytorch.kernels.LinearKernel(batch_shape=batch_shape),
            # Add more kernels here as needed
        }
        if task == "reg":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            targets = train_y
        elif task == "cls":
            # treat as a multi-task regression problem
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=num_classes, rank=0
            )
            # one hot encode the targets
            targets = torch.nn.functional.one_hot(
                train_y, num_classes=num_classes
            ).float()
        else:
            raise NotImplementedError(f"Task {task} not implemented.")

        self.model = GPModel(
            train_x,
            targets,
            likelihood,
            kernel_dict[kernel_str],
            batch_shape=batch_shape,
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )

    def forward(self, x):
        return self.model(x)

    def sample_mean(self, x, num_samples=100):
        self.model.eval()
        with torch.no_grad():
            f_samples = self.model(x).sample(torch.Size([num_samples]))
            return f_samples

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    def confidence_region(self, x):
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad():
            return self.model.likelihood(self.model(x)).confidence_region()

    def mutual_information(self, x, sample=False, num_samples=100):
        """
        MI between x and model parameters.
        Inputs:
            x: torch.Tensor (B, N, D)
            sample: bool, whether to sample (otherwise exact covariance from GP)
            num_samples: int, number of samples to use if sample=True
        Outputs:
            torch.Tensor (B,): mutual information between x and model parameters for each batch
        """
        self.model.eval()
        self.model.likelihood.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            if self.task == "reg":
                if sample:
                    F = self.model(x).sample(torch.Size([num_samples]))
                    S, B, N = F.shape
                    cov_matrices = []
                    # TODO can batch this
                    for i in range(B):
                        sample_cov = torch.cov(F[:, i, :].T)
                        cov_matrices.append(sample_cov)
                    K = (
                        torch.stack(cov_matrices) / self.model.likelihood.noise
                    )  # TODO can use noise for sampled MI?
                else:
                    K = self.model(x).covariance_matrix / self.model.likelihood.noise
                K += (
                    torch.eye(K.shape[-1]).to(K.device).float()
                )  # allows for batched inputs
                return 0.5 * torch.slogdet(K)[-1]
            elif self.task == "cls":
                B, N, D = x.shape
                cov_matrices = []
                # TODO can vectorise this
                for batch in range(B):
                    if sample:
                        F = self.model(x[batch]).sample(
                            torch.Size([num_samples])
                        )  # (S, N, C)
                        sample_cov = torch.stack(
                            [torch.cov(F[..., i].T) for i in range(F.shape[-1])]
                        )  # (C, N, N)
                        cov_matrices.append(sample_cov)
                    else:
                        cov = self.model(
                            x[batch]
                        ).covariance_matrix  # (NxC, NxC) interleaved
                        # seperate out the covariances of each task
                        task_covs = []
                        for i in range(self.num_classes):
                            task_covs.append(
                                cov[i :: self.num_classes, i :: self.num_classes]
                            )
                        task_covs = torch.stack(task_covs)  # (C, N, N)
                        cov_matrices.append(task_covs)
                K = (
                    torch.stack(cov_matrices) / self.model.likelihood.noise
                )  # (B, C, N, N)
                K += torch.eye(K.shape[-1]).to(K.device).float()
                return 0.5 * torch.sum(
                    torch.slogdet(K)[-1], dim=-1
                )  # (B,)    # block diagonal covariance matrix
            else:
                raise NotImplementedError(f"Task {self.task} not implemented.")

    def training_step(self, batch, batch_idx):
        self.model.train()
        self.model.likelihood.train()
        output = self.model(batch[0])
        loss = -self.mll(output, batch[1])
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                self.model.train_inputs[0], self.model.train_targets
            ),
            batch_size=len(self.model.train_targets),
            shuffle=False,
        )
