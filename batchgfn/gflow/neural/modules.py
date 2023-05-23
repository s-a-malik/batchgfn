"""
Modules for function estimation for GFN models.
"""

from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gfn.modules import GFNModule

from batchgfn.gflow.neural.base import MLPEncoder
from batchgfn.gflow.neural.npt import get_attention_mask


class StateNet(nn.Module, GFNModule):
    """Implements a basic MLP which takes a state as input
    We want to be set invariant so we sum over embeddings of the acquired points so far.
    Args:
        feature_dim (int): number of features for each item in the pool
        hidden_dim (Optional[int], optional): Number of units per hidden layer. Defaults to 256.
        n_hidden_layers (Optional[int], optional): Number of hidden layers. Defaults to 2.
        activation_fn (Optional[Literal[relu, tanh]], optional): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        gfn_base_module: str,
        feature_dim: int,
        label_dim: int,
        output_dim: Optional[int] = 1,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[tuple] = None,
    ):
        super().__init__()
        self._output_dim = output_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.label_dim = label_dim
        self.gfn_base_module = gfn_base_module

        # TODO use gfn_module to check for what input should be from preprocessor

        assert (
            n_hidden_layers is not None and n_hidden_layers >= 0
        ), "n_hidden_layers must be >= 0"
        assert torso is not None, "torso must be provided for state net"

        self.torso = torso
        if self.gfn_base_module == "npt":
            self.rff = torso[0]
            self.label_to_embedding_dim = torso[1]
            self.npt = torso[2]
            combined_dim = self.npt.dim_hidden
        else:
            self.batch_embed = torso[0]
            combined_dim = hidden_dim
            if self.gfn_base_module == "poolandtrainnet":
                self.train_embed = torso[2]
                combined_dim = 2 * hidden_dim

        self.combined_embed = MLPEncoder(
            input_dim=combined_dim,
            hidden_dim=hidden_dim,
            encoding_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation_fn=activation_fn,
        )

        self.last_layer = nn.Linear(self.hidden_dim, self._output_dim)

        self.device = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, query_size, feature_dim)
        Returns:
            torch.Tensor: (batch_size, pool_size+1)
            last column is the probability of not acquiring any point (i.e. stopping)
        """
        if self.device is None:
            if isinstance(x, tuple):
                self.device = x[0].device
            else:
                self.device = x.device
            self.to(self.device)

        # unpack input
        if self.gfn_base_module == "poolandtrainnet":
            (
                pool,
                labels,
                batch_mask,
                current_dataset,
            ) = x
            batch_embed = self.batch_embed(pool)  # (pool_size, hidden_dim)
            sum_encoded_batch = torch.sum(
                batch_embed * batch_mask.unsqueeze(-1), dim=1
            )  # (batch_size, hidden_dim)
            # train embeddings
            pool_cat_labels = torch.cat(
                (pool, labels), dim=-1
            )  # (pool_size, feature_dim + label_dim)
            train_embed = self.train_embed(pool_cat_labels)  # (pool_size, hidden_dim)
            sum_encoded_train = torch.sum(
                train_embed * current_dataset.unsqueeze(-1), dim=0
            )  # (hidden_dim,)
            # repeat for each batch
            sum_encoded_train = sum_encoded_train.repeat(
                batch_mask.shape[0], 1
            )  # (batch_size, hidden_dim)
            embedded_state = torch.cat(
                (sum_encoded_batch, sum_encoded_train), dim=-1
            )  # (batch_size, 2*hidden_dim)
        elif self.gfn_base_module == "poolnet":
            pool, batch_mask = x
            batch_embed = self.batch_embed(pool)  # (pool_size, hidden_dim)
            embedded_state = torch.sum(
                batch_embed * batch_mask.unsqueeze(-1), dim=1
            )  # (batch_size, hidden_dim)
        elif self.gfn_base_module == "querynet":
            batch_embed = self.batch_embed(x)  # (batch_size, query_size, hidden_dim)
            embedded_state = torch.sum(batch_embed, dim=1)  # (batch_size, hidden_dim)
        elif self.gfn_base_module == "npt":
            (
                pool,
                labels,
                batch_mask,
                current_dataset,
            ) = x  # (pool_size, feature_dim), (pool_size, label_dim), (batch_size, pool_size), (pool_size,)

            # Embed input features
            feature_embeddings = self.rff(pool)  # (pool_size, cm.dim_feat_hidden)
            feature_embeddings = feature_embeddings.repeat(  # (batch_size, pool_size, cm.dim_feat_hidden)
                batch_mask.shape[0], 1, 1
            )

            # Mask out the target values at the mask indices
            labels[current_dataset == 0, :] = 0

            labels = labels.repeat(
                batch_mask.shape[0], 1, 1
            )  # (batch_size, pool_size, label_dim)
            training_indictators = torch.zeros(
                (labels.shape[0], labels.shape[1], 1), device=labels.device
            )
            training_indictators[
                :, current_dataset > 0, :
            ] = 1  # (batch_size, pool_size, 1)
            batch_indictators = batch_mask.unsqueeze(2)  # (batch_size, pool_size, 1)

            attention_mask = get_attention_mask(
                training_indictators.squeeze(),
                batch_indictators.squeeze(),
                self.npt.sab_args.num_heads,
                device=labels.device,
            )

            masked_targets = torch.cat(
                (labels, training_indictators, batch_indictators), dim=2
            )  # (batch_size, pool_size, label_dim + 2)
            target_embeddings = self.label_to_embedding_dim(masked_targets)

            embeddings = torch.cat(
                (feature_embeddings, target_embeddings), dim=2
            )  # (batch_size, pool_size, cm.dim_feat_hidden + cm.dim_label_hidden)

            # NPT treats first dim as batch, performs attention over the second dim
            npt_out = self.npt(
                embeddings, attention_mask
            )  # (batch_size, pool_size, combined_dim)

            # sum over pool
            embedded_state = torch.sum(npt_out, dim=1)  # (batch_size, combined_dim)

        else:
            raise ValueError(f"Invalid gfn_base_module: {self.gfn_base_module}")

        # pass it through a MLP
        hid = self.combined_embed(embedded_state)  # (batch_size, hidden_dim)
        logits = self.last_layer(hid)  # (batch_size, output_dim)
        return logits

    @property
    def output_dim(self) -> int:
        return self._output_dim


class PoolAndTrainNet(nn.Module, GFNModule):
    """Implements a basic MLP which outputs logits for acquiring each of the pool items.
    We want to be set invariant so we sum over embeddings of the acquired points so far, and also the labelled dataset
    Then we concatenate the summed embeddings with the embeddings of each of the points in the pool and pass it through an MLP.
    TODO: add dropout, batchnorm, etc.
    Args:
        feature_dim (int): number of features for each item in the pool
        label_dim (int): number of dimensions of the label (normally 1)
        pool_size (int): size of the pool (output size)
        output_final_state (bool, optional): Whether to output the final state of the pool. Defaults to True.
        hidden_dim (Optional[int], optional): Number of units per hidden layer. Defaults to 256.
        n_hidden_layers (Optional[int], optional): Number of hidden layers. Defaults to 2.
        activation_fn (Optional[Literal[relu, tanh]], optional): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        feature_dim: int,
        label_dim: int,
        pool_size: int,
        output_final_state: bool = True,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[tuple] = None,
    ):
        super().__init__()
        self.output_final_state = output_final_state
        # TODO make this general for any pool size - can infer pool size from input dimensions.
        self._output_dim = pool_size + 1 if output_final_state else pool_size
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size

        assert (
            n_hidden_layers is not None and n_hidden_layers >= 0
        ), "n_hidden_layers must be >= 0"
        assert activation_fn is not None, "activation_fn must be provided"

        if torso is not None:
            self.torso = torso
            self.batch_embed = torso[0]
            self.combined_embed = torso[1]
            self.train_embed = torso[2]
            self.pool_embed = torso[3]
        else:
            self.batch_embed = MLPEncoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.pool_embed = MLPEncoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.train_embed = MLPEncoder(
                input_dim=feature_dim + label_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.combined_embed = MLPEncoder(
                input_dim=hidden_dim * 3,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.torso = (
                self.batch_embed,
                self.combined_embed,
                self.train_embed,
                self.pool_embed,
            )

        self.last_layer = nn.Linear(self.hidden_dim, 1)

        self.device = None

    def forward(self, x):
        """
        Args:
            x (tuple): tuple containing:
                pool (torch.Tensor): (pool_size, feature_dim)
                labels (torch.Tensor): (pool_size, [label_dim])
                batch_mask (torch.Tensor): (batch_size, pool_size)
                current_dataset (torch.Tensor): (pool_size) 1 if in current dataset, 0 otherwise
        Returns:
            torch.Tensor: (batch_size, pool_size+1)
            last column is the probability of not acquiring any point (i.e. stopping)
        """
        (
            pool,
            labels,
            batch_mask,
            current_dataset,
        ) = x  # (pool_size, feature_dim), (pool_size, label_dim), (batch_size, pool_size), (pool_size,)
        if self.device is None:  # from gfn package
            self.device = batch_mask.device
            self.to(self.device)

        # batch embeddings
        batch_embed = self.batch_embed(pool)  # (pool_size, hidden_dim)
        sum_encoded_batch = torch.sum(
            batch_embed * batch_mask.unsqueeze(-1), dim=1
        )  # (batch_size, hidden_dim)
        # repeat for each point in the pool
        sum_encoded_batch = sum_encoded_batch.repeat(1, self._output_dim).view(
            -1, self._output_dim, self.hidden_dim
        )  # (batch_size, pool_size (+1), hidden_dim)

        # train embeddings
        pool_cat_labels = torch.cat(
            (pool, labels), dim=-1
        )  # (pool_size, feature_dim + label_dim)
        train_embed = self.train_embed(pool_cat_labels)  # (pool_size, hidden_dim)
        sum_encoded_train = torch.sum(
            train_embed * current_dataset.unsqueeze(-1), dim=0
        )  # (hidden_dim,)
        # repeat for each batch
        sum_encoded_train = sum_encoded_train.repeat(
            batch_mask.shape[0], self._output_dim, 1
        )

        # pool embeddings
        if self.output_final_state:
            # add a column of zeros to the pool to represent the probability of not acquiring any point
            pool = torch.cat(
                (pool, torch.zeros((1, pool.shape[1]), device=batch_mask.device)), dim=0
            )  # (pool_size+1, feature_dim)
        pool_embed = self.pool_embed(pool)  # (pool_size(+1), hidden_dim)
        # repeat pool embeddings for each batch
        pool_embed = pool_embed.repeat(
            batch_mask.shape[0], 1, 1
        )  # (batch_size, pool_size(+1), hidden_dim)

        # concatenate
        pool_cat_batch_cat_train = torch.cat(
            (pool_embed, sum_encoded_batch, sum_encoded_train), dim=-1
        )  # (batch_size, pool_size, hidden_dim*3)
        # pass it through a MLP
        hid = self.combined_embed(
            pool_cat_batch_cat_train
        )  # (batch_size, pool_size, hidden_dim)
        logits = self.last_layer(hid)  # (batch_size, pool_size, 1)
        return logits.squeeze(-1)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class PoolNet(nn.Module, GFNModule):
    """Implements a basic MLP which outputs logits for acquiring each of the pool items.
    We want to be set invariant so we sum over embeddings of the acquired points so far.
    Then we concatenate the summed embeddings with the embeddings of each of the points in the pool and pass it through an MLP.
    TODO: add dropout, batchnorm, etc.
    Args:
        feature_dim (int): number of features for each item in the pool
        pool_size (int): size of the pool (output size)
        output_final_state (bool, optional): Whether to output the probability of not acquiring any point. Defaults to True.
        hidden_dim (Optional[int], optional): Number of units per hidden layer. Defaults to 256.
        n_hidden_layers (Optional[int], optional): Number of hidden layers. Defaults to 2.
        activation_fn (Optional[Literal[relu, tanh]], optional): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        feature_dim: int,
        pool_size: int,
        output_final_state: bool = True,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[tuple] = None,
    ):
        super().__init__()
        self.output_final_state = output_final_state
        self._output_dim = pool_size + 1 if output_final_state else pool_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size

        assert (
            n_hidden_layers is not None and n_hidden_layers >= 0
        ), "n_hidden_layers must be >= 0"
        assert activation_fn is not None, "activation_fn must be provided"

        if torso is not None:
            self.torso = torso
            self.batch_embed = torso[0]
            self.combined_embed = torso[1]
            self.pool_embed = torso[2]
        else:
            self.batch_embed = MLPEncoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.pool_embed = MLPEncoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.combined_embed = MLPEncoder(
                input_dim=hidden_dim * 2,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )

            self.torso = (self.batch_embed, self.combined_embed, self.pool_embed)

        self.last_layer = nn.Linear(self.hidden_dim, 1)

        self.device = None

    def forward(self, x):
        """
        Args:
            x (tuple): tuple containing:
                pool (torch.Tensor): (pool_size, feature_dim)
                batch_mask (torch.Tensor): (batch_size, pool_size)
        Returns:
            torch.Tensor: (batch_size, pool_size+1)
            last column is the probability of not acquiring any point (i.e. stopping)
        """
        pool, batch_mask = x  # (pool_size, feature_dim), (batch_size, pool_size)
        if self.device is None:  # from gfn package
            self.device = batch_mask.device
            self.to(self.device)

        # batch embeddings
        batch_embed = self.batch_embed(pool)  # (pool_size, hidden_dim)
        sum_encoded_batch = torch.sum(
            batch_embed * batch_mask.unsqueeze(-1), dim=1
        )  # (batch_size, hidden_dim)
        # repeat for each point in the pool
        sum_encoded_batch = sum_encoded_batch.repeat(1, self._output_dim).view(
            -1, self._output_dim, self.hidden_dim
        )  # (batch_size, pool_size (+1), hidden_dim)

        # pool embeddings
        if self.output_final_state:
            # add a column of zeros to the pool to represent the probability of not acquiring any point
            pool = torch.cat(
                (pool, torch.zeros((1, pool.shape[1]), device=batch_mask.device)), dim=0
            )  # (pool_size+1, feature_dim)
        pool_embed = self.pool_embed(pool)  # (pool_size(+1), hidden_dim)
        # repeat pool embeddings for each batch
        pool_embed = pool_embed.repeat(
            batch_mask.shape[0], 1, 1
        )  # (batch_size, pool_size(+1), hidden_dim)

        # concatenate the two
        pool_cat_batch = torch.cat(
            (pool_embed, sum_encoded_batch), dim=-1
        )  # (batch_size, pool_size, hidden_dim*2)
        # pass it through a MLP
        hid = self.combined_embed(pool_cat_batch)  # (batch_size, pool_size, hidden_dim)
        logits = self.last_layer(hid)  # (batch_size, pool_size, 1)
        return logits.squeeze(-1)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class QueryNet(nn.Module, GFNModule):
    """Implements a basic MLP which outputs logits for acquiring each of the pool items.
    We want to be set invariant so we sum over embeddings of the acquired points so far.
    This just outputs a probability for each item in the pool, without any embedding of pool points.
    Args:
        feature_dim (int): number of features for each item in the pool
        pool_size (int): size of the pool (output size)
        output_final_state (bool, optional): Whether to output the probability of not acquiring any point (i.e. stopping). Defaults to True.
        hidden_dim (Optional[int], optional): Number of units per hidden layer. Defaults to 256.
        n_hidden_layers (Optional[int], optional): Number of hidden layers. Defaults to 2.
        activation_fn (Optional[Literal[relu, tanh]], optional): Activation function. Defaults to "relu".
    """

    def __init__(
        self,
        feature_dim: int,
        pool_size: int,
        output_final_state: bool = True,
        hidden_dim: Optional[int] = 256,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh"]] = "relu",
        torso: Optional[tuple] = None,
    ):
        super().__init__()
        self.output_final_state = output_final_state
        self._output_dim = pool_size + 1 if output_final_state else pool_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size

        assert (
            n_hidden_layers is not None and n_hidden_layers >= 0
        ), "n_hidden_layers must be >= 0"

        if torso is not None:
            self.torso = torso
            self.batch_embed = torso[0]
            self.combined_embed = torso[1]
        else:
            self.batch_embed = MLPEncoder(
                input_dim=feature_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )
            self.combined_embed = MLPEncoder(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                encoding_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                activation_fn=activation_fn,
            )
            self.torso = (self.batch_embed, self.combined_embed)

        self.last_layer = nn.Linear(self.hidden_dim, self._output_dim)

        self.device = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (batch_size, query_size, feature_dim)
        Returns:
            torch.Tensor: (batch_size, pool_size+1)
            last column is the probability of not acquiring any point (i.e. stopping)
        """
        if self.device is None:  # from gfn package
            self.device = x.device
            self.to(self.device)
        # batch embeddings
        batch_embed = self.batch_embed(x)  # (B, query_size, hidden_dim)
        sum_encoded_batch = torch.sum(batch_embed, dim=1)  # (B, hidden_dim)
        # pass it through a MLP
        hid = self.combined_embed(sum_encoded_batch)  # (B, hidden_dim)
        logits = self.last_layer(hid)  # (batch_size, pool_size)
        return logits

    @property
    def output_dim(self) -> int:
        return self._output_dim
