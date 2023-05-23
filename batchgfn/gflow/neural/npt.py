"""
Non-parametric transformer (NPT) module and utils.
c.f. https://arxiv.org/pdf/2106.02584.pdf and https://github.com/OATML/non-parametric-transformers
"""

import math

from argparse import Namespace
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from gfn.modules import GFNModule

from batchgfn.gflow.neural.base import MLPEncoder


class NPT(nn.Module, GFNModule):
    """Non-parametric transformer (NPT) module."""

    def __init__(
        self,
        feature_dim: int,
        label_dim: int,
        pool_size: int,
        output_final_state: bool = True,
        feature_dim_hid: int = 128,
        label_dim_hid: int = 32,
        n_transformer_layers: Optional[int] = 4,
        n_transformer_heads: Optional[int] = 8,
        torso: Optional[tuple] = None,
    ):
        """Initialises Full NPT model.
        Consists of input embedding + NPT layers.
        """
        super().__init__()
        self._output_dim = pool_size + 1 if output_final_state else pool_size
        self.output_final_state = output_final_state
        self.pool_size = pool_size
        self.feature_dim = feature_dim
        self.label_dim = label_dim
        self.feature_dim_hid = feature_dim_hid
        self.label_dim_hid = label_dim_hid
        self.n_transformer_layers = n_transformer_layers
        self.n_transformer_heads = n_transformer_heads

        # hypers TODO move to config file
        cm = {
            "dim_feat_hidden": feature_dim_hid,
            "dim_label_hidden": label_dim_hid,
            "stacking_depth": n_transformer_layers,
            "n_out": 1,
            # model_num_inds: 16,
            "dropout_prob": 0,  # percent of labels masked and used as train_labels
            "use_rff": True,  # use rff or simple linear layer
            "variance_eps": 1e-10,
            "restrict_attention": True,
            "n_rff_layers": 1,
            "output_final_state": output_final_state,
        }
        cattention = {
            "mix_heads": True,
            "num_heads": n_transformer_heads,
            "sep_res_embed": True,
            "att_block_layer_norm": True,
            "rff_depth": 1,
            "att_score_norm": "softmax",
            "pre_layer_norm": True,
            "rff_gated_gelu": False,
            "ablate_rff": False,
            "share_qk_sab_embedding": False,
            "layer_norm_eps": 1e-12,
            "hidden_dropout_prob": 0.0,
            "att_score_dropout_prob": 0.0,
            "att_additive_encoding": False,
            "att_multiplicative_encoding": False,
            "num_inds": 10,
            "viz_att_maps": False,
        }
        cm = Namespace(**cm)
        cattention = Namespace(**cattention)
        self.cattention = cattention
        self.cm = cm

        if torso is not None:
            self.torso = torso
            self.rff = torso[0]
            self.label_to_embedding_dim = torso[1]
            self.npt = torso[2]
        else:
            if cm.use_rff:
                self.rff = MLPEncoder(
                    input_dim=feature_dim,
                    hidden_dim=feature_dim
                    * 4,  # 4 times expansion factor as in 'Attention is all you need'
                    encoding_dim=cm.dim_feat_hidden,
                    n_hidden_layers=cm.n_rff_layers,
                    activation_fn="gelu",
                )
            else:
                self.rff = nn.Linear(feature_dim, cm.dim_feat_hidden)

            # We separately embed the (label + mask) columns of the input
            self.label_to_embedding_dim = nn.Linear(label_dim + 2, cm.dim_label_hidden)

            # The total input hidden dim for the NPT is thus
            self.npt_dim_hidden = cm.dim_feat_hidden + cm.dim_label_hidden
            # And this needs to evenly divide by the number of heads
            assert self.npt_dim_hidden % cattention.num_heads == 0
            kwargs = dict(
                sab_args=cattention,
                stacking_depth=cm.stacking_depth,
                dim_hidden=self.npt_dim_hidden,
                n_out=cm.n_out,
            )
            self.npt = NPTBase(**kwargs)
            self.torso = (self.rff, self.label_to_embedding_dim, self.npt)

        # TODO removed decoder in NPTBase
        # self.last_layer = nn.Linear(self.npt.dim_hidden, 1)
        self.last_layer = nn.Sequential(
            nn.Linear(self.npt.dim_hidden, 4 * self.npt.dim_hidden),
            nn.GELU(),
            nn.Linear(4 * self.npt.dim_hidden, 1),
        )

        self.device = None

    def forward(self, x):
        """
        Args:
            x (tuple): tuple containing:
                pool (torch.Tensor): (pool_size, feature_dim)
                labels (torch.Tensor): (pool_size, [label_dim])
                batch_mask (torch.Tensor): (batch_size, pool_size)
                current_dataset (torch.Tensor): (pool_size,)
        Returns:
            torch.Tensor: (batch_size, pool_size+1)
                last column is the probability of not acquiring any point (i.e. stopping)
        """
        (
            pool,
            labels_raw,
            batch_mask,
            current_dataset,
        ) = x  # (pool_size, feature_dim), (pool_size, label_dim), (batch_size, pool_size), (pool_size,)
        if self.device is None:  # from gfn package
            self.device = pool.device
            self.to(self.device)

        # Mask out the target values at the mask indices
        # clone so don't modify the original
        labels = labels_raw.clone()
        labels[current_dataset == 0, :] = 0

        if self.output_final_state:
            # add a column of zeros to the pool to represent the probability of not acquiring any point
            pool = torch.cat(
                (pool, torch.zeros((1, pool.shape[1]), device=batch_mask.device)), dim=0
            )  # (pool_size+1, feature_dim)
            labels = torch.cat(
                (labels, torch.zeros((1, labels.shape[1]), device=batch_mask.device)),
                dim=0,
            )  # (pool_size+1, label_dim)
            current_dataset = torch.cat(
                (current_dataset, torch.zeros((1,), device=batch_mask.device)), dim=0
            )  # (pool_size+1,)
            batch_mask = torch.cat(
                (
                    batch_mask,
                    torch.zeros((batch_mask.shape[0], 1), device=batch_mask.device),
                ),
                dim=1,
            )  # (batch_size, pool_size+1)

        # Embed input features
        feature_embeddings = self.rff(pool)  # (pool_size, cm.dim_feat_hidden)
        feature_embeddings = (
            feature_embeddings.repeat(  # (batch_size, pool_size, cm.dim_feat_hidden)
                batch_mask.shape[0], 1, 1
            )
        )

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

        if self.cm.restrict_attention:
            attention_mask = get_attention_mask(
                training_indictators.squeeze(),
                batch_indictators.squeeze(),
                self.cattention.num_heads,
                device=labels.device,
            )
        else:
            attention_mask = None

        masked_targets = torch.cat(
            (labels, training_indictators, batch_indictators), dim=2
        )  # (batch_size, pool_size, label_dim + 2)
        target_embeddings = self.label_to_embedding_dim(masked_targets)

        embeddings = torch.cat(
            (feature_embeddings, target_embeddings), dim=2
        )  # (batch_size, pool_size, cm.dim_feat_hidden + cm.dim_label_hidden)

        # NPT treats first dim as batch, performs attention over the second dim
        # Predicts the location and scale of the MVN distribution
        npt_out = self.npt(embeddings, attention_mask)  # [-n_indices_to_mask:]

        logits = self.last_layer(npt_out)  # (batch_size, pool_size+1)

        return logits.squeeze()

    @property
    def output_dim(self) -> int:
        return self._output_dim


class NPTBase(nn.Module):
    def __init__(self, sab_args, stacking_depth, dim_hidden, n_out):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.sab_args = sab_args
        enc = []
        for _ in range(stacking_depth):
            enc.append(SAB(dim_hidden, dim_hidden, dim_hidden, sab_args))
        self.enc = nn.ModuleList(enc)

        # self.decode = nn.Linear(dim_hidden, n_out)
        # self.decode = nn.Sequential(
        #     nn.Linear(dim_hidden, 4 * dim_hidden),
        #     nn.GELU(),
        #     nn.Linear(4 * dim_hidden, n_out))

    def forward(self, x, attention_mask=None):
        for block in self.enc:
            x = block(x, attention_mask)
        # x = self.decode(x)
        return x


class SAB(nn.Module):
    """Multi-head Self-Attention Block."""

    has_inducing_points = False

    def __init__(self, dim_in, dim_emb, dim_out, c, num_input_features=None):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_emb, dim_out, c, dim_att=num_input_features)

    def forward(self, X, attention_mask=None):
        return self.mab(X, X, attention_mask)


class MAB(nn.Module):
    """Multi-head Attention Block."""

    def __init__(
        self,
        dim_Q,
        dim_KV,
        dim_emb,
        dim_out,
        c,
        dim_att=None,
        force_ablate_rff=False,
        ablate_res=False,
    ):
        """
        Inputs have shape (B_A, N_A, F_A), where
        * `B_A` is a batch dimension, along we parallelise computation,
        * `N_A` is the number of samples in each batch, along which we perform
        attention, and
        * `F_A` is dimension of the embedding at input
            * `F_A` is `dim_Q` for query matrix
            * `F_A` is `dim_KV` for key and value matrix.
        Q, K, and V then all get embedded to `dim_emb`.
        `dim_out` is the output dimensionality of the MAB which has shape
        (B_A, N_A, dim_out), this can be different from `dim_KV` due to
        the head_mixing.
        This naming scheme is inherited from set-transformer paper.
        dim_att: Tuple[int, int], needs to be specified when we aim to learn
            and apply either additive encodings to the attention weight tensor
            (pre-softmax) or multiplicative encodings to the attention score
            tensor (post-softmax).
            NOTE: this is only valid when performing attention over the
            columns, as in nested attention (else it would break row
            equivariance).
        force_ablate_rff: bool, if True, do not apply the rFF on this MAB.
        ablate_res: bool, if True, ablate residual connections.
        """
        super(MAB, self).__init__()
        self.att_score_norm = c.att_score_norm
        self.pre_layer_norm = c.pre_layer_norm
        self.viz_att_maps = c.viz_att_maps
        self.ablate_rff = c.ablate_rff
        self.force_ablate_rff = force_ablate_rff
        self.ablate_res = ablate_res
        self.share_qk_sab_embedding = c.share_qk_sab_embedding

        if self.viz_att_maps:
            self.save_att_maps = SaveAttMaps()

        if dim_out is None:
            dim_out = dim_emb
        elif (dim_out is not None) and (c.mix_heads is None):
            print("Warning: dim_out transformation does not apply.")
            dim_out = dim_emb

        self.num_heads = c.num_heads
        self.dim_KV = dim_KV
        self.dim_split = dim_emb // c.num_heads

        if self.share_qk_sab_embedding:
            self.fc_qk = nn.Linear(dim_Q, dim_emb)
        else:
            self.fc_q = nn.Linear(dim_Q, dim_emb)
            self.fc_k = nn.Linear(dim_KV, dim_emb)

        self.fc_v = nn.Linear(dim_KV, dim_emb)

        self.fc_mix_heads = nn.Linear(dim_emb, dim_out) if c.mix_heads else None
        self.fc_res = nn.Linear(dim_Q, dim_out) if c.sep_res_embed else None

        # Initialize additive and multiplicative encodings
        self.init_additive_multiplicative_encodings(c, dim_att)

        if c.att_block_layer_norm:
            if self.pre_layer_norm:  # Applied to X
                self.ln0 = nn.LayerNorm(dim_Q, eps=c.layer_norm_eps)
            else:  # Applied after MHA and residual
                self.ln0 = nn.LayerNorm(dim_out, eps=c.layer_norm_eps)

            self.ln1 = nn.LayerNorm(dim_out, eps=c.layer_norm_eps)
        else:
            self.ln0 = None
            self.ln1 = None

        self.hidden_dropout = (
            nn.Dropout(p=c.hidden_dropout_prob) if c.hidden_dropout_prob else None
        )

        self.att_scores_dropout = (
            nn.Dropout(p=c.att_score_dropout_prob) if c.att_score_dropout_prob else None
        )

        if not self.ablate_rff and not self.force_ablate_rff:
            if c.rff_gated_gelu:
                self.rff = DenseGatedGeluDense(
                    dim_out=dim_out, dropout=self.hidden_dropout
                )
            else:
                self.init_rff(dim_out, c.rff_depth)

    def init_additive_multiplicative_encodings(self, c, dim_att):
        att_additive_encoding = None
        att_multiplicative_encoding = None

        if dim_att is not None:
            # dimension of attention
            if isinstance(dim_att, int):
                dims = (self.num_heads, dim_att, dim_att)
            else:
                dims = (self.num_heads, *dim_att)

            if c.att_additive_encoding:
                att_additive_encoding = nn.Parameter(torch.Tensor(*dims))
                # Centered at 0
                nn.init.xavier_uniform_(att_additive_encoding, gain=1)

            if c.att_multiplicative_encoding:
                att_multiplicative_encoding = nn.Parameter(torch.Tensor(*dims))
                # Centered at 1 (defaults to identity)
                xavier_uniform_loc_(att_multiplicative_encoding, loc=1, gain=1)

        self.att_additive_encoding = att_additive_encoding
        self.att_multiplicative_encoding = att_multiplicative_encoding

    def init_rff(self, dim_out, rff_depth):
        # Linear layer with 4 times expansion factor as in 'Attention is
        # all you need'!
        self.rff = [nn.Linear(dim_out, 4 * dim_out), nn.GELU()]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        for i in range(rff_depth - 1):
            self.rff += [nn.Linear(4 * dim_out, 4 * dim_out), nn.GELU()]

            if self.hidden_dropout is not None:
                self.rff.append(self.hidden_dropout)

        self.rff += [nn.Linear(4 * dim_out, dim_out)]

        if self.hidden_dropout is not None:
            self.rff.append(self.hidden_dropout)

        self.rff = nn.Sequential(*self.rff)

    def forward(self, X, Y, attention_mask=None):
        if self.pre_layer_norm and self.ln0 is not None:
            X_multihead = self.ln0(X)
        else:
            X_multihead = X

        if self.share_qk_sab_embedding:
            Q = self.fc_qk(X_multihead)
        else:
            Q = self.fc_q(X_multihead)

        if not self.ablate_res:
            if self.fc_res is None:
                X_res = Q
            else:
                X_res = self.fc_res(X)  # Separate embedding for residual

        if self.share_qk_sab_embedding:
            K = self.fc_qk(Y)
        else:
            K = self.fc_k(Y)

        V = self.fc_v(Y)

        Q_ = torch.cat(Q.split(self.dim_split, 2), 0)
        K_ = torch.cat(K.split(self.dim_split, 2), 0)
        V_ = torch.cat(V.split(self.dim_split, 2), 0)

        # TODO: track issue at
        # https://github.com/juho-lee/set_transformer/issues/8
        # A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        A = torch.einsum("ijl,ikl->ijk", Q_, K_)

        # Perform elementwise addition using learned "additive encodings" on
        # the pre-softmax attention weights.
        # These allow the model to, for example, focus on modelling some
        # interactions between columns while avoiding others entirely at
        # a particular row. Inspired by tree-based methods.
        if self.att_additive_encoding is not None:
            additive_stack = self.att_additive_encoding.repeat(
                int(A.size(0) / self.att_additive_encoding.size(0)), 1, 1
            )
            A = additive_stack + A

        if attention_mask is not None:
            # A = A + attention_mask.unsqueeze(0)
            A = A + attention_mask  # TODO: check this change for 3D masks (multi-head?)

        if self.att_score_norm == "softmax":
            A = torch.softmax(A / math.sqrt(self.dim_KV), 2)
        elif self.att_score_norm == "constant":
            A = A / self.dim_split
        else:
            raise NotImplementedError

        # Perform elementwise multiplication using learned "multiplicative
        # encodings" on the post-softmax attention scores.
        # See above for explanation.
        if self.att_multiplicative_encoding is not None:
            mult_stack = self.att_multiplicative_encoding.repeat(
                int(A.size(0) / self.att_multiplicative_encoding.size(0)), 1, 1
            )
            A = mult_stack * A

        if self.viz_att_maps:
            A = self.save_att_maps(A, Q_, K_, V_)

        # Attention scores dropout is applied to the N x N_v matrix of
        # attention scores.
        # Hence, it drops out entire rows/cols to attend to.
        # This follows Vaswani et al. 2017 (original Transformer paper).

        if self.att_scores_dropout is not None:
            A = self.att_scores_dropout(A)

        multihead = A.bmm(V_)
        multihead = torch.cat(multihead.split(Q.size(0), 0), 2)

        # Add mixing of heads in hidden dim.
        # TODO: ablate effect of this

        if self.fc_mix_heads is not None:
            H = self.fc_mix_heads(multihead)
        else:
            H = multihead

        # Follow Vaswani et al. 2017 in applying dropout prior to
        # residual and LayerNorm
        if self.hidden_dropout is not None:
            H = self.hidden_dropout(H)

        # True to the paper would be to replace
        # self.fc_mix_heads = nn.Linear(dim_V, dim_Q)
        # and Q_out = X
        # Then, the output dim is equal to input dim, just like it's written
        # in the paper. We should definitely check if that boosts performance.
        # This will require changes to downstream structure (since downstream
        # blocks expect input_dim=dim_V and not dim_Q)

        if not self.ablate_res:
            # Residual connection
            Q_out = X_res
            H = H + Q_out

        # Post Layer-Norm, as in SetTransformer and BERT.
        if not self.pre_layer_norm and self.ln0 is not None:
            H = self.ln0(H)

        if self.pre_layer_norm and self.ln1 is not None:
            H_rff = self.ln1(H)
        else:
            H_rff = H

        if self.ablate_rff or self.force_ablate_rff:
            expanded_linear_H = H_rff
        else:
            # Apply row-wise feed forward network
            expanded_linear_H = self.rff(H_rff)

        if not self.ablate_res:
            # Residual connection
            expanded_linear_H = H + expanded_linear_H

        if not self.pre_layer_norm and self.ln1 is not None:
            expanded_linear_H = self.ln1(expanded_linear_H)

        if self.viz_att_maps:
            self.save_att_maps.out = nn.Parameter(expanded_linear_H)
            self.save_att_maps.out_pre_res = nn.Parameter(H)

        return expanded_linear_H


class DenseGatedGeluDense(nn.Module):
    """
    Due to Huggingface's implementation of Google's T5.
    https://github.com/huggingface/transformers/blob/948b730f9777174335812cf7
    6de2a9dd9e4cf20e/src/transformers/models/t5/modeling_t5.py
    See also Shazeer 2020 (https://arxiv.org/pdf/2002.05202.pdf).
    Fixed to a 4x expansion factor, and depth 1.
    """

    def __init__(self, dim_out, dropout):
        super().__init__()
        self.wi_0 = nn.Linear(dim_out, dim_out * 4, bias=False)
        self.wi_1 = nn.Linear(dim_out, dim_out * 4, bias=False)
        self.wo = nn.Linear(dim_out * 4, dim_out, bias=False)
        self.dropout = dropout

    def gelu_new(self, x):
        """
        Implementation of the GELU activation function currently in Google
        BERT repo (identical to OpenAI GPT). Also see the Gaussian Error
        Linear Units paper: https://arxiv.org/abs/1606.08415
        """
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_new(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        if self.dropout is not None:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class SaveAttMaps(nn.Module):
    def __init__(self):
        super().__init__()
        self.curr_att_maps = None
        self.Q = None
        self.K = None
        self.V = None
        self.out = None
        self.out_pre_res = None

    def forward(self, X, Q, K, V):
        self.curr_att_maps = nn.Parameter(X)
        self.Q = nn.Parameter(Q)
        self.K = nn.Parameter(K)
        self.V = nn.Parameter(V)

        return X


def xavier_uniform_loc_(tensor, loc, gain=1.0):
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(loc - a, loc + a)` where
    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}
    Also known as Glorot initialization.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor
    Examples:
        >>> w = torch.empty(3, 5)
        >>> xavier_uniform_loc_(w, loc=1, gain=nn.init.calculate_gain('relu'))
    """
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return nn.init._no_grad_uniform_(tensor, loc - a, loc + a)


def get_attention_mask(training_indicators, batch_indicators, num_heads, device):
    """
    Restricts attention matrix in the following way:
    - unacquired (not in training or query batch set) points cannot attend to each other
    Args:
    - training_indicators (batch_size, pool_size): 0,1 indicating whether the point is in training set
    - batch_indicators (batch_size, pool_size): 0,1 indicating whether the point is in the current query batch

    Returns:
    - mask (torch.Tensor): attention mask (batch_size, pool_size, pool_size)
        denotes whether i is allowed to attend to j
    """
    # attention mask is added to logits before softmax, so zero is allowed attention
    # first disallow all attention
    mask = torch.ones(
        training_indicators.shape[0],
        training_indicators.shape[1],
        training_indicators.shape[1],
        device=device,
    )
    mask = mask * (-torch.inf)

    # 1. allow all (unacquired, batch, test) to attend to unacquired
    train_and_batch = training_indicators + batch_indicators != 0
    mask[train_and_batch.unsqueeze(1).repeat(1, training_indicators.shape[1], 1)] = 0
    # allow all points to attend to themselves
    mask[
        torch.eye(training_indicators.shape[1], dtype=torch.bool)
        .unsqueeze(0)
        .repeat(training_indicators.shape[0], 1, 1)
    ] = 0

    # multihead attention, repeat for each head
    mask = torch.repeat_interleave(mask, num_heads, dim=0)

    return mask
