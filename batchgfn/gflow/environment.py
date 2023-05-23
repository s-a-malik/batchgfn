from typing import Literal, Tuple, cast, Callable
from gfn.envs.env import Env
from gfn.envs.preprocessors import Preprocessor
from gfn.containers.states import States
from gymnasium.spaces import Discrete

import torch
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from torchtyping import TensorType

from batchgfn.models.cnn import CnnAutoencoder
from batchgfn.datasets.mnist import get_data, get_targets


class PoolAndTrainNetPreprocessor(Preprocessor):
    """Implements a function that turns a MiFlowStates object representing the indices
    of the examples in the pool in the current state, to an object that can be fed to
    the models (e.g. NeuralNets/transformers for PF and PB)
    TODO add current dataset to state to allow for different datasets in batch
    """

    def __init__(
        self,
        tensors_pool: torch.tensor,
        labels_pool: torch.tensor,
        current_dataset: torch.tensor,
    ) -> None:
        """
        Args:
            tensors_pool (torch.tensor): the pool dataset (pool_size, feature_size)
            current_dataset (torch.tensor): indicator of current labelled dataset (pool_size,)
        """
        super().__init__(output_shape=len(tensors_pool))
        # save tensors
        self.tensors_pool = tensors_pool
        self.labels_pool = labels_pool
        self.current_dataset = current_dataset

    def preprocess(self, states: States):
        # feed model the pool dataset, the current state mask, and labelled points
        preprocessed_states = (
            self.tensors_pool,
            self.labels_pool,
            states.states_tensor,
            self.current_dataset,
        )
        return preprocessed_states


class PoolNetPreprocessor(Preprocessor):
    """Implements a function that turns a MiFlowStates object representing the indices
    of the examples in the pool in the current state, to an object that can be fed to
    the models (e.g. NeuralNets/transformers for PF and PB)"""

    def __init__(self, tensors_pool: torch.utils.data.Dataset) -> None:
        """
        Args:
            tensors_pool (torch.tensor): the pool dataset
        """
        super().__init__(output_shape=len(tensors_pool))
        # save tensors
        self.tensors_pool = tensors_pool

    def preprocess(self, states: States):
        # feed model the pool dataset and the current state mask
        preprocessed_states = (self.tensors_pool, states.states_tensor)

        return preprocessed_states


class QueryNetPreprocessor(Preprocessor):
    """Implements a function that turns a MiFlowStates object representing the indices
    of the examples in the pool in the current state, to an object that can be fed to
    the models (e.g. NeuralNets/transformers for PF and PB)
    """

    def __init__(self, tensors_pool: torch.utils.data.Dataset) -> None:
        """Preprocessor for environments with enumerable states (finite number of states).
        Each state is represented by a unique integer (>= 0) index.

        Args:
            tensors_pool (torch.tensor): the pool dataset
        """
        super().__init__(output_shape=len(tensors_pool))
        self.tensors_pool = tensors_pool

    def preprocess(self, states: States):
        query_batch = states.states_tensor
        # get indices of points in each state in batch
        query_indices = [torch.nonzero(x, as_tuple=True)[0] for x in query_batch]
        if len(query_indices) == 0:
            return self.tensors_pool[[]].unsqueeze(1)
        else:
            # get the corresponding pool points, padded to the same length
            return pad_sequence(
                [self.tensors_pool[x] for x in query_indices], batch_first=True
            )


class MiFlowEnv(Env):
    def __init__(
        self,
        ds_pool: torch.utils.data.Dataset,
        proxy_reward_function: Callable,
        reward_temp: float = 1.0,
        stop_after: int | None = None,
        preprocessor_str: Literal = "poolnet",
        device_str: Literal["cpu", "cuda"] = "cpu",
        compute_full_dist: bool = False,
        one_hot_labels: bool = False,
        autoencode: bool = False,
        autoencoder_latent_dim: int = 32,
    ):
        """GFlowNet Active Learning environment. States are encoded as K-hot vectors.

        Args:
            ds_pool (torch.utils.data.Dataset): the pool dataset
            proxy_reward_function (Callable): A function that takes as input the current batch and outputs the reward for the batch
            reward_temp (float, optional): Temperature for the reward function. Defaults to 1.0.
            stop_after (int | None, optional): If None, all states are terminal. Otherwise, only states with
                    exactly stop_after ones are terminal. Defaults to None.
            preprocessor_str (Literal, optional): Preprocessor to use. Defaults to "poolnet".
            device_str (cuda | cpu, optional). Defaults to "cpu".
            compute_full_dist (bool, optional): Whether to compute the full distance matrix. Defaults to False.
            one_hot_labels (bool, optional): Whether to use one-hot labels (for classification). Defaults to False.
            autoencode (bool, optional): Whether to use an autoencoder to preprocess the data. Defaults to False.
            autoencoder_latent_dim (int, optional): Latent dimension of the autoencoder. Defaults to 32.

        self.current_dataset represents the current conditioning dataset. It is a K-hot vector,
        used to mask the actions corresponding to examples that are already there.
        It can be updated with self.update_labelled_dataset(indices_to_add)
        """
        self.pool_size = len(ds_pool)
        self.stop_after = stop_after
        self.proxy_reward_function = proxy_reward_function
        self.reward_temp = reward_temp
        self.device_str = device_str
        self.preprocessor_str = preprocessor_str
        self.compute_full_dist = compute_full_dist
        self.one_hot_labels = one_hot_labels
        self.autoencode = autoencode
        self.autoencoder_latent_dim = autoencoder_latent_dim
        # TODO full pool on GPU, memory intensive
        if self.autoencode:
            self.tensors_pool = get_data(ds_pool)[0].to(device_str).clone()
        else:
            self.tensors_pool = ds_pool[:][0].to(device_str).clone()

        if self.one_hot_labels:
            self.labels_pool = torch.nn.functional.one_hot(
                get_targets(ds_pool).to(self.device_str)
            ).clone()  # (pool_size, num_classes)
        else:
            self.labels_pool = (
                get_targets(ds_pool).unsqueeze(-1).to(self.device_str).clone()
            )

        if self.compute_full_dist:
            # enumeration of possible combinations using itertools (this is large for large pool/query size!)
            self.state_indices = torch.combinations(
                torch.arange(self.pool_size, device=self.device_str), r=self.stop_after
            )

        s0 = torch.zeros(
            self.pool_size, dtype=torch.long, device=torch.device(device_str)
        )
        sf = torch.full_like(s0, -1)

        self.current_dataset = torch.zeros_like(s0)

        action_space = Discrete(self.pool_size + 1)

        # preprocessor TODO do this in cleaner way in preprocessor
        if autoencode:
            # train a simple autoencoder on the pool and use latents as pool tensors
            print("Training autoencoder on pool")
            self.autoencoder = CnnAutoencoder(
                latent_dim=self.autoencoder_latent_dim, kernel_size=3, strides=1
            )
            ds_pool_dataloader = torch.utils.data.DataLoader(
                ds_pool, batch_size=64, shuffle=False
            )
            trainer = pl.Trainer(max_epochs=600)
            trainer.fit(self.autoencoder, ds_pool_dataloader)
            _, embeddings = trainer.predict(self.autoencoder, ds_pool_dataloader)[0]
            embeddings = embeddings.detach().clone().to(self.device_str)
        else:
            embeddings = self.tensors_pool
        if preprocessor_str == "poolnet":
            preprocessor = PoolNetPreprocessor(
                tensors_pool=embeddings,
            )
        elif preprocessor_str == "querynet":
            preprocessor = QueryNetPreprocessor(
                tensors_pool=embeddings,
            )
        elif preprocessor_str in ["poolandtrainnet", "npt"]:
            preprocessor = PoolAndTrainNetPreprocessor(
                tensors_pool=embeddings,
                labels_pool=self.labels_pool,
                current_dataset=self.current_dataset,
            )
        else:
            raise ValueError(f"preprocessor_str {preprocessor_str} not recognized")
        super().__init__(
            action_space=action_space,
            s0=s0,
            sf=sf,
            preprocessor=preprocessor,
            device_str=device_str,
        )

    def reset_labelled_dataset(self) -> None:
        self.current_dataset = torch.zeros_like(self.current_dataset)
        if self.preprocessor_str in ["poolandtrainnet", "npt"]:
            self.preprocessor.current_dataset = self.current_dataset

    def update_labelled_dataset(
        self, indices_to_add: TensorType["size", torch.long]
    ) -> None:
        self.current_dataset[indices_to_add] = 1
        if self.preprocessor_str in ["poolandtrainnet", "npt"]:
            self.preprocessor.current_dataset[indices_to_add] = 1

    def update_pool_dataset(self, ds_pool) -> None:
        if self.autoencode:
            self.tensors_pool = get_data(ds_pool)[0].to(self.device_str).clone()
            # train a simple autoencoder on the pool and use latents as pool tensors
            self.autoencoder = CnnAutoencoder(
                latent_dim=self.autoencoder_latent_dim, kernel_size=3, strides=1
            )
            ds_pool_dataloader = torch.utils.data.DataLoader(
                ds_pool, batch_size=64, shuffle=False
            )
            trainer = pl.Trainer(max_epochs=600)
            trainer.fit(self.autoencoder, ds_pool_dataloader)
            _, embeddings = trainer.predict(self.autoencoder, ds_pool_dataloader)[0]
            embeddings = embeddings.detach().clone().to(self.device_str)
        else:
            self.tensors_pool = ds_pool[:][0].to(self.device_str).clone()
            embeddings = self.tensors_pool

        if self.one_hot_labels:
            self.labels_pool = torch.nn.functional.one_hot(
                get_targets(ds_pool).to(self.device_str)
            ).clone()  # (pool_size, num_classes)
        else:
            self.labels_pool = (
                get_targets(ds_pool).unsqueeze(-1).to(self.device_str).clone()
            )

        self.pool_size = len(ds_pool)
        self.preprocessor.tensors_pool = embeddings
        self.preprocessor.labels_pool = self.labels_pool

    def make_States_class(self) -> type[States]:
        env = self

        class MiFlowStates(States):
            state_shape = (env.pool_size,)
            s0 = env.s0
            sf = env.sf

            @classmethod
            def make_random_states_tensor(
                cls, batch_shape: Tuple[int]
            ) -> TensorType["batch_shape", "state_shape", torch.long]:
                return torch.randint(
                    0, 2, batch_shape + cls.state_shape, dtype=torch.long
                )

            def make_masks(
                self,
            ) -> Tuple[
                TensorType["batch_shape", "n_actions", torch.bool],
                TensorType["batch_shape", "n_actions - 1", torch.bool],
            ]:
                forward_masks = torch.ones(
                    self.batch_shape + (env.n_actions,),
                    dtype=torch.bool,
                    device=self.device,
                )
                backward_masks = torch.ones(
                    self.batch_shape + (env.n_actions - 1,),
                    dtype=torch.bool,
                    device=self.device,
                )
                return forward_masks, backward_masks

            def update_masks(self) -> None:
                self.backward_masks = self.states_tensor == 1

                # Next line is for typing only
                self.forward_masks = cast(torch.Tensor, self.forward_masks)
                if env.stop_after is None:
                    self.forward_masks[..., :-1] = self.states_tensor == 0
                else:
                    states_size = self.states_tensor.sum(dim=-1)
                    self.forward_masks[..., -1] = states_size == env.stop_after
                    self.forward_masks[..., :-1][states_size >= env.stop_after] = False
                    self.forward_masks[..., :-1][states_size < env.stop_after] = (
                        self.states_tensor[states_size < env.stop_after] == 0
                    )
                self.forward_masks[..., :-1][..., env.current_dataset > 0] = False

        return MiFlowStates

    def is_exit_actions(
        self, actions: TensorType["batch_shape"]
    ) -> TensorType["batch_shape"]:
        return actions == self.action_space.n - 1

    def maskless_step(
        self,
        states: TensorType["batch_type", "state_shape", torch.long],
        actions: TensorType["batch_shape"],
    ) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 1)

    def maskless_backward_step(
        self,
        states: TensorType["batch_type", "state_shape", torch.long],
        actions: TensorType["batch_shape"],
    ) -> None:
        states.scatter_(-1, actions.unsqueeze(-1), 0)

    def log_reward(self, final_states: States) -> TensorType["batch_shape"]:
        if final_states.batch_shape == (0,):
            return torch.zeros((0,), device=self.device)
        query_batch = final_states.states_tensor
        # get indices of points in each state in batch
        query_indices = [torch.nonzero(x, as_tuple=True)[0] for x in query_batch]

        if len(query_indices) == 0:
            query_points = self.tensors_pool[[]].unsqueeze(1)
        else:
            # get the corresponding pool points, padded to the same length
            query_points = pad_sequence(
                [self.tensors_pool[x] for x in query_indices], batch_first=True
            )
        # return reward for each state in batch
        raw_reward = self.proxy_reward_function(query_points)
        # exponentiate and scale by temperature
        return raw_reward / self.reward_temp

    def get_states_indices(self, states: States) -> torch.Tensor:
        """Return the indices of the states in the pool (only necessary for terminating states?)"""
        states_raw = states.states_tensor
        # get the indices of non-zero elements in each state in batch
        query_indices = torch.stack(
            [torch.nonzero(x, as_tuple=True)[0] for x in states_raw]
        )
        _, indices = torch.topk(
            ((self.state_indices.t() == query_indices.unsqueeze(-1)).all(dim=1)).int(),
            k=1,
            dim=1,
        )

        return indices.squeeze(-1)

    def get_terminating_states_indices(self, states: States) -> torch.Tensor:
        """
        Get the indices of the terminating states
        """
        return self.get_states_indices(states)

    @property
    def n_states(self) -> int:
        """TODO"""
        pass

    @property
    def n_terminating_states(self) -> int:
        """number of terminating states"""
        return self.state_indices.shape[0]

    @property
    def all_states(self) -> States:
        """TODO
        Return all possible query batches
        """
        pass

    @property
    def terminating_states(self) -> States:
        """
        States that terminate the episode (fill the batch)
        """
        terminating_states_shape = (self.n_terminating_states, self.pool_size)
        # expand indices to states
        terminating_states = torch.scatter(
            input=torch.zeros(terminating_states_shape, device=self.device),
            dim=1,
            index=self.state_indices,
            src=torch.ones(terminating_states_shape, device=self.device),
        )
        # return states object
        return self.States(terminating_states)

    @property
    def true_dist_pmf(self) -> torch.Tensor:
        """
        Get the true reward distribution.
        """
        terminating_states = self.terminating_states
        # Create a mask that is 1 for reachable terminating states and 0 for unreachable states
        # if a state has an already labelled pool point, it is unreachable
        mask = ((terminating_states.states_tensor + self.current_dataset) != 2).all(
            dim=1
        )
        true_dist = self.reward(terminating_states) * mask
        true_dist /= true_dist.sum()
        return true_dist

    @property
    def log_partition(self) -> float:
        """
        Log partition function
        """
        terminating_states = self.terminating_states
        mask = ((terminating_states.states_tensor + self.current_dataset) != 2).all(
            dim=1
        )
        rewards = self.reward(self.terminating_states) * mask
        return torch.log(rewards.sum()).item()
