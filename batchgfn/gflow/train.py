"""Functions and utils for training the GFN.
"""
from tqdm.autonotebook import tqdm, trange
import time
import math

import torch
import torch.nn.functional as F

from gfn import (
    LogitPBEstimator,
    LogitPFEstimator,
    LogZEstimator,
    LogEdgeFlowEstimator,
    LogStateFlowEstimator,
)
from gfn.distributions import EmpiricalTerminatingStatesDistribution
from gfn.utils import trajectories_to_training_samples
from gfn.losses import (
    TBParametrization,
    SubTBParametrization,
    TrajectoryBalance,
    SubTrajectoryBalance,
    FMParametrization,
    FlowMatching,
    DetailedBalance,
    DBParametrization,
    LogPartitionVarianceLoss,
    PFBasedParametrization,
)
from gfn.samplers import DiscreteActionsSampler, TrajectoriesSampler

from batchgfn.gflow.environment import MiFlowEnv
from batchgfn.gflow.neural import PoolNet, QueryNet, PoolAndTrainNet, StateNet, NPT


def get_optimizer_and_scheduler(cfg, params):
    if cfg.gfn_opt == "adam":
        optimizer = torch.optim.Adam(params)
    elif cfg.gfn_opt == "sgd":
        optimizer = torch.optim.SGD(params)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    if cfg.gfn_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.gfn_n_iterations
        )
    elif cfg.gfn_scheduler == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(cfg.n_iterations * cfg.warmup_prop), cfg.n_iterations
        )
    elif cfg.gfn_scheduler == "constant":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler}")

    return optimizer, scheduler


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases
    following the values of the cosine function between 0 and
    `pi * cycles` after a warmup period during which it increases
    linearly between 0 and 1.
    Copied from HuggingFace.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_gfn_module(
    cfg,
    feature_dim,
    label_dim=None,
    pool_size=None,
    torso=None,
    output_final_state: bool = False,
    state_flow: bool = False,
):
    if state_flow:
        # TODO: NPT version
        return StateNet(
            gfn_base_module=cfg.gfn_module,
            feature_dim=feature_dim,
            label_dim=label_dim,
            hidden_dim=cfg.gfn_hidden_dim,
            output_dim=1,
            n_hidden_layers=cfg.gfn_n_hidden_layers,
            activation_fn=cfg.gfn_activation_fn,
            torso=torso,
        )
    if cfg.gfn_module == "poolnet":
        return PoolNet(
            feature_dim=feature_dim,
            pool_size=pool_size,
            output_final_state=output_final_state,
            hidden_dim=cfg.gfn_hidden_dim,
            n_hidden_layers=cfg.gfn_n_hidden_layers,
            activation_fn=cfg.gfn_activation_fn,
            torso=torso,
        )
    elif cfg.gfn_module == "querynet":
        return QueryNet(
            feature_dim=feature_dim,
            pool_size=pool_size,
            output_final_state=output_final_state,
            hidden_dim=cfg.gfn_hidden_dim,
            n_hidden_layers=cfg.gfn_n_hidden_layers,
            activation_fn=cfg.gfn_activation_fn,
            torso=torso,
        )
    elif cfg.gfn_module == "poolandtrainnet":
        return PoolAndTrainNet(
            feature_dim=feature_dim,
            label_dim=label_dim,
            pool_size=pool_size,
            output_final_state=output_final_state,
            hidden_dim=cfg.gfn_hidden_dim,
            n_hidden_layers=cfg.gfn_n_hidden_layers,
            activation_fn=cfg.gfn_activation_fn,
            torso=torso,
        )
    elif cfg.gfn_module == "npt":
        return NPT(
            feature_dim=feature_dim,
            label_dim=label_dim,
            pool_size=pool_size,
            output_final_state=output_final_state,
            feature_dim_hid=cfg.gfn_hidden_dim,
            label_dim_hid=cfg.label_dim_hid,
            n_transformer_layers=cfg.n_transformer_layers,
            n_transformer_heads=cfg.n_transformer_heads,
            torso=torso,
        )
    else:
        raise ValueError(f"Unknown gfn module: {cfg.gfn_module}")


def get_gfn(
    cfg,
    ds_pool,
    proxy_reward_function,
    one_hot_labels=False,
    autoencode=False,
    autoencoder_latent_dim=32,
):
    """Get full GFN model: environment, actions/trajectories sampler, criterion and optimizer"""
    # environment
    env = MiFlowEnv(
        ds_pool=ds_pool,
        proxy_reward_function=proxy_reward_function,
        reward_temp=cfg.gfn_reward_temp,
        stop_after=cfg.query_size,
        preprocessor_str=cfg.gfn_module,
        device_str=cfg.device,
        compute_full_dist=cfg.gfn_compute_full_dist,
        one_hot_labels=one_hot_labels,
        autoencode=autoencode,
        autoencoder_latent_dim=autoencoder_latent_dim,
    )
    feature_dim = autoencoder_latent_dim if autoencode else ds_pool[0][0].shape[0]
    label_dim = (
        cfg.num_classes if one_hot_labels else 1
    )  # one-hot encoding for classification

    # loss and samplers
    if cfg.gfn_loss in ("tb", "subtb", "db", "var"):
        logit_PF = LogitPFEstimator(
            env=env,
            module=get_gfn_module(
                cfg,
                feature_dim=feature_dim,
                label_dim=label_dim,
                pool_size=len(ds_pool),
                output_final_state=True,
            ),
        )
        logit_PB = LogitPBEstimator(
            env=env,
            module=get_gfn_module(
                cfg,
                feature_dim=feature_dim,
                label_dim=label_dim,
                pool_size=len(ds_pool),
                output_final_state=False,
                torso=logit_PF.module.torso,  # To share parameters between PF and PB
            ),
        )

        actions_sampler = DiscreteActionsSampler(
            estimator=logit_PF,
            temperature=cfg.gfn_sample_temp,
            epsilon=cfg.gfn_sample_eps,
        )
        trajectories_sampler = TrajectoriesSampler(
            env=env, actions_sampler=actions_sampler
        )

        if cfg.gfn_loss == "tb":
            logZ = LogZEstimator(torch.tensor(0.0))
            parametrization = TBParametrization(logit_PF, logit_PB, logZ)
            criterion = TrajectoryBalance(parametrization=parametrization)
        elif cfg.gfn_loss == "var":
            parametrization = PFBasedParametrization(logit_PF, logit_PB)
            criterion = LogPartitionVarianceLoss(parametrization=parametrization)
        elif cfg.gfn_loss in ("subtb", "db"):
            logF = LogStateFlowEstimator(
                env=env,
                module=get_gfn_module(
                    cfg,
                    feature_dim=feature_dim,
                    label_dim=label_dim,
                    pool_size=len(ds_pool),
                    state_flow=True,
                    torso=logit_PF.module.torso,
                ),
                forward_looking=cfg.gfn_forward_looking,
            )
            if cfg.gfn_loss == "db":
                parametrization = DBParametrization(logit_PF, logit_PB, logF)
                criterion = DetailedBalance(parametrization=parametrization)
            else:
                parametrization = SubTBParametrization(logit_PF, logit_PB, logF)
                criterion = SubTrajectoryBalance(
                    parametrization=parametrization, lamda=cfg.gfn_lamda
                )

    elif cfg.gfn_loss == "fm":
        logF = LogEdgeFlowEstimator(
            env=env,
            module=get_gfn_module(
                cfg,
                feature_dim=feature_dim,
                label_dim=label_dim,
                pool_size=len(ds_pool),
                output_final_state=True,
            ),
        )
        parametrization = FMParametrization(logF=logF)

        actions_sampler = DiscreteActionsSampler(
            estimator=logF,
            temperature=cfg.gfn_sample_temp,
            epsilon=cfg.gfn_sample_eps,
        )
        trajectories_sampler = TrajectoriesSampler(
            env=env, actions_sampler=actions_sampler
        )

        criterion = FlowMatching(parametrization=parametrization)
    else:
        raise ValueError(f"Unknown GFN loss: {cfg.gfn_loss}")

    # optimisation
    params = [
        {
            "params": [
                val for key, val in parametrization.parameters.items() if key != "logZ"
            ],
            "lr": cfg.gfn_lr,
        },
        {
            "params": [
                val for key, val in parametrization.parameters.items() if key == "logZ"
            ],
            "lr": 0.1,
        },
    ]
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, params)

    if cfg.gfn_opt == "adam":
        optimizer = torch.optim.Adam(params)
    else:
        raise ValueError(f"Unknown GFN optimizer: {cfg.gfn_opt}")

    return env, trajectories_sampler, criterion, optimizer, scheduler


def freeze_gfn(cfg, env, criterion, optimizer, proxy_reward_function):
    # just retrain the final layer of the gfn
    for key, param in criterion.parametrization.parameters.items():
        if ("last_layer" not in key) and ("logZ" not in key):
            param.requires_grad = False
    params = [
        {
            "params": [
                val
                for key, val in criterion.parametrization.parameters.items()
                if key != "logZ"
            ],
            "lr": cfg.gfn_lr,
        },
        {"params": [criterion.parametrization.parameters["logZ"]], "lr": 0.1},
    ]
    optimizer = torch.optim.Adam(params)
    # update the reward function
    env.proxy_reward_function = proxy_reward_function

    return optimizer


def compute_true_and_empirical_pmf(
    env, parametrization, n_validation_samples, visited_terminating_states=None
):
    """Compute the true and empirical pmf of the given parametrization on the given environment."""
    print("Computing true and empirical pmf...")
    true_logZ = env.log_partition
    true_dist_pmf = env.true_dist_pmf
    if isinstance(true_dist_pmf, torch.Tensor):
        true_dist_pmf = true_dist_pmf.cpu()
    else:
        # The environment does not implement a true_dist_pmf property, nor a log_partition property
        # We cannot validate the parametrization
        return {}

    logZ = None
    if isinstance(parametrization, TBParametrization):
        logZ = parametrization.logZ.tensor.item()
    if visited_terminating_states is None:
        final_states_dist = parametrization.P_T(env, n_validation_samples)
    else:
        final_states_dist = EmpiricalTerminatingStatesDistribution(
            env, visited_terminating_states[-n_validation_samples:]
        )
        n_visited_states = visited_terminating_states.batch_shape[0]
        n_validation_samples = min(n_visited_states, n_validation_samples)

    final_states_dist_pmf = final_states_dist.pmf()

    return true_dist_pmf, final_states_dist_pmf, true_logZ, logZ


def validate_gfn(
    env,
    parametrization,
    n_validation_samples=1000,
    visited_terminating_states=None,
):
    """Ref: Based of validate function in gfn package.
    Evaluates the current parametrization on the given environment.
    This is for environments with known target reward. The validation is done by computing the l1 distance between the
    learned empirical and the target distributions.

    Args:
        env: The environment to evaluate the parametrization on.
        parametrization: The parametrization to evaluate.
        n_validation_samples: The number of samples to use to evaluate the pmf.
        visited_terminating_states: The terminating states visited during training. If given, the pmf is obtained from
            these last n_validation_samples states. Otherwise, n_validation_samples are resampled for evaluation.

    Returns:
        Dict[str, float]: A dictionary containing the l1 validation metric. If the parametrization is a TBParametrization,
        i.e. contains LogZ, then the (absolute) difference between the learned and the target LogZ is also returned in the
        dictionary.
    """

    (
        true_dist_pmf,
        final_states_dist_pmf,
        true_logZ,
        logZ,
    ) = compute_true_and_empirical_pmf(
        env, parametrization, n_validation_samples, visited_terminating_states
    )

    l1_dist = (final_states_dist_pmf - true_dist_pmf).abs().mean().item()
    # jenson-shannon divergence
    m = 0.5 * (final_states_dist_pmf + true_dist_pmf)
    eps = 1e-8
    js_div = (
        0.5
        * (
            F.kl_div(torch.log(final_states_dist_pmf + eps), m, reduction="batchmean")
            + F.kl_div(torch.log(true_dist_pmf + eps), m, reduction="batchmean")
        ).item()
    )
    validation_info = {"l1_dist": l1_dist, "js_div": js_div}
    if logZ is not None:
        validation_info["logZ_diff"] = abs(logZ - true_logZ)
    return validation_info


def gfn_choose_next_query(cfg, trajectories_sampler):
    """choose next query batch from pool set"""
    # instantise a new sampler without temperature and epsilon random.
    query_sampler = TrajectoriesSampler(
        env=trajectories_sampler.env,
        actions_sampler=DiscreteActionsSampler(
            estimator=trajectories_sampler.actions_sampler.estimator
        ),
    )
    # batch sampling to avoid memory issues
    query_idx = None
    best_reward = 0
    for _ in range(cfg.n_stoch_queries // cfg.gfn_batch_size):
        trajectories = query_sampler.sample(n_trajectories=cfg.gfn_batch_size)
        # choose highest reward batch
        reward, traj_idx = torch.max(trajectories.log_rewards, dim=0)
        if reward > best_reward:
            best_reward = reward
            last_state = trajectories.last_states[traj_idx].states_tensor
            # print(f"Sampled last state: {last_state}")
            query_idx = torch.nonzero(last_state, as_tuple=True)[0].cpu().numpy()

    # print(
    #     f"Sampled {len(trajectories)} trajectories with rewards: {trajectories.log_rewards}"
    # )  # TODO save this to wandb?

    # TODO log jmi and time to wandb

    return list(query_idx)


def train_gfn(
    cfg,
    env,
    trajectories_sampler,
    criterion,
    optimizer,
    scheduler=None,
    validation_interval=-1,
    step=0,
):
    """Train GFN"""
    start_time = time.time()
    loss_curve = []
    states_visited = 0
    states_visited_curve = []
    time_curve = []
    if cfg.gfn_compute_full_dist:
        val_step_curve = []
        l1_dist_curve = []
        logZ_diff_curve = []
        js_div_curve = []

    visited_terminating_states = (
        env.States.from_batch_shape((0,)) if not cfg.resample_for_validation else None
    )

    # TODO add replay buffer/scheduler etc. to help training
    num_epochs = int(cfg.gfn_n_iterations * (cfg.gfn_iters_decay_rate**step))
    for i in trange(num_epochs):
        trajectories = trajectories_sampler.sample(n_trajectories=cfg.gfn_batch_size)
        training_samples = trajectories_to_training_samples(trajectories, criterion)

        optimizer.zero_grad()
        loss = criterion(training_samples)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if visited_terminating_states is not None:
            visited_terminating_states.extend(trajectories.last_states)

        loss_curve.append(loss.item())
        states_visited += len(trajectories)
        states_visited_curve.append(states_visited)
        time_curve.append(time.time() - start_time)
        if validation_interval > 0 and i % validation_interval == 0:
            if cfg.gfn_compute_full_dist:
                validation_info = validate_gfn(
                    env,
                    criterion.parametrization,
                    cfg.validation_samples,
                    visited_terminating_states,
                )
                l1_dist_curve.append(validation_info["l1_dist"])
                js_div_curve.append(validation_info["js_div"])
                logZ_diff_curve.append(validation_info.get("logZ_diff", None))
                val_step_curve.append(i)
            else:
                validation_info = {}
            tqdm.write(
                f"{i}: loss: {loss.item()}, states_visited: {states_visited}, time: {time_curve[-1]},"
                f"l1_dist: {validation_info.get('l1_dist', None)}, logZ_diff: {validation_info.get('logZ_diff', None)}, js_div: {validation_info.get('js_div', None)}"
            )
    if validation_interval > 0 and cfg.gfn_compute_full_dist:
        true_dist_pmf, final_dist_pmf, _, _ = compute_true_and_empirical_pmf(
            env,
            criterion.parametrization,
            cfg.validation_samples,
            visited_terminating_states,
        )
        # save only the non-zero entries with the corresponding state indices
        non_zero_indices = torch.nonzero(true_dist_pmf).squeeze()
        true_dist_pmf = [
            [x, y]
            for (x, y) in zip(
                non_zero_indices.numpy().tolist(),
                true_dist_pmf[non_zero_indices].numpy().tolist(),
            )
        ]
        non_zero_indices = torch.nonzero(final_dist_pmf).squeeze()
        final_dist_pmf = [
            [x, y]
            for (x, y) in zip(
                non_zero_indices.numpy().tolist(),
                final_dist_pmf[non_zero_indices].numpy().tolist(),
            )
        ]
        return (
            loss_curve,
            states_visited_curve,
            time_curve,
            val_step_curve,
            l1_dist_curve,
            js_div_curve,
            logZ_diff_curve,
            true_dist_pmf,
            final_dist_pmf,
        )
    else:
        return (
            loss_curve,
            states_visited_curve,
            time_curve,
            [None],
            [None],
            [None],
            [None],
            [None],
            [None],
        )
