"""
Active learning experiments on simulated data. 
Acquire batches of points from a pool set, train a model on the acquired points,
and evaluate the model on a test set. 
Amortise by looking ahead to a hallucinated batch of points based on the model's predictions. 
"""

from copy import deepcopy
from functools import partial
from pathlib import Path

import wandb

import torch
import pytorch_lightning as pl

import numpy as np

from batchgfn.datasets import Simulated
from batchgfn.acquisitions import (
    batch_bald_cov,
    batch_bald_cov_gp,
    bald,
    bald_gp,
    stochastic_bald,
    stochastic_bald_gp,
    stochastic_batch_bald,
    stochastic_batch_bald_gp,
    random_multi,
    random_multi_gp,
)
from batchgfn.models import GP
from batchgfn.evaluation import plotting
from batchgfn.gflow import get_gfn, train_gfn, gfn_choose_next_query


def evaluate_and_plot_acquisition(
    cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step
):
    """TODO this is for debug GP only currently. Make more general."""
    # sample means from pool set
    query_points = ds_pool[query_idx][0].to(cfg.device)
    mi = (
        model.mutual_information(
            query_points, sample=cfg.sample_mi, num_samples=cfg.sample_mi_n
        )
        .squeeze()
        .cpu()
        .float()
    )
    # get test loss TODO: batching for larger datasets
    test_loss = (
        torch.nn.MSELoss()(
            model.predict(ds_test.data.to(cfg.device)).mean,
            ds_test.targets.to(cfg.device),
        )
        .data.cpu()
        .item()
    )
    # plot
    test_x = torch.linspace(-5, 5, 100, device=cfg.device)
    lower, upper = model.confidence_region(test_x)
    f_samples = model.sample_mean(test_x, cfg.num_gp_samples)
    fig = plotting.plot_acquisition(
        x=test_x.squeeze(-1).cpu().numpy(),
        f_samples=f_samples.cpu(),
        lower=lower.cpu().numpy(),
        upper=upper.cpu().numpy(),
        x_train=ds_pool[labelled_idx][0].squeeze(-1),
        y_train=ds_pool[labelled_idx][1],
        x_acquired=ds_pool[query_idx][0].squeeze(-1),
        y_acquired=ds_pool[query_idx][1],
        loss=test_loss,
        mi=mi,
        step=step,
        title=f"{cfg.al_method} acquisition",
        output_file=None,
    )
    return mi, test_loss, fig


def baseline_choose_next_query(cfg, model, ds_pool, labelled_idx):
    """choose next query batch from pool set"""
    string_to_func = {
        "bald": bald_gp if not cfg.sample_mi else bald,
        "batchbald": batch_bald_cov_gp if not cfg.sample_mi else batch_bald_cov,
        "stoch_bald": partial(
            stochastic_bald_gp, reps=cfg.n_stoch_queries, temp=cfg.stoch_temp
        )
        if not cfg.sample_mi
        else partial(stochastic_bald, reps=cfg.n_stoch_queries, temp=cfg.stoch_temp),
        "stoch_batchbald": partial(
            stochastic_batch_bald_gp, reps=cfg.n_stoch_queries, temp=cfg.stoch_temp
        )
        if not cfg.sample_mi
        else partial(
            stochastic_batch_bald, reps=cfg.n_stoch_queries, temp=cfg.stoch_temp
        ),
        "random": partial(random_multi_gp, labelled_idx=labelled_idx, reps=1)
        if not cfg.sample_mi
        else partial(random_multi, labelled_idx=labelled_idx, reps=1),
        "stoch_random": partial(
            random_multi_gp, labelled_idx=labelled_idx, reps=cfg.n_stoch_queries
        )
        if not cfg.sample_mi
        else partial(random_multi, labelled_idx=labelled_idx, reps=cfg.n_stoch_queries),
    }

    if cfg.sample_mi:
        # sample means from pool set
        f_samples = (
            model.sample_mean(ds_pool[:][0].to(cfg.device), cfg.sample_mi_n)
            # .cpu()
            # .numpy()
        )
        # choose query batch
        query_idx = string_to_func[cfg.al_method](
            f_samples,
            batch_size=cfg.query_size,
        )
    else:
        x_pool = ds_pool[:][0].to(cfg.device)
        query_idx = string_to_func[cfg.al_method](
            x_pool=x_pool,
            model=model,
            batch_size=cfg.query_size,
        )

    return list(query_idx)


def lookahead_training(
    cfg,
    model,
    ds_pool,
    ds_test,
    labelled_idx,
    env,
    trajectories_sampler,
    criterion,
    optimizer,
    scheduler,
    al_step=1,
):
    """
    Fake acquire points on a hallucinated batch of points, train model on acquired points,
    """
    lookahead_labelled_idx = labelled_idx.copy()
    unlabelled_idx = [i for i in range(len(ds_pool)) if i not in lookahead_labelled_idx]
    for lookahead_step in range(1, cfg.meta_train_n_episodes + 1):
        print(f"\nLOOKAHEAD STEP {lookahead_step}")
        # choose next query batch from pool set
        # TODO vary size of lookahead query batch?
        # TODO multiple step lookahead?
        if cfg.lookahead_method == "random":
            lookahead_idx = np.random.choice(
                unlabelled_idx, cfg.query_size, replace=False
            )
        elif cfg.lookahead_method == "gfn":
            # this is training trajectory sampler (eps random etc.)
            trajectories = trajectories_sampler.sample(n_trajectories=1)
            last_state = trajectories.last_states[0].states_tensor
            lookahead_idx = torch.nonzero(last_state, as_tuple=True)[0].cpu().numpy()
        else:
            raise NotImplementedError(
                f"Lookahead method {cfg.lookahead_method} not implemented"
            )

        # make copy of dataset
        ds_pool_hallucinated = deepcopy(ds_pool)
        if not cfg.use_true_lookahead:
            # sample joint points from GP
            hallucinated_labels = (
                model.sample_mean(
                    ds_pool_hallucinated[lookahead_idx][0].to(cfg.device), 1
                )
                .squeeze()
                .cpu()
            )
            # update pool tensors with hallucinated labels
            ds_pool_hallucinated.targets[lookahead_idx] = hallucinated_labels
        # add to labelled set
        training_idx = lookahead_labelled_idx + list(lookahead_idx)
        # train model on acquired points
        print(f"Lookahead Training set size: {len(training_idx)}")
        lookahead_model = GP(
            train_x=ds_pool_hallucinated[training_idx][0],
            train_y=ds_pool_hallucinated[training_idx][1],
            kernel_str=cfg.gp_kernel,
        )
        trainer = pl.Trainer(
            max_epochs=cfg.num_gp_epochs,
            # accelerator="gpu" if cfg.device != "cpu" else None,   # TODO
            default_root_dir=cfg.output_dir,
            enable_model_summary=False,
        )
        trainer.fit(lookahead_model)
        lookahead_model.to(cfg.device)

        env.reset_labelled_dataset()
        env.update_labelled_dataset(training_idx)
        env.update_pool_dataset(ds_pool_hallucinated)
        # update the reward function
        env.proxy_reward_function = partial(
            lookahead_model.mutual_information,
            sample=cfg.sample_mi,
            num_samples=cfg.sample_mi_n,
        )
        # train the GFN using new reward
        (
            loss_curve,
            states_visited_curve,
            time_curve,
            val_step_curve,
            l1_dist_curve,
            js_div_curve,
            logZ_diff_curve,
            true_dist_pmf,
            final_dist_pmf,
        ) = train_gfn(
            cfg,
            env,
            trajectories_sampler,
            criterion,
            optimizer,
            scheduler,
            validation_interval=-1,  # don't validate lookahead GFN
            step=0,
        )

        print(
            f"GFN final loss: {loss_curve[-1]}, GFN final states visited: {states_visited_curve[-1]}, GFN train time: {time_curve[-1]},"
            f"GFN l1 dist: {l1_dist_curve[-1]}, GFN logZ diff: {logZ_diff_curve[-1]}, GFN js div: {js_div_curve[-1]}"
        )
        to_log = {}
        train_step = list(range(len(loss_curve)))
        gfn_loss = [[x, y] for (x, y) in zip(train_step, loss_curve)]
        gfn_loss_table = wandb.Table(data=gfn_loss, columns=["train_step", "loss"])
        gfn_states = [[x, y] for (x, y) in zip(train_step, states_visited_curve)]
        gfn_states_table = wandb.Table(
            data=gfn_states, columns=["train_step", "states_visited"]
        )
        to_log.update(
            {
                "lookahead_gfn_final_loss": loss_curve[-1],
                "lookahead_gfn_final_states_visited": states_visited_curve[-1],
                "lookahead_gfn_states_visited_curve": states_visited_curve,
                "lookahead_gfn_train_time": time_curve[-1],
                "lookahead_gfn_loss_curve": wandb.plot.line(
                    gfn_loss_table, "train_step", "loss", title="GFN loss curve"
                ),
                "lookahead_gfn_states_visited_curve": wandb.plot.line(
                    gfn_states_table,
                    "train_step",
                    "states_visited",
                    title="GFN states visited curve",
                ),
            }
        )
        # evaluate
        query_idx = gfn_choose_next_query(cfg, trajectories_sampler)
        mi, test_loss, fig = evaluate_and_plot_acquisition(
            cfg,
            lookahead_model,
            ds_pool_hallucinated,
            ds_test,
            training_idx,
            query_idx,
            step=lookahead_step,
        )
        print(
            f"\nLOOKAHEAD STEP {lookahead_step} SUMMARY:\nMutual info of batch: {mi}, Test loss: {test_loss}"
        )
        to_log.update(
            {
                "lookahead_query_mi": mi,
                "lookahead_test_loss": test_loss,
                "lookahead_num_labelled": len(labelled_idx),
                "lookahead_acquisition_plot": wandb.Image(fig),
            }
        )
        wandb.log(to_log, step=(al_step * cfg.meta_train_n_episodes) + lookahead_step)

        # save model periodically
        if cfg.gfn_validation_interval % lookahead_step == 0:
            criterion.parametrization.save_state_dict(path=wandb.run.dir)


def lookahead_simulated_experiment(cfg):
    """run experiments."""
    # check using poolandtrainnet if gfn
    if cfg.al_method == "gfn":
        assert cfg.gfn_module not in ["poolnet", "querynet"]

    # set up wandb
    run = wandb.init(project=cfg.wandb_project, group=cfg.wandb_group, save_code=True)
    wandb.config.update(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # set up data
    labelled_idx = []  # indices of labelled pool data used for training
    ds_pool = Simulated(n=cfg.pool_size, seed=cfg.seed)
    ds_test = Simulated(n=cfg.test_size, seed=cfg.seed + 1)
    print(f"Pool set size: {len(ds_pool)}, Test set size: {len(ds_test)}")

    # set up (prior) model
    model = GP(
        train_x=ds_pool[labelled_idx][0],
        train_y=ds_pool[labelled_idx][1],
        kernel_str=cfg.gp_kernel,
    )
    model.to(cfg.device)
    print(f"Model: {model}")

    # acquire seed set randomly
    query_idx = np.random.choice(len(ds_pool), cfg.seed_size, replace=False)
    mi, test_loss, fig = evaluate_and_plot_acquisition(
        cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=0
    )
    print(f"\nINIT:\nMutual info of batch: {mi}, Test loss: {test_loss}")
    # Note: we plot the *next* acquisition, so the test loss is for the current model before these points are added
    to_log = {
        "query_mi": mi,
        "test_loss": test_loss,
        "num_labelled": len(labelled_idx),
        "acquisition_plot": wandb.Image(fig),
    }
    wandb.log(to_log, step=0)
    # add to labelled set
    labelled_idx += list(query_idx)

    if cfg.al_method == "gfn":
        # set up GFN
        env, trajectories_sampler, criterion, optimizer, scheduler = get_gfn(
            cfg,
            ds_pool=ds_pool,
            proxy_reward_function=partial(
                model.mutual_information,
                sample=cfg.sample_mi,
                num_samples=cfg.sample_mi_n,
            ),
        )
        env.update_labelled_dataset(query_idx)
        print(f"GFN: {env}, {trajectories_sampler}, {criterion}, {optimizer}")

    # active learning loop. Note: the first step is training on the randomly sampled seed set
    for al_step in range(1, cfg.n_queries + 1):
        if len(labelled_idx) + cfg.query_size > len(ds_pool):
            print("No more points to query.")
            break

        print(f"\nActive learning step {al_step}:")
        to_log = {}
        # train the model on all the training data
        print(f"Training set size: {len(labelled_idx)}")
        model = GP(
            train_x=ds_pool[labelled_idx][0],
            train_y=ds_pool[labelled_idx][1],
            kernel_str=cfg.gp_kernel,
        )
        trainer = pl.Trainer(
            max_epochs=cfg.num_gp_epochs,
            # accelerator="gpu" if cfg.device != "cpu" else None,   # TODO
            default_root_dir=output_dir,
            enable_model_summary=False,
        )
        trainer.fit(model)
        model.to(cfg.device)
        if cfg.al_method == "gfn":
            # update the reward function
            env.proxy_reward_function = partial(
                model.mutual_information,
                sample=cfg.sample_mi,
                num_samples=cfg.sample_mi_n,
            )
            try:
                # train the GFN using new reward
                (
                    loss_curve,
                    states_visited_curve,
                    time_curve,
                    val_step_curve,
                    l1_dist_curve,
                    js_div_curve,
                    logZ_diff_curve,
                    true_dist_pmf,
                    final_dist_pmf,
                ) = train_gfn(
                    cfg,
                    env,
                    trajectories_sampler,
                    criterion,
                    optimizer,
                    scheduler,
                    cfg.gfn_validation_interval,
                    step=al_step - 1,
                )
            except KeyboardInterrupt:
                print("Keyboard interrupt. Exiting training.")
                pass

            # choose points to query using GFN sampler
            query_idx = gfn_choose_next_query(cfg, trajectories_sampler)
            if len(loss_curve) > 0:
                print(
                    f"GFN final loss: {loss_curve[-1]}, GFN final states visited: {states_visited_curve[-1]}, GFN train time: {time_curve[-1]},"
                    f"GFN l1 dist: {l1_dist_curve[-1]}, GFN logZ diff: {logZ_diff_curve[-1]}, GFN js div: {js_div_curve[-1]}"
                )
                train_step = list(range(len(loss_curve)))
                gfn_loss = [[x, y] for (x, y) in zip(train_step, loss_curve)]
                gfn_loss_table = wandb.Table(
                    data=gfn_loss, columns=["train_step", "loss"]
                )
                gfn_states = [
                    [x, y] for (x, y) in zip(train_step, states_visited_curve)
                ]
                gfn_states_table = wandb.Table(
                    data=gfn_states, columns=["train_step", "states_visited"]
                )
                if cfg.gfn_compute_full_dist:
                    gfn_l1_dist = [
                        [x, y] for (x, y) in zip(val_step_curve, l1_dist_curve)
                    ]
                    gfn_l1_dist_table = wandb.Table(
                        data=gfn_l1_dist, columns=["train_step", "l1_dist"]
                    )
                    gfn_js_div = [
                        [x, y] for (x, y) in zip(val_step_curve, js_div_curve)
                    ]
                    gfn_js_div_table = wandb.Table(
                        data=gfn_js_div, columns=["train_step", "js_div"]
                    )
                    gfn_logZ_diff = [
                        [x, y] for (x, y) in zip(val_step_curve, logZ_diff_curve)
                    ]
                    gfn_logZ_diff_table = wandb.Table(
                        data=gfn_logZ_diff, columns=["train_step", "logZ_diff"]
                    )
                    true_dist_pmf_table = wandb.Table(
                        data=true_dist_pmf, columns=["final_state_idx", "true_dist_pmf"]
                    )
                    final_dist_pmf_table = wandb.Table(
                        data=final_dist_pmf,
                        columns=["final_state_idx", "final_dist_pmf"],
                    )
                    to_log.update(
                        {
                            "gfn_l1_dist_curve": wandb.plot.line(
                                gfn_l1_dist_table,
                                "train_step",
                                "l1_dist",
                                title="GFN l1 dist curve",
                            ),
                            "gfn_js_div_curve": wandb.plot.line(
                                gfn_js_div_table,
                                "train_step",
                                "js_div",
                                title="GFN js div curve",
                            ),
                            "gfn_logZ_diff_curve": wandb.plot.line(
                                gfn_logZ_diff_table,
                                "train_step",
                                "logZ_diff",
                                title="GFN logZ diff curve",
                            ),
                            "gfn_true_dist_pmf": wandb.plot.bar(
                                true_dist_pmf_table,
                                "final_state_idx",
                                "true_dist_pmf",
                                title="true dist pmf",
                            ),
                            "gfn_final_dist_pmf": wandb.plot.bar(
                                final_dist_pmf_table,
                                "final_state_idx",
                                "final_dist_pmf",
                                title="GFN final dist pmf",
                            ),
                            "gfn_l1_dist": l1_dist_curve[-1],
                            "gfn_js_div": js_div_curve[-1],
                            "gfn_logZ_diff": logZ_diff_curve[-1],
                        }
                    )
                to_log.update(
                    {
                        "gfn_final_loss": loss_curve[-1],
                        "gfn_final_states_visited": states_visited_curve[-1],
                        "gfn_states_visited_curve": states_visited_curve,
                        "gfn_train_time": time_curve[-1],
                        "gfn_loss_curve": wandb.plot.line(
                            gfn_loss_table, "train_step", "loss", title="GFN loss curve"
                        ),
                        "gfn_states_visited_curve": wandb.plot.line(
                            gfn_states_table,
                            "train_step",
                            "states_visited",
                            title="GFN states visited curve",
                        ),
                    }
                )
        else:
            # choose next query batch from pool set
            query_idx = baseline_choose_next_query(cfg, model, ds_pool, labelled_idx)
        print(f"labelled_idx: {labelled_idx}")
        print(f"Sampled new query_idx: {query_idx}")

        # evaluate
        mi, test_loss, fig = evaluate_and_plot_acquisition(
            cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=al_step
        )
        print(
            f"\nSTEP {al_step} SUMMARY:\nMutual info of batch: {mi}, Test loss: {test_loss}"
        )
        to_log.update(
            {
                "query_mi": mi,
                "test_loss": test_loss,
                "num_labelled": len(labelled_idx),
                "acquisition_plot": wandb.Image(fig),
            }
        )
        # wandb.log(to_log, step=al_step)
        wandb.log(to_log)

        if cfg.al_method == "gfn":
            if cfg.only_seed_lookahead and al_step != 1:
                print("Skipping lookahead training.")
                pass
            else:
                try:
                    # train on possible future functions using predictions
                    lookahead_training(
                        cfg,
                        model,
                        ds_pool,
                        ds_test,
                        labelled_idx,
                        env,
                        trajectories_sampler,
                        criterion,
                        optimizer,
                        scheduler,
                        al_step,
                    )
                except KeyboardInterrupt:
                    print("Keyboard interrupt. Exiting training.")
                    pass
            env.reset_labelled_dataset()
            env.update_labelled_dataset(labelled_idx)
        # add to labelled set
        labelled_idx += list(query_idx)
        if cfg.al_method == "gfn":
            env.update_labelled_dataset(query_idx)

    run.finish()
