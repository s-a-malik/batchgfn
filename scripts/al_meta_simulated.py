"""Train an amortised gfn model using samples from a distribution over functions and training sets.
Only for simulatedMeta dataset currently.
"""
from functools import partial
from itertools import product
from pathlib import Path

import wandb

import numpy as np

import torch
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split


from batchgfn.datasets import SimulatedMeta, TensorDataset
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
    """Evaluate acquisition function and plot results.
    Also compare to MI of baseline acquisitions
    """
    # sample means from pool set
    query_points = ds_pool[query_idx][0].to(cfg.device)
    mi = model.mutual_information(query_points).squeeze().cpu().float()
    # get test loss TODO: batching for larger datasets
    test_loss = (
        torch.nn.MSELoss()(
            model.predict(ds_test[:][0].to(cfg.device)).mean,
            ds_test[:][1].to(cfg.device),
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
        f_samples = model.sample_mean(ds_pool[:][0].to(cfg.device), cfg.sample_mi_n)
        # choose query batch
        query_idx = string_to_func[cfg.al_method](f_samples, batch_size=cfg.query_size)
    else:
        x_pool = ds_pool[:][0].to(cfg.device)
        query_idx = string_to_func[cfg.al_method](
            x_pool=x_pool, model=model, batch_size=cfg.query_size
        )

    return query_idx


def meta_evaluate(cfg, meta_test_dataset, step, env=None, trajectories_sampler=None):
    # evaluate gfn or baselines on meta-test set
    # random seed, same for each eval so same functions are sampled
    np.random.seed(0)
    test_func_idxs = np.random.randint(
        0, len(meta_test_dataset), cfg.meta_test_n_episodes
    )

    to_log = {"test_func_idx": test_func_idxs.tolist()}
    final_test_losses = []
    first_step_mi = []
    for meta_test_step, func_idx in enumerate(test_func_idxs):
        print(f"\n\nMeta test step: {meta_test_step}, meta test idx: ", func_idx)
        x_all, y_all = meta_test_dataset[func_idx]
        # randomly split into pool and test sets - same for each eval
        x_pool, x_test, y_pool, y_test = train_test_split(
            x_all, y_all, test_size=cfg.test_size, random_state=meta_test_step
        )
        # make into dataset
        ds_pool = TensorDataset(x_pool, y_pool)
        ds_test = TensorDataset(x_test, y_test)
        labelled_idx = []
        if env:
            env.update_pool_dataset(ds_pool)
            env.reset_labelled_dataset()
        model = GP(
            train_x=ds_pool[labelled_idx][0],
            train_y=ds_pool[labelled_idx][1],
            kernel_str=cfg.gp_kernel,
        )
        model.to(cfg.device)
        # same seed set for each eval
        np.random.seed(cfg.seed)
        query_idx = np.random.choice(len(ds_pool), cfg.seed_size, replace=False)
        mi, test_loss, fig = evaluate_and_plot_acquisition(
            cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=0
        )
        meta_test_mi = [[0, mi]]
        meta_test_loss = [[0, test_loss]]
        meta_test_num_labelled = [[0, len(labelled_idx)]]
        # Note: we plot the *next* acquisition, so the test loss is for the current model before these points are added
        to_log.update(
            {
                f"meta_test_{meta_test_step}_acquisition_plot_0": wandb.Image(fig),
            }
        )
        # add to labelled set
        labelled_idx += list(query_idx)
        for al_step in range(1, cfg.n_queries + 1):
            # train the model on all the training data
            print(f"Training set size: {len(labelled_idx)}")
            model = model = GP(
                train_x=ds_pool[labelled_idx][0],
                train_y=ds_pool[labelled_idx][1],
                kernel_str=cfg.gp_kernel,
            )
            trainer = pl.Trainer(
                max_epochs=cfg.num_gp_epochs,
                # accelerator="gpu" if cfg.device != "cpu" else None,   # TODO
                default_root_dir=cfg.output_dir,
                enable_model_summary=False,
            )
            trainer.fit(model)
            model.to(cfg.device)
            if cfg.al_method == "gfn":
                env.update_labelled_dataset(query_idx)
                env.proxy_reward_function = partial(
                    model.mutual_information,
                    sample=cfg.sample_mi,
                    num_samples=cfg.sample_mi_n,
                )
                query_idx = gfn_choose_next_query(cfg, trajectories_sampler)
            else:
                query_idx = baseline_choose_next_query(
                    cfg, model, ds_pool, labelled_idx
                )
                print(f"labelled_idx: {labelled_idx}")
                print(f"Sampled new query_idx: {query_idx}")
            mi, test_loss, fig = evaluate_and_plot_acquisition(
                cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=al_step
            )
            print(
                f"\nSTEP {al_step} SUMMARY:\nMutual info of batch: {mi}, Test loss: {test_loss}"
            )
            meta_test_mi.append([al_step, mi])
            meta_test_loss.append([al_step, test_loss])
            meta_test_num_labelled.append([al_step, len(labelled_idx)])
            to_log.update(
                {
                    f"meta_test_{meta_test_step}_acquisition_plot_{al_step}": wandb.Image(
                        fig
                    ),
                }
            )
            wandb.log(to_log, step=step)
            # add to labelled set
            labelled_idx += list(query_idx)
        final_test_losses.append(test_loss)
        first_step_mi.append(meta_test_mi[1][1])
        wandb.log(
            {
                f"meta_test_{meta_test_step}_query_mi": wandb.Table(
                    data=meta_test_mi, columns=["al_step", "query_mi"]
                ),
                f"meta_test_{meta_test_step}_test_loss": wandb.Table(
                    data=meta_test_loss, columns=["al_step", "test_loss"]
                ),
                f"meta_test_{meta_test_step}_num_labelled": wandb.Table(
                    data=meta_test_num_labelled, columns=["al_step", "num_labelled"]
                ),
            },
            step=step,
        )
    # TODO better summary metrics?
    avg_mi = np.mean(first_step_mi)
    avg_test_loss = np.mean(final_test_losses)

    print(f"meta test summary: avg_mi: {avg_mi}, avg_test_loss: {avg_test_loss}")
    wandb.log(
        {
            "avg_first_step_meta_test_mi": avg_mi,
            "avg_final_meta_test_loss": avg_test_loss,
        },
        step=step,
    )


def meta_simulated_experiment(cfg):
    # check using poolandtrainnet if gfn
    if cfg.al_method == "gfn":
        assert cfg.gfn_module not in ["poolnet", "querynet"]

    # Set up wandb
    run = wandb.init(project=cfg.wandb_project, group=cfg.wandb_group, save_code=True)
    wandb.config.update(cfg)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up metadatasets
    meta_train_dataset = SimulatedMeta(
        num_samples=cfg.meta_train_size,
        num_points=cfg.pool_size + cfg.test_size,
        seed=cfg.seed,
    )
    meta_test_dataset = SimulatedMeta(
        num_samples=cfg.meta_test_size,
        num_points=cfg.pool_size + cfg.test_size,
        seed=cfg.seed + 1,
    )
    env, trajectories_sampler, criterion, optimizer = None, None, None, None

    if cfg.al_method == "gfn":
        # try except to stop training
        try:
            # meta-train loop
            for meta_train_step in range(cfg.meta_train_n_episodes):
                print(f"\n\nMeta-train step {meta_train_step}\n")
                # Sample a function from meta-dataset
                func_idx = np.random.randint(0, len(meta_train_dataset))
                to_log = {"func_idx": func_idx}
                x_all, y_all = meta_train_dataset[func_idx]
                # randomly split into pool and test sets
                x_pool, x_test, y_pool, y_test = train_test_split(
                    x_all, y_all, test_size=cfg.test_size
                )
                # make into dataset
                ds_pool = TensorDataset(x_pool, y_pool)
                ds_test = TensorDataset(x_test, y_test)
                # randomly sample a seed set from pool
                indices = list(range(len(ds_pool)))
                _, labelled_idx = train_test_split(
                    indices,
                    test_size=np.random.randint(
                        cfg.meta_train_min_seed_size, cfg.meta_train_max_seed_size + 1
                    ),
                )
                print(f"func_idx: {func_idx}, seed set size: {len(labelled_idx)}")
                # Train GP on seed set TODO this is going to be slow, parallelise
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

                # Train gfn to sample MI estimate
                if not env:
                    (
                        env,
                        trajectories_sampler,
                        criterion,
                        optimizer,
                        scheduler,
                    ) = get_gfn(
                        cfg,
                        ds_pool=ds_pool,
                        proxy_reward_function=partial(
                            model.mutual_information,
                            sample=cfg.sample_mi,
                            num_samples=cfg.sample_mi_n,
                        ),
                    )
                env.update_pool_dataset(ds_pool)
                env.reset_labelled_dataset()
                env.update_labelled_dataset(labelled_idx)
                env.proxy_reward_function = partial(
                    model.mutual_information,
                    sample=cfg.sample_mi,
                    num_samples=cfg.sample_mi_n,
                )
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
                    step=0,
                )
                # choose points to query using GFN sampler
                query_idx = gfn_choose_next_query(cfg, trajectories_sampler)
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
                # evaluate
                mi, test_loss, fig = evaluate_and_plot_acquisition(
                    cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=0
                )
                print(
                    f"\nmeta-train step {meta_train_step} SUMMARY:\nMutual info of batch: {mi}, Test loss: {test_loss}"
                )
                to_log.update(
                    {
                        "query_mi": mi,
                        "test_loss": test_loss,
                        "num_labelled": len(labelled_idx),
                        "acquisition_plot": wandb.Image(fig),
                    }
                )
                wandb.log(to_log, step=meta_train_step)

                # validate gfn on meta-test set periodically
                if meta_train_step % cfg.meta_eval_interval == 0:
                    meta_evaluate(
                        cfg,
                        meta_test_dataset,
                        step=meta_train_step,
                        env=env,
                        trajectories_sampler=trajectories_sampler,
                    )
                    # save model
                    criterion.parametrization.save_state_dict(path=wandb.run.dir)
        except KeyboardInterrupt:
            print("Keyboard interrupt, saving model and stopping training.")
            criterion.parametrization.save_state_dict(path=wandb.run.dir)
    # just evaluate on meta-test set for baselines
    print("FINAL EVALUATION on meta-test set:")
    step = cfg.meta_train_n_episodes if cfg.al_method == "gfn" else 0
    meta_evaluate(cfg, meta_test_dataset, step, env, trajectories_sampler)

    run.finish()
