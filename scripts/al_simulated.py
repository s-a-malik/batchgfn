"""
Active learning experiments on simulated data. 
Acquire batches of points from a pool set, train a model on the acquired points,
and evaluate the model on a test set.
"""

from functools import partial
from pathlib import Path

import wandb

import torch
import pytorch_lightning as pl

import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from batchgfn.datasets import Simulated, SimulatedCls, TwoBells
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
from batchgfn.gflow import get_gfn, train_gfn, freeze_gfn, gfn_choose_next_query


def evaluate_and_plot_acquisition(
    cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step
):
    """TODO this is for GP use only currently. Make more general."""
    if cfg.experiment == "simulated":
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
    elif cfg.experiment == "simulated_cls":
        query_points = ds_pool[query_idx][0].to(cfg.device)
        mi = (
            model.mutual_information(
                query_points, sample=cfg.sample_mi, num_samples=cfg.sample_mi_n
            )
            .squeeze()
            .cpu()
            .float()
        )
        preds = model.predict(ds_test.data.to(cfg.device)).mean
        y_pred = preds.argmax(-1)
        acc = (
            ((y_pred == ds_test.targets.to(cfg.device)).sum().float() / len(ds_test))
            .cpu()
            .numpy()
        )
        test_loss = (
            torch.nn.CrossEntropyLoss()(preds, ds_test.targets.to(cfg.device))
            .data.cpu()
            .item()
        )
        prec, rec, f1, sup = precision_recall_fscore_support(
            ds_test.targets.numpy(),
            y_pred.cpu().numpy(),
            labels=list(range(cfg.num_classes)),
            average=None,
        )
        print(
            f"Test set: CE loss: {test_loss:.4f}, Precision: {prec}, Recall: {rec}, F1: {f1}"
        )
        non_acquired_idx = [
            i for i in range(len(ds_pool)) if i not in query_idx + labelled_idx
        ]
        fig = plotting.plot_acquisition_2d_cls(
            model,
            x_pool=ds_pool[non_acquired_idx][0].cpu().numpy(),
            x_train=ds_pool[labelled_idx][0].cpu().numpy(),
            y_train=ds_pool[labelled_idx][1].cpu().numpy(),
            x_acquired=ds_pool[query_idx][0].cpu().numpy(),
            y_acquired=ds_pool[query_idx][1].cpu().numpy(),
            loss=test_loss,
            acc=acc,
            mi=mi,
            step=step,
            title=f"{cfg.al_method} acquisition",
            output_file=None,
            device=cfg.device,
        )
        # log per class metrics
        wandb.log(
            {
                "confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=ds_test.targets.cpu().numpy(),
                    preds=y_pred.cpu().numpy(),
                ),
                "pr_curve": wandb.plot.pr_curve(
                    y_true=ds_test.targets.cpu().numpy(),
                    y_probas=preds.cpu().numpy(),
                    labels=list(range(cfg.num_classes)),
                ),
                "roc_curve": wandb.plot.roc_curve(
                    y_true=ds_test.targets.cpu().numpy(),
                    y_probas=preds.cpu().numpy(),
                    labels=list(range(cfg.num_classes)),
                ),
            },
            step=step,
        )

    else:
        raise NotImplementedError(f"Experiment {cfg.experiment} not implemented.")
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
        query_idx = string_to_func[cfg.al_method](
            f_samples,
            batch_size=cfg.query_size,
            task="reg" if cfg.experiment == "simulated" else "cls",
        )
    else:
        x_pool = ds_pool[:][0].to(cfg.device)
        query_idx = string_to_func[cfg.al_method](
            x_pool=x_pool,
            model=model,
            batch_size=cfg.query_size,
            task="reg" if cfg.experiment == "simulated" else "cls",
        )

    return list(query_idx)


def simulated_experiment(cfg):
    """run experiments."""

    # set up wandb
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    run = wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        save_code=True,
        dir=cfg.output_dir,
    )
    wandb.config.update(cfg)

    # set up data
    labelled_idx = []  # indices of labelled pool data used for training
    if cfg.experiment == "simulated":
        ds_pool = Simulated(n=cfg.pool_size, seed=cfg.seed)
        ds_test = Simulated(n=cfg.test_size, seed=cfg.seed + 1)
        # acquire seed set randomly
        query_idx = list(np.random.choice(len(ds_pool), cfg.seed_size, replace=False))
    elif cfg.experiment == "simulated_cls":
        if cfg.dataset == "two_bells":
            ds_pool = TwoBells(
                n_train=cfg.pool_size, n_test=cfg.test_size, seed=cfg.seed
            )
            ds_test = TwoBells(
                n_train=cfg.pool_size, n_test=cfg.test_size, seed=cfg.seed, train=False
            )
            cfg.num_classes = 2
        elif cfg.dataset == "simulated":
            ds_pool = SimulatedCls(
                num_classes=cfg.num_classes,
                num_samples_per_class=cfg.pool_size,
                minority_class_ratio=cfg.minority_class_ratio,
                seed=cfg.seed,
            )
            ds_test = SimulatedCls(
                num_classes=cfg.num_classes,
                num_samples_per_class=cfg.test_size,
                minority_class_ratio=1.0,  # no imbalance in test set
                seed=cfg.seed + 1,
            )
        else:
            raise NotImplementedError(f"Dataset {cfg.dataset} not implemented.")
        # ensure at least one sample from each class in seed set
        # classes are in order in the dataset
        query_idx = [
            i for i in range(0, cfg.pool_size * (cfg.num_classes - 1), cfg.pool_size)
        ]
        query_idx += [len(ds_pool) - 1]  # add last sample (minority class)
        # fill up seed set with random samples
        query_idx += list(
            np.random.choice(
                [i for i in range(len(ds_pool)) if i not in query_idx],
                cfg.seed_size - len(query_idx),
                replace=False,
            )
        )
    else:
        raise NotImplementedError(f"Experiment {cfg.experiment} not implemented.")
    labelled_idx += query_idx
    print(
        f"Pool set size: {len(ds_pool)}, Test set size: {len(ds_test)}, Seed set size: {len(labelled_idx)}"
    )

    # set up (prior) model
    model = GP(
        train_x=ds_pool[labelled_idx][0],
        train_y=ds_pool[labelled_idx][1],
        task="reg" if cfg.experiment == "simulated" else "cls",
        num_classes=cfg.num_classes,
        kernel_str=cfg.gp_kernel,
    )
    model.to(cfg.device)
    print(f"Model: {model}")
    mi, test_loss, fig = evaluate_and_plot_acquisition(
        cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=0
    )
    print(f"\nINIT:\nMutual info of batch: {mi}, Test loss: {test_loss}")
    to_log = {
        "query_mi": mi,
        "test_loss": test_loss,
        "num_labelled": len(labelled_idx),
        "acquisition_plot": wandb.Image(fig),
    }
    wandb.log(to_log, step=0)

    # active learning loop. Note: the first step is training on the randomly sampled seed set
    try:
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
                task="reg" if cfg.experiment == "simulated" else "cls",
                num_classes=cfg.num_classes,
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
                if al_step == 1:
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
                        one_hot_labels=True
                        if cfg.experiment == "simulated_cls"
                        else False,
                    )
                    env.update_labelled_dataset(labelled_idx)
                    print(
                        f"GFN: {env}, {trajectories_sampler}, {criterion}, {optimizer}, {scheduler}"
                    )
                if (cfg.gfn_retrain == "full") and (al_step != 1):
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
                        one_hot_labels=True
                        if cfg.experiment == "simulated_cls"
                        else False,
                    )
                    env.update_labelled_dataset(labelled_idx)
                    print(f"reinitialised GFN")
                elif (cfg.gfn_retrain == "last_layer") and (al_step != 1):
                    optimizer = freeze_gfn(
                        cfg,
                        env=env,
                        criterion=criterion,
                        optimizer=optimizer,
                        proxy_reward_function=partial(
                            model.mutual_information,
                            sample=cfg.sample_mi,
                            num_samples=cfg.sample_mi_n,
                        ),
                    )
                    print(f"retraining final layer of GFN")
                else:
                    # update the reward function
                    env.proxy_reward_function = partial(
                        model.mutual_information,
                        sample=cfg.sample_mi,
                        num_samples=cfg.sample_mi_n,
                    )

                # (re)train the GFN using new reward
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
                    step=al_step - 1,  # start from 0
                )
                # save model
                criterion.parametrization.save_state_dict(path=wandb.run.dir)
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
            else:
                # choose next query batch from pool set
                query_idx = baseline_choose_next_query(
                    cfg, model, ds_pool, labelled_idx
                )
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
            wandb.log(to_log, step=al_step)

            # add to labelled set
            labelled_idx += query_idx
            if cfg.al_method == "gfn":
                env.update_labelled_dataset(query_idx)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping training")
        pass

    run.finish()
