"""
Active learning experiments on simulated data. 
Acquire batches of points from a pool set, train a model on the acquired points,
and evaluate the model on a test set.
"""

from functools import partial
from pathlib import Path
from collections import Counter
from tqdm.autonotebook import tqdm
import wandb

import torch
import torch.nn as nn

import numpy as np

from batchbald_redux import active_learning


from batchgfn.datasets import create_modified_MNIST_dataset, get_data, get_targets
from batchgfn.acquisitions import (
    batch_bald_cov,
    bald,
    stochastic_bald,
    random_multi,
)
from batchgfn.models import ConvolutionalNeuralNetwork
from batchgfn.gflow import get_gfn, train_gfn, freeze_gfn, gfn_choose_next_query


def evaluate_acquisition(cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step):
    ds_query = torch.utils.data.Subset(ds_pool, query_idx)
    query_points, _ = get_data(ds_query)
    mi = (
        model.mutual_information(
            query_points.to(cfg.device), num_samples=cfg.sample_mi_n
        )
        .squeeze()
        .cpu()
        .float()
    )

    print(f"evaluating on test set...")
    ds_test_dataloader = torch.utils.data.DataLoader(
        ds_test, batch_size=cfg.model_batch_size, shuffle=False
    )
    test_loss, test_acc, prec, rec, f1, sup, test_pred, test_true = model.evaluate(
        ds_test_dataloader,
        device=cfg.device,
        num_inference_samples=cfg.sample_mi_n,
        task="test",
    )

    print(
        f"Test: loss={test_loss}, acc={test_acc}, prec={prec}, rec={rec}, f1={f1}, sup={sup}"
    )
    # log per class metrics
    label_counts = Counter(get_targets(ds_pool)[labelled_idx].tolist())
    label_counts = np.array([[k, v] for k, v in label_counts.items()])
    # accuracy on the minority class (0)
    minority_class_prec = prec[0]
    minority_class_rec = rec[0]
    minority_class_f1 = f1[0]
    test_pred_array = np.array(test_pred)
    test_true_array = np.array(test_true)
    true_positives = np.sum(test_true_array[test_pred_array == 0] == 0)
    minority_class_acc = true_positives / np.sum(test_true_array == 0)

    wandb.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_true,
                preds=test_pred,
            ),
            # prec, rec, f1, sup table
            "test_per_class_metrics": wandb.Table(
                data=np.array([prec, rec, f1, sup]).T,
                columns=["precision", "recall", "f1", "support"],
            ),
            "training_class_counts": wandb.Table(
                data=label_counts,
                columns=["label", "count"],
            ),
            "minority_class_prec": minority_class_prec,
            "minority_class_rec": minority_class_rec,
            "minority_class_f1": minority_class_f1,
            "minority_class_acc": minority_class_acc,
        },
        step=step,
    )

    return mi, test_loss, test_acc


def baseline_choose_next_query(cfg, model, ds_pool, labelled_idx):
    """choose next query batch from pool set"""
    string_to_func = {
        "bald": bald,
        "batchbald": batch_bald_cov,
        "stoch_bald": partial(
            stochastic_bald, reps=cfg.n_stoch_queries, temp=cfg.stoch_temp
        ),
        "random": partial(random_multi, labelled_idx=labelled_idx, reps=1),
        "stoch_random": partial(
            random_multi, labelled_idx=labelled_idx, reps=cfg.n_stoch_queries
        ),
    }
    unlabelled_idx = np.array([i for i in range(len(ds_pool)) if i not in labelled_idx])
    ds_unlabelled = torch.utils.data.Subset(
        ds_pool, unlabelled_idx
    )  # to avoid picking already labelled samples
    ds_pool_dataloader = torch.utils.data.DataLoader(
        ds_unlabelled, batch_size=64, shuffle=False
    )
    N = len(ds_pool_dataloader.dataset)
    K = cfg.sample_mi_n
    C = cfg.num_classes
    logits_N_K_C = torch.empty(
        (len(ds_pool_dataloader.dataset), K, C), dtype=torch.double
    )
    print(f"Computing {cfg.al_method} scores for {N} samples in pool set...")
    with torch.no_grad():
        model.eval()
        for i, (data, _) in enumerate(
            tqdm(ds_pool_dataloader, desc="Evaluating Acquisition Set", leave=False)
        ):
            data = data.to(device=cfg.device)
            lower = i * ds_pool_dataloader.batch_size
            upper = min(lower + ds_pool_dataloader.batch_size, N)
            logits_N_K_C[lower:upper].copy_(model(data, K).double(), non_blocking=True)

        # choose query batch
        # expects (K,N,C) logits
        logits_K_N_C = logits_N_K_C.permute(1, 0, 2)

        query_idx = string_to_func[cfg.al_method](
            logits_K_N_C,
            batch_size=cfg.query_size,
            task="cls",
        )

    # get the indices of the query batch in the pool set
    return list(unlabelled_idx[query_idx])


def init_and_train_cnn(cfg, ds_pool, labelled_idx):
    model = ConvolutionalNeuralNetwork(
        input_shape=(1, 28, 28), output_size=3, dropout_rate=cfg.model_dropout
    )
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model_lr)
    criterion = nn.CrossEntropyLoss()
    ds_train = torch.utils.data.Subset(ds_pool, labelled_idx)
    # TODO need a validation set too to stop overfitting
    ds_train_dataloader = torch.utils.data.DataLoader(
        ds_train, batch_size=cfg.model_batch_size, shuffle=True
    )
    for epoch in range(cfg.num_gp_epochs):
        loss, acc = model.evaluate(
            ds_train_dataloader,
            optimizer,
            scheduler=None,
            criterion=criterion,
            device=cfg.device,
            num_inference_samples=cfg.sample_mi_n,
            task="train",
        )
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={loss}, acc={acc}")

    return model


def mnist_experiment(cfg):
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
    # set required cfg for mnist (override)
    cfg.sample_mi = True

    wandb.config.update(cfg)

    # set up data
    labelled_idx = []  # indices of labelled pool data used for training
    ds_pool, ds_test = create_modified_MNIST_dataset(
        num_classes=cfg.num_classes,
        num_samples_per_class=cfg.pool_size,
        minority_class_ratio=cfg.minority_class_ratio,
        num_repetitions=cfg.num_repetitions,
        add_noise=True,
    )
    targets_pool = get_targets(ds_pool)
    query_idx = active_learning.get_balanced_sample_indices(
        targets_pool, cfg.num_classes, cfg.seed_size // cfg.num_classes
    )
    labelled_idx += query_idx
    print(
        f"Pool set size: {len(ds_pool)}, Test set size: {len(ds_test)}, Seed set size: {len(labelled_idx)}"
    )
    # set up and train model on seed set
    model = init_and_train_cnn(cfg, ds_pool, labelled_idx)

    print(f"Model: {model}")
    mi, test_loss, test_acc = evaluate_acquisition(
        cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=0
    )
    print(
        f"\nINIT:\nMutual info of batch: {mi}, Test loss: {test_loss}, Test acc: {test_acc}"
    )
    to_log = {
        "query_mi": mi,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_labelled": len(labelled_idx),
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
            model = init_and_train_cnn(cfg, ds_pool, labelled_idx)

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
                        one_hot_labels=True,
                        autoencode=True,
                        autoencoder_latent_dim=cfg.autoencoder_latent_dim,
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
                        one_hot_labels=True,
                        autoencode=True,
                        autoencoder_latent_dim=cfg.autoencoder_latent_dim,
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
            mi, test_loss, test_acc = evaluate_acquisition(
                cfg, model, ds_pool, ds_test, labelled_idx, query_idx, step=al_step
            )
            print(
                f"\nSTEP {al_step} SUMMARY:\nMutual info of batch: {mi}, Test loss: {test_loss}, Test accuracy: {test_acc}\n"
            )
            to_log.update(
                {
                    "query_mi": mi,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "num_labelled": len(labelled_idx),
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
