"""
Main entry point for running experiments with batchgfn.
Argmuents set with argparse.
"""
import sys
import logging
from argparse import ArgumentParser

import torch

import numpy as np

from scripts.al_simulated import simulated_experiment
from scripts.al_meta_simulated import meta_simulated_experiment
from scripts.al_lookahead_simulated import lookahead_simulated_experiment
from scripts.al_mnist import mnist_experiment


def parse_args():
    # TODO: make config yaml files

    parser = ArgumentParser()

    # Experiment arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable_cuda", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="batchgfn")
    parser.add_argument("--wandb_group", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="/home/scratch/output")
    parser.add_argument("--al_method", type=str, default="gfn")
    parser.add_argument(
        "--experiment",
        type=str,
        default="simulated",
        choices=[
            "simulated",
            "simulated_cls",
            "meta_simulated",
            "lookahead_simulated",
            "mnist",
        ],
    )
    # cls arguments
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--minority_class_ratio", type=float, default=1.0)
    parser.add_argument("--num_repetitions", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="simulated")
    parser.add_argument("--autoencoder_latent_dim", type=int, default=32)

    # AL arguments
    parser.add_argument("--query_size", type=int, default=10)
    parser.add_argument("--pool_size", type=int, default=2000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--seed_size", type=int, default=5)
    parser.add_argument("--n_queries", type=int, default=10)
    parser.add_argument("--n_stoch_queries", type=int, default=20)
    parser.add_argument("--stoch_temp", type=float, default=1)
    parser.add_argument("--sample_mi", action="store_true", help="Sample MI estimate")
    parser.add_argument(
        "--sample_mi_n",
        type=int,
        default=5000,
        help="Number of samples for MI estimate",
    )

    # meta-train arguments
    parser.add_argument(
        "--use_true_lookahead",
        action="store_true",
        help="Use true labels for lookahead training (default: use model predictions)",
    )
    parser.add_argument(
        "--only_seed_lookahead",
        action="store_true",
        help="Only use seed data for lookahead training (default: repeat lookahead training)",
    )
    parser.add_argument("--lookahead_method", type=str, default="random")
    parser.add_argument(
        "--meta_train_size",
        type=int,
        default=10,
        help="Number of meta-train tasks (or lookaheads)",
    )
    parser.add_argument("--meta_test_size", type=int, default=10)
    parser.add_argument("--meta_train_min_seed_size", type=int, default=10)
    parser.add_argument("--meta_train_max_seed_size", type=int, default=50)
    parser.add_argument("--meta_train_n_episodes", type=int, default=20)
    parser.add_argument("--meta_test_n_episodes", type=int, default=10)
    parser.add_argument("--meta_eval_interval", type=int, default=5)

    # model arguments
    parser.add_argument("--model", type=str, default="gp")
    parser.add_argument("--num_gp_samples", type=int, default=20)
    parser.add_argument("--num_gp_epochs", type=int, default=1000)
    parser.add_argument("--gp_kernel", type=str, default="matern")
    parser.add_argument("--gp_fixed", action="store_true")
    parser.add_argument("--model_batch_size", type=int, default=16)
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--model_opt", type=str, default="adam")
    parser.add_argument("--model_dropout", type=float, default=0.5)

    # GFN arguments
    parser.add_argument("--gfn_loss", type=str, default="tb")
    parser.add_argument("--gfn_sample_temp", type=float, default=1.0)
    parser.add_argument("--gfn_sample_eps", type=float, default=0.1)
    parser.add_argument("--gfn_reward_temp", type=float, default=0.2)
    parser.add_argument(
        "--gfn_retrain",
        type=str,
        default="full",
        choices=["full", "last_layer", "false"],
    )
    parser.add_argument(
        "--gfn_iters_decay_rate",
        type=float,
        default=1.0,
        help="Decay rate for number of training iterations",
    )
    parser.add_argument("--gfn_lamda", type=float, default=0.9)
    parser.add_argument("--gfn_forward_looking", action="store_true", default=False)
    parser.add_argument("--gfn_n_iterations", type=int, default=2000)
    parser.add_argument("--gfn_validation_interval", type=int, default=100)
    parser.add_argument("--gfn_batch_size", type=int, default=16)
    parser.add_argument("--gfn_module", type=str, default="poolnet")
    parser.add_argument("--gfn_opt", type=str, default="adam")
    parser.add_argument("--gfn_scheduler", type=str, default="constant")
    parser.add_argument("--warmup_prop", type=float, default=0.01)
    parser.add_argument("--gfn_lr", type=float, default=1e-3)
    parser.add_argument("--gfn_hidden_dim", type=int, default=256)
    parser.add_argument("--gfn_n_hidden_layers", type=int, default=2)
    parser.add_argument("--gfn_activation_fn", type=str, default="relu")
    parser.add_argument("--gfn_compute_full_dist", action="store_true")
    parser.add_argument(
        "--resample_for_validation",
        action="store_true",
        default=False,
        help="If False (default), the pmf is obtained from the latest visited terminating states",
    )
    parser.add_argument(
        "--validation_samples",
        type=int,
        default=50000,
        help="Number of validation samples to use to evaluate the pmf.",
    )

    # NPT arguments
    parser.add_argument("--n_transformer_layers", type=int, default=8)
    parser.add_argument("--n_transformer_heads", type=int, default=8)
    parser.add_argument("--label_dim_hid", type=int, default=32)

    cfg = parser.parse_args(sys.argv[1:])

    cfg.device = (
        torch.device("cuda")
        if (not cfg.disable_cuda) and torch.cuda.is_available()
        else torch.device("cpu")
    )

    return cfg


if __name__ == "__main__":
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    # Parse arguments
    cfg = parse_args()
    # set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    # run experiment
    print(f"Running experiment with config: {cfg}")
    if cfg.experiment in ["simulated", "simulated_cls"]:
        simulated_experiment(cfg)
    elif cfg.experiment == "meta_simulated":
        meta_simulated_experiment(cfg)
    elif cfg.experiment == "lookahead_simulated":
        lookahead_simulated_experiment(cfg)
    elif cfg.experiment == "mnist":
        mnist_experiment(cfg)
    else:
        raise NotImplementedError(f"Experiment {cfg.experiment} not implemented")
