# ライブラリのインポート
from solutions.cnn import CNN

import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', help='Number of loop', type=int, default=1)
    parser.add_argument('--app-path', help='Path to unity application', default='Applications/20robots_default')
    parser.add_argument('--log-dir', help='Log directory', default='log')
    parser.add_argument('--load-model', help='Path to model file.', default=None)
    parser.add_argument('--timesteps', help='timesteps per episode', type=int, default=1000)
    parser.add_argument('--max-iter', help='Max training iterations.', type=int, default=1000)
    parser.add_argument('--save-interval', help='Model saving period.', type=int, default=50)
    parser.add_argument('--seed', help='Random seed for evaluation.', type=int, default=42)
    parser.add_argument('--reps', help='Number of rollouts for fitness.', type=int, default=1)
    parser.add_argument('--init-sigma', help='Initial std.', type=float, default=1.0)
    parser.add_argument('--algo-number', help='0: CMA-ES, 1: SNES', type=int, default=1)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    agent = CNN(
        device=torch.device('cpu'),
        file_name=config.app_path,
        act_dim=4,
        feat_dim=128
    )

    agent.train(
        t=config.t,
        timesteps=config.timesteps,
        max_iter=config.max_iter,
        reps=config.reps,
        save_interval=config.save_interval,
        log_dir=config.log_dir,
        seed=config.seed,
        load_model=config.load_model,
        init_sigma=config.init_sigma,
        algo_number=config.algo_number
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
