import argparse
import os
import pickle
import shutil
from datetime import datetime

from genesis_drone_env.envs.hover_env import HoverEnv
from rsl_rl.runners import OnPolicyRunner
from genesis_drone_env.utils.config import process_config

import genesis as gs


def main():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    envkey = {"drone-hovering": "env_config"}
    trainkey = {"drone-hovering": "train_config"}

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="drone-hovering")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=300)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_dir = os.path.join(current_dir,
                           f"logs/{args.exp_name}_{timestamp}")
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = process_config(
        args, current_dir, envkey, trainkey
    )

    if args.vis:
        env_cfg["visualize_target"] = True

    env = HoverEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
