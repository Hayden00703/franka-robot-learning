"""Evaluate a Franka lift checkpoint with a sustained-lift criterion."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_ROOT", PROJECT_ROOT / "third_party" / "IsaacLab")).expanduser().resolve()
RSL_RL_SCRIPT_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(RSL_RL_SCRIPT_DIR))

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.seed import configure_seed  # noqa: E402
from isaaclab.utils.string import list_intersection, string_to_callable  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, setup_preset_cli  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import cli_args  # noqa: E402
from metrics import update_hold_streak  # noqa: E402


with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--minimum_lift", type=float, default=0.05)
parser.add_argument("--hold_seconds", type=float, default=1.0)
parser.add_argument("--required_success_rate", type=float, default=0.8)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--external_callback", default=None)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

remaining_args_env_registration = None
if args_cli.external_callback:
    callback = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = callback()
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run one randomized episode in each parallel environment."""
    if args_cli.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")
    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    with launch_simulation(env_cfg, args_cli):
        installed_version = metadata.version("rsl-rl-lib")
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.scene.num_envs = args_cli.episodes
        env_cfg.seed = args_cli.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = str(checkpoint.parent)

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        configure_seed(env_cfg.seed, True)
        runner.load(str(checkpoint))
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        obs = env.get_observations()
        object_asset = env.unwrapped.scene["object"]
        initial_height = object_asset.data.root_pos_w.torch[:, 2].clone()
        max_lift = torch.zeros(args_cli.episodes, device=env.unwrapped.device)
        streak = torch.zeros(args_cli.episodes, dtype=torch.long, device=env.unwrapped.device)
        succeeded = torch.zeros(args_cli.episodes, dtype=torch.bool, device=env.unwrapped.device)
        completed = torch.zeros(args_cli.episodes, dtype=torch.bool, device=env.unwrapped.device)
        required_steps = max(1, round(args_cli.hold_seconds / env.unwrapped.step_dt))

        with torch.inference_mode():
            while not bool(torch.all(completed)):
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                object_height = object_asset.data.root_pos_w.torch[:, 2]
                active = ~completed
                max_lift = torch.where(active, torch.maximum(max_lift, object_height - initial_height), max_lift)
                next_streak, next_succeeded = update_hold_streak(
                    object_height,
                    initial_height,
                    streak,
                    succeeded,
                    minimum_lift=args_cli.minimum_lift,
                    required_steps=required_steps,
                )
                streak = torch.where(active, next_streak, streak)
                succeeded = torch.where(active, next_succeeded, succeeded)
                completed |= dones.bool()
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)

        success_count = int(succeeded.sum().item())
        result = {
            "checkpoint": str(checkpoint),
            "episodes": args_cli.episodes,
            "seed": env_cfg.seed,
            "minimum_lift_m": args_cli.minimum_lift,
            "hold_seconds": args_cli.hold_seconds,
            "required_consecutive_steps": required_steps,
            "successes": success_count,
            "success_rate": success_count / args_cli.episodes,
            "mean_max_lift_m": float(max_lift.mean().item()),
            "minimum_max_lift_m": float(max_lift.min().item()),
            "required_success_rate": args_cli.required_success_rate,
        }
        output = args_cli.output or checkpoint.parent / "evaluation_100.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        env.close()

        if result["success_rate"] < args_cli.required_success_rate:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
