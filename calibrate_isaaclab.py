"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import os
import torch

from amp_rsl_rl.runners import AMPOnPolicyRunner
import numpy as np
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab.utils.dict import print_dict
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.assets import Articulation

# Import extensions to set up environment tasks
import quadruped_rl_collection.tasks  # noqa: F401

import utility

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )



    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)


    # reset environment
    timestep = 0
    env_ids = torch.arange(args_cli.num_envs, device=env.unwrapped.device)  # Updated to include all env IDs


    datasets = utility.load_datasets()
    all_dataset_actual_joint_pos = datasets["all_dataset_actual_joint_pos"]
    all_dataset_actual_joint_vel = datasets["all_dataset_actual_joint_vel"]
    all_dataset_desired_joint_pos = datasets["all_dataset_desired_joint_pos"]
    all_dataset_desired_joint_vel = datasets["all_dataset_desired_joint_vel"]
    dataset_fps = datasets["dataset_fps"]

    base_positions = torch.tensor(
        [0, 0, 0.4], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)
    # space the base positions with a fixed offset for each environment
    base_positions[:, 0] += torch.arange(
        args_cli.num_envs, dtype=torch.float32, device=env.device
    ) * 0.5  # Adjust the offset as needed
    base_positions[:, 1] += torch.arange(
        args_cli.num_envs, dtype=torch.float32, device=env.device
    ) * 0.5  # Adjust the offset as needed
    base_orientations = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)

    # Sample different Kp and Kd values for each environment
    kp_values = 20. + (torch.rand(args_cli.num_envs, device=env.device) - 0.5) * 10.0  # Random Kp
    kd_values = 2.5 + (torch.rand(args_cli.num_envs, device=env.device) - 0.5) * 4.  # Random Kd

    # Apply the Kp and Kd values to the robot's joints
    asset_cfg.name = "bo?"
    asset: Articulation = env.scene[asset_cfg.name]
    for actuator in asset.actuators.values():
        actuator.stiffness[env_ids] = kp_values
        actuator.damping[env_ids] = kd_values

    error_joint_pos = torch.zeros(
        (args_cli.num_envs, len(env.unwrapped._robot.joint_names)), dtype=torch.float32, device=env.device 
    )
    error_joint_vel = torch.zeros(
        (args_cli.num_envs, len(env.unwrapped._robot.joint_names)), dtype=torch.float32, device=env.device
    )

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():

            if( timestep >= all_dataset_actual_joint_pos.shape[0] - 1):
                print("End of dataset reached, resetting to the beginning.")
                break
            
            joint_pos = torch.tensor(
                all_dataset_actual_joint_pos[timestep], dtype=torch.float32, device=env.device
            )

            if(joint_pos == torch.tensor([-10.0] * joint_pos.shape[0], device=env.device)).all():
                print("End of motion reached, reset initial robot configuration.")
                # reset the environment
                env.unwrapped._robot.write_root_pose_to_sim(
                    torch.cat([base_positions, base_orientations], dim=-1), env_ids=env_ids
                )

                joint_pos = torch.tensor(
                    all_dataset_actual_joint_pos[timestep+1], dtype=torch.float32, device=env.device
                )
                env.unwrapped._robot.write_joint_state_to_sim(
                    joint_pos, joint_pos*0.0, env_ids=env_ids
                )

                env.env.render()
                time.sleep(2.0)
            
            else:
                
                joint_pos = torch.tensor(
                    all_dataset_actual_joint_pos[timestep+1], dtype=torch.float32, device=env.device
                )

                joint_vel = torch.tensor(
                    all_dataset_actual_joint_vel[timestep], dtype=torch.float32, device=env.device
                )

                desired_joint_pos = torch.tensor(
                    all_dataset_desired_joint_pos[timestep], dtype=torch.float32, device=env.device
                )

                env.unwrapped._robot.write_root_pose_to_sim(
                    torch.cat([base_positions, base_orientations], dim=-1), env_ids=env_ids
                )
                
                env.unwrapped._robot.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=env_ids
                )
                
                # control the robot with the joint positions and 
                actions = desired_joint_pos
                obs, _, _, _ = env.step(actions)
                
                # Compute error between desired and actual joint positions
                error_joint_pos[env_ids] += torch.abs(joint_pos - obs["joint_positions"])
                error_joint_vel[env_ids] += torch.abs(joint_vel - obs["joint_velocities"])
                # Print the errors for debugging



            timestep += 1

            # env stepping
            env.env.render()

            time.sleep(1.0 / dataset_fps)

    # Print the average errors
    avg_error = error_joint_pos.mean(dim=0) + error_joint_vel.mean(dim=0)
    print("Average Joint Position Error:", avg_error.cpu().numpy())
    print("Average Joint Velocity Error:", avg_error.cpu().numpy())

    # take the best Kp and Kd values
    best_kp = kp_values[avg_error.argmin()]
    best_kd = kd_values[avg_error.argmin()]

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
