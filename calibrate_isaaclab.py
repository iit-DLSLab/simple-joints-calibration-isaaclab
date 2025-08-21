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

import numpy as np
import time

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)

from isaaclab.assets import Articulation

# Import extensions to set up environment tasks
import basic_locomotion_dls_isaaclab.tasks  # noqa: F401

import utility
import config
from isaaclab.managers import SceneEntityCfg


def main():
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


    freezed_base_positions = torch.tensor(
        [0, 0, 0.8], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)
    freezed_base_velocities = torch.tensor(
        [0, 0, 0, 0, 0, 0], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)
    
    # space the base positions over a 20m x 20m square grid
    grid_size = int(torch.ceil(torch.sqrt(torch.tensor(args_cli.num_envs, dtype=torch.float32))))
    spacing = 20.0 / (grid_size - 1) if grid_size > 1 else 0.0
    
    for i in range(args_cli.num_envs):
        row = i // grid_size
        col = i % grid_size
        freezed_base_positions[i, 0] += col * spacing - 10.0  # Center around origin (-10 to +10)
        freezed_base_positions[i, 1] += row * spacing - 10.0  # Center around origin (-10 to +10)
    freezed_base_orientations = torch.tensor(
        [0, 0, 0, 1], dtype=torch.float32, device=env.device
    ).repeat(args_cli.num_envs,1)

    # Sample different Kp and Kd values for each environment
    nominal_kp = config.Kp_walking
    nominal_kd = config.Kd_walking
    # Randomize ±50% around nominal values
    kp_values = nominal_kp * (1.0 + (torch.rand((args_cli.num_envs, 4), device=env.device) - 0.5))  # ±50% of nominal Kp
    kd_values = nominal_kd * (1.0 + (torch.rand((args_cli.num_envs, 4), device=env.device) - 0.5))  # ±50% of nominal Kd

    # Sample different friction static and dynamic values for each environment
    friction_static_values = (torch.rand((args_cli.num_envs, 4), device=env.device)) * 2.0  # Random static friction
    friction_dynamic_values = (torch.rand((args_cli.num_envs, 4), device=env.device)) * 2.0  # Random dynamic friction

    # Apply the Kp and Kd values to the robot's joints
    asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
    asset: Articulation = env.unwrapped.scene[asset_cfg.name]
    for actuator in asset.actuators.values():
        actuator.stiffness[:] = kp_values
        actuator.damping[:] = kd_values

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
            
            print(f"Running timestep: {timestep}")
            
            if(timestep >= all_dataset_actual_joint_pos.shape[0] - 1):
                print("End of dataset reached, resetting to the beginning.")
                break
            
            joint_pos = torch.tensor(
                all_dataset_actual_joint_pos[timestep], dtype=torch.float32, device=env.device
            )

            if(joint_pos == torch.tensor([-10.0] * joint_pos.shape[0], device=env.device)).all():
                print("End of motion reached, reset initial robot configuration.")
                # reset the environment
                env.unwrapped._robot.write_root_pose_to_sim(
                    torch.cat([freezed_base_positions, freezed_base_orientations], dim=-1), env_ids=env_ids
                )
                env.unwrapped._robot.write_root_velocity_to_sim(
                    freezed_base_velocities, env_ids=env_ids
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
                    torch.cat([freezed_base_positions, freezed_base_orientations], dim=-1), env_ids=env_ids
                )

                env.unwrapped._robot.write_root_velocity_to_sim(
                    freezed_base_velocities, env_ids=env_ids
                )
                
                """env.unwrapped._robot.write_joint_state_to_sim(
                    joint_pos, joint_vel, env_ids=env_ids
                )"""
                
                # control the robot with the joint positions and 
                actions = desired_joint_pos
                obs, _, _, _ = env.step(actions)
                
                # Compute error between desired and actual joint positions
                error_joint_pos[env_ids] += torch.abs(joint_pos - env.unwrapped._robot.data.joint_pos)
                error_joint_vel[env_ids] += torch.abs(joint_vel - env.unwrapped._robot.data.joint_vel)
                



            timestep += 1

            # env stepping
            env.env.render()

            time.sleep(1.0 / dataset_fps)

    
    # Print the average errors
    avg_error = error_joint_pos.mean(dim=1) + error_joint_vel.mean(dim=1)
    print("Average Joint Error:", avg_error)

    # take the best Kp and Kd values
    best_kp = kp_values[avg_error.argmin()]
    best_kd = kd_values[avg_error.argmin()]

    print(f"Best Kp: ", best_kp, "Best Kd: ", best_kd)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
