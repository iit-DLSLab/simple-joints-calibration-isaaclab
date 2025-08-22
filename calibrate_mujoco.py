# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv

import utility
import config

import numpy as np
import copy
import time
import mujoco

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    robot_name = config.robot
    simulation_dt = 0.002


    # Create the quadruped robot environment -----------------------------------------------------------
    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=simulation_dt,
        base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
    )


    env.reset(random=False)
    env.render()  # Pass in the first render call any mujoco.viewer.KeyCallbackType

    freezed_base_position = np.array([0, 0, 0.4])
    freezed_base_orientation = np.array([1, 0, 0, 0])
    freezed_base_linear_velocity = np.array([0, 0, 0.])
    freezed_base_angular_velocity = np.array([0, 0, 0])
    

    # Load datasets for calibration
    expected_joint_order = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    datasets_path = config.datasets_path
    datasets = utility.load_datasets(datasets_path, expected_joint_order)
    all_dataset_actual_joint_pos = datasets["all_dataset_actual_joint_pos"]
    all_dataset_actual_joint_vel = datasets["all_dataset_actual_joint_vel"]
    all_dataset_desired_joint_pos = datasets["all_dataset_desired_joint_pos"]
    all_dataset_desired_joint_vel = datasets["all_dataset_desired_joint_vel"]
    dataset_fps = datasets["dataset_fps"]

    timestep = 0

    while True:
        
        print(f"Running timestep: {timestep}")
        
        if(timestep >= all_dataset_actual_joint_pos.shape[0] - 1):
            print("End of dataset reached, resetting to the beginning.")
            break
        
        joint_pos = all_dataset_actual_joint_pos[timestep]
        
        env.mjData.qpos[0:3] = copy.deepcopy(freezed_base_position)
        env.mjData.qpos[3:7] = copy.deepcopy(freezed_base_orientation)
        env.mjData.qvel[0:3] = copy.deepcopy(freezed_base_linear_velocity)
        env.mjData.qvel[3:6] = copy.deepcopy(freezed_base_angular_velocity)
        env.mjModel.opt.timestep = simulation_dt

        if((joint_pos == np.array([-10.0]*joint_pos.shape[0])).all()):
            print("End of motion reached, reset initial robot configuration.")
            # reset the environment
            
            
            joint_pos = all_dataset_actual_joint_pos[timestep+1]
            env.mjData.qpos[7:] = copy.deepcopy(joint_pos)
            env.mjData.qvel[6:] = copy.deepcopy(joint_pos*0.0)

            """mujoco.mj_forward(env.mjModel, env.mjData) 
            env.render()
            breakpoint()
            time.sleep(2.0)"""
        
        else:
            
            joint_pos = all_dataset_actual_joint_pos[timestep+1]
            joint_vel = all_dataset_actual_joint_vel[timestep+1]

            """desired_joint_pos = torch.tensor(
                all_dataset_desired_joint_pos[timestep], dtype=torch.float32, device=env.device
            )"""

            if((joint_pos == np.array([-10.0]*joint_pos.shape[0])).all()):
                print("##################")
            else:
                env.mjData.qpos[7:] = copy.deepcopy(joint_pos)
                env.mjData.qvel[6:] = copy.deepcopy(joint_pos*0.0)
            


        mujoco.mj_forward(env.mjModel, env.mjData) 
        timestep += 1

        env.render()
        #time.sleep(1.0 / dataset_fps)
        time.sleep(0.2)