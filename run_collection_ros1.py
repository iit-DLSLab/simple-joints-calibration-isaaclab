# Description: This script is used to run the policy on the real robot

# Authors:
# Giulio Turrisi

import rospy
from dls_msgs.msg import rl_signal_in, rl_signal_out

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

import threading
import copy
import os 

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr
import config


# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py

USE_MUJOCO_RENDER = False
USE_MUJOCO_SIMULATION = False


CONTROL_FREQ = 50


class Basic_Locomotion_DLS_Isaaclab_Node():
    def __init__(self):

        # Mujoco env
        robot_name = config.robot
        simulation_dt = 0.002


        # Create the quadruped robot environment -----------------------------------------------------------
        self.env = QuadrupedEnv(
            robot=robot_name,
            scene="flat",
            sim_dt=simulation_dt,
            base_vel_command_type="human",  # "forward", "random", "forward+rotate", "human"
        )
        self.env.reset(random=False)

        self.last_render_time = time.time()
        if USE_MUJOCO_RENDER:
            self.env.render()

        self.stand_up_and_down_actions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        keyframe_id = mujoco.mj_name2id(self.env.mjModel, mujoco.mjtObj.mjOBJ_KEY, "down")
        goDown_qpos = self.env.mjModel.key_qpos[keyframe_id]
        self.stand_up_and_down_actions.FL = goDown_qpos[7:10]
        self.stand_up_and_down_actions.FR = goDown_qpos[10:13]
        self.stand_up_and_down_actions.RL = goDown_qpos[13:16]
        self.stand_up_and_down_actions.RR = goDown_qpos[16:29]
        self.Kp_stand_up_and_down = config.Kp_walking
        self.Kd_stand_up_and_down = config.Kd_walking

        self.calibration_reference_joint_positions = None
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None
        self.num_traj_saved = 0

        # Interactive Command Line ----------------------------
        from console import Console
        self.console = Console(controller_node=self)
        thread_console = threading.Thread(target=self.console.interactive_command_line)
        thread_console.daemon = True
        thread_console.start()

        # Subscribers and Publishers
        self.subscription_state = rospy.Subscriber("/rl/rl_signal_in", rl_signal_in, self.get_state_callback, tcp_nodelay=True, queue_size=1)
        self.publisher = rospy.Publisher('/rl/rl_signal_out', rl_signal_out, queue_size=1)
        self.timer = rospy.Timer(rospy.Duration(1.0/CONTROL_FREQ), self.compute_control)

        # Safety check to not do anything until a first base and blind state are received
        self.first_message_base_arrived = False
        self.first_message_joints_arrived = False 

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None

        # Base State
        self.position = np.zeros(3)
        self.orientation = np.zeros(4)
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

        # Blind State
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
    

    def get_state_callback(self, msg):
        
        self.position = np.array(msg.position) #world frame
        # For the quaternion, the order is [w, x, y, z] on mujoco, and [x, y, z, w] on DLS1
        self.orientation = np.roll(np.array(msg.orientation_quat), 1) #world frame
        self.linear_velocity = np.array(msg.linear_velocity) #world frame
        self.angular_velocity = np.array(msg.angular_velocity) #base frame

        self.joint_positions = np.array(msg.joint_positions)
        self.joint_velocities = np.array(msg.joint_velocities)

        # Fix convention DLS
        if(config.robot == "aliengo"):
            self.joint_positions[0] = -self.joint_positions[0]
            self.joint_positions[6] = -self.joint_positions[6]
            self.joint_velocities[0] = -self.joint_velocities[0]
            self.joint_velocities[6] = -self.joint_velocities[6]

        self.first_message_base_arrived = True
        self.first_message_joints_arrived = True


    def _initialize_calibration_setpoint(self):
        """Initialize calibration setpoint with random values"""
        print("Generating first a setpoint..")
        hip_setpoint = np.random.uniform(0.6, 1.8)
        thigh_setpoint = np.random.uniform(-3.5, 0.5)
        calf_setpoint = np.random.uniform(0., 1.5)
        hip_setpoint = 0.6
        self.calibration_reference_joint_positions = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        self.calibration_reference_joint_positions.FL = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions.FR = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions.RL = np.array([0.0+hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.calibration_reference_joint_positions.RR = np.array([0.0-hip_setpoint, 1.21+thigh_setpoint, -2.794+calf_setpoint])
        self.start_collection_time = time.time()
        
        if self.console.falling_collection:
            # This is used as a global variable to active the free falling phase one time per setpoint
            self.free_falling_phase = False
            self.final_falling_time = 0.35
            #if(hip_setpoint > 1.5):
            #    self.final_falling_time = 0.45
            self.final_thrust_time = 0.2


    def _get_desired_positions_and_gains(self, env):
        """Get desired joint positions and control gains based on collection type"""
        if self.console.setpoint_collection:
            # Setpoint collection: maintain target position
            desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
            desired_joint_pos.FL = copy.deepcopy(self.calibration_reference_joint_positions.FL)
            desired_joint_pos.FR = copy.deepcopy(self.calibration_reference_joint_positions.FR)
            desired_joint_pos.RL = copy.deepcopy(self.calibration_reference_joint_positions.RL)
            desired_joint_pos.RR = copy.deepcopy(self.calibration_reference_joint_positions.RR)
            Kp = self.Kp_stand_up_and_down
            Kd = self.Kd_stand_up_and_down
            
        elif self.console.falling_collection:
            temp = time.time() - self.start_collection_time
            if(temp < self.final_thrust_time or temp > self.final_falling_time):
                # Reach the target
                Kp = self.Kp_stand_up_and_down
                Kd = self.Kd_stand_up_and_down
                desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
                desired_joint_pos.FL = copy.deepcopy(self.calibration_reference_joint_positions.FL)
                desired_joint_pos.FR = copy.deepcopy(self.calibration_reference_joint_positions.FR)
                desired_joint_pos.RL = copy.deepcopy(self.calibration_reference_joint_positions.RL)
                desired_joint_pos.RR = copy.deepcopy(self.calibration_reference_joint_positions.RR)
            
            else:
                # Free falling!!
                Kp = 0.0
                Kd = 0.0
                desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
                desired_joint_pos.FL = np.zeros((int(env.mjModel.nu/4),)) + 20.
                desired_joint_pos.FR = np.zeros((int(env.mjModel.nu/4),)) + 20.
                desired_joint_pos.RL = np.zeros((int(env.mjModel.nu/4),)) + 20.
                desired_joint_pos.RR = np.zeros((int(env.mjModel.nu/4),)) + 20.
                if(temp > self.final_thrust_time + 0.02 and temp < self.final_falling_time - 0.02):
                    self.free_falling_phase = True
                else:
                    self.free_falling_phase = False

        return desired_joint_pos, Kp, Kd


    def _collect_trajectory_data(self, joints_pos, joints_vel, desired_joint_pos):
        """Collect trajectory data by concatenating and storing joint information"""
        concatenated_actual_joints_position = np.concatenate([joints_pos.FL, joints_pos.FR,
                                                            joints_pos.RL, joints_pos.RR])
        concatenated_actual_joints_velocity = np.concatenate([joints_vel.FL, joints_vel.FR,
                                                            joints_vel.RL, joints_vel.RR])
        concatenated_desired_joints_position = np.concatenate([desired_joint_pos.FL, desired_joint_pos.FR,
                                                            desired_joint_pos.RL, desired_joint_pos.RR])
        concatenated_desired_joints_velocity = np.concatenate([joints_vel.FL*0.0, joints_vel.FR*0.0,
                                                            joints_vel.RL*0.0, joints_vel.RR*0.0])

        if(self.console.falling_collection and not self.free_falling_phase):
            # Do not collect data during the anticipation phase
            return
        

        if self.saved_actual_joints_position is None:
            self.saved_actual_joints_position = concatenated_actual_joints_position
            self.saved_actual_joints_velocity = concatenated_actual_joints_velocity
            self.saved_desired_joints_position = concatenated_desired_joints_position
            self.saved_desired_joints_velocity = concatenated_desired_joints_velocity
        else:
            self.saved_actual_joints_position = np.vstack([self.saved_actual_joints_position, concatenated_actual_joints_position])
            self.saved_actual_joints_velocity = np.vstack([self.saved_actual_joints_velocity, concatenated_actual_joints_velocity])
            self.saved_desired_joints_position = np.vstack([self.saved_desired_joints_position, concatenated_desired_joints_position])
            self.saved_desired_joints_velocity = np.vstack([self.saved_desired_joints_velocity, concatenated_desired_joints_velocity])


    def _check_collection_complete(self, joints_pos, desired_joint_pos):
        """Check if data collection is complete based on collection type"""
        if self.console.setpoint_collection:
            # Complete when target is reached or timeout
            target_reached = (np.linalg.norm(desired_joint_pos.FL - joints_pos.FL) < 0.1 and
                            np.linalg.norm(desired_joint_pos.FR - joints_pos.FR) < 0.1 and
                            np.linalg.norm(desired_joint_pos.RL - joints_pos.RL) < 0.1 and
                            np.linalg.norm(desired_joint_pos.RR - joints_pos.RR) < 0.1)
            timeout = time.time() - self.start_collection_time > 2.0
            return target_reached or timeout
            
        elif self.console.falling_collection:
            # Complete after falling phase timeout
            return time.time() - self.start_collection_time > self.final_falling_time


    def _save_trajectory_data(self):
        """Save collected trajectory data to file and reset buffers"""
        self.calibration_reference_joint_positions = None

        # Saving to file trajectory
        desired_fps = CONTROL_FREQ
        data = {
            "joints_list": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", 
                                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", 
                                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
            "actual_joints_position": self.saved_actual_joints_position,
            "actual_joints_velocity": self.saved_actual_joints_velocity,
            "desired_joints_position": self.saved_desired_joints_position,
            "desired_joints_velocity": self.saved_desired_joints_velocity,
            "fps": desired_fps,
        }
        # Save the data to an .npy file
        output_file = f"datasets/traj_{self.num_traj_saved}.npy"
        np.save(output_file, data)

        self.num_traj_saved += 1
        self.saved_actual_joints_position = None
        self.saved_actual_joints_velocity = None
        self.saved_desired_joints_position = None
        self.saved_desired_joints_velocity = None

        input("Press enter to continue.")


    def compute_control(self, event):
        # Update the loop time
        start_time = time.perf_counter()
        if(self.last_start_time is not None):
            self.loop_time = (start_time - self.last_start_time)
        self.last_start_time = start_time
        simulation_dt = self.loop_time

        # Safety check to not do anything until a first base and blind state are received
        if(not USE_MUJOCO_SIMULATION and self.first_message_base_arrived==False and self.first_message_joints_arrived==False):
            return


        # Update the mujoco model
        if(not USE_MUJOCO_SIMULATION):
            self.env.mjData.qpos[0:3] = copy.deepcopy(self.position)
            self.env.mjData.qpos[3:7] = copy.deepcopy(self.orientation)
            self.env.mjData.qvel[0:3] = copy.deepcopy(self.linear_velocity)
            self.env.mjData.qvel[3:6] = copy.deepcopy(self.angular_velocity)
            self.env.mjData.qpos[7:] = copy.deepcopy(self.joint_positions)
            self.env.mjData.qvel[6:] = copy.deepcopy(self.joint_velocities)
            self.env.mjModel.opt.timestep = simulation_dt
            mujoco.mj_forward(self.env.mjModel, self.env.mjData)  
        else:
            self.env.mjData.qpos[0:3] = np.array([0, 0, 0.4])
            self.env.mjData.qpos[3:7] = np.array([1, 0, 0, 0])
            self.env.mjData.qvel[0:3] = np.array([0, 0, 0.])
            self.env.mjData.qvel[3:6] = np.array([0, 0, 0.0])


        env = self.env
        
        qpos, qvel = env.mjData.qpos, env.mjData.qvel


        joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[env.legs_qpos_idx.RR]
    
        joints_vel = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[env.legs_qvel_idx.RR]
    

        if(not self.console.isActivated):
            desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
            desired_joint_pos.FL = self.stand_up_and_down_actions.FL
            desired_joint_pos.FR = self.stand_up_and_down_actions.FR
            desired_joint_pos.RL = self.stand_up_and_down_actions.RL
            desired_joint_pos.RR = self.stand_up_and_down_actions.RR

            # Impedence Loop
            Kp = self.Kp_stand_up_and_down
            Kd = self.Kd_stand_up_and_down
            

        elif(self.console.isActivated and (self.console.setpoint_collection or self.console.falling_collection)):
            
            # Initialize setpoint if needed
            if self.calibration_reference_joint_positions is None:
                self._initialize_calibration_setpoint()

            # Get desired joint positions and control gains based on collection type
            desired_joint_pos, Kp, Kd = self._get_desired_positions_and_gains(env)
            
            # Collect data
            self._collect_trajectory_data(joints_pos, joints_vel, desired_joint_pos)
            
            # Check if collection is complete
            collection_complete = self._check_collection_complete(joints_pos, desired_joint_pos)            
            if collection_complete:
    
                desired_joint_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
                desired_joint_pos.FL = copy.deepcopy(joints_pos.FL)
                desired_joint_pos.FR = copy.deepcopy(joints_pos.FR)
                desired_joint_pos.RL = copy.deepcopy(joints_pos.RL)
                desired_joint_pos.RR = copy.deepcopy(joints_pos.RR)
                Kp = self.Kp_stand_up_and_down
                Kd = self.Kd_stand_up_and_down

                # Hack for now
                rl_signal_out_msg = rl_signal_out()
                rl_signal_out_msg.desired_joint_positions = np.concatenate([desired_joint_pos.FL, desired_joint_pos.FR, desired_joint_pos.RL, desired_joint_pos.RR], axis=0).flatten()
                rl_signal_out_msg.kp = np.ones(12)*Kp
                rl_signal_out_msg.kd = np.ones(12)*Kd
                self.publisher.publish(rl_signal_out_msg)
                
                self._save_trajectory_data()


        
        if USE_MUJOCO_SIMULATION:
            for j in range(10): #Hardcoded for now, if RL is 50Hz, this runs the simulation at 500Hz
                joints_pos.FL = qpos[env.legs_qpos_idx.FL]
                joints_pos.FR = qpos[env.legs_qpos_idx.FR]
                joints_pos.RL = qpos[env.legs_qpos_idx.RL]
                joints_pos.RR = qpos[env.legs_qpos_idx.RR]
            
                joints_vel.FL = qvel[env.legs_qvel_idx.FL]
                joints_vel.FR = qvel[env.legs_qvel_idx.FR]
                joints_vel.RL = qvel[env.legs_qvel_idx.RL]
                joints_vel.RR = qvel[env.legs_qvel_idx.RR]

                error_joints_pos = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
                error_joints_pos.FL = desired_joint_pos.FL - joints_pos.FL
                error_joints_pos.FR = desired_joint_pos.FR - joints_pos.FR
                error_joints_pos.RL = desired_joint_pos.RL - joints_pos.RL
                error_joints_pos.RR = desired_joint_pos.RR - joints_pos.RR
                
                tau = LegsAttr(*[np.zeros((1, int(env.mjModel.nu/4))) for _ in range(4)])
                tau.FL = Kp * (error_joints_pos.FL) - Kd * joints_vel.FL
                tau.FR = Kp * (error_joints_pos.FR) - Kd * joints_vel.FR
                tau.RL = Kp * (error_joints_pos.RL) - Kd * joints_vel.RL
                tau.RR = Kp * (error_joints_pos.RR) - Kd * joints_vel.RR


                action = np.zeros(self.env.mjModel.nu)
                action[self.env.legs_tau_idx.FL] = tau.FL.reshape((3,))
                action[self.env.legs_tau_idx.FR] = tau.FR.reshape((3,))
                action[self.env.legs_tau_idx.RL] = tau.RL.reshape((3,))
                action[self.env.legs_tau_idx.RR] = tau.RR.reshape((3,))
                self.env.step(action=action)

        
        # Fix convention DLS and send PD target
        if(config.robot == "aliengo"):
            desired_joint_pos.FL[0] = -desired_joint_pos.FL[0]
            desired_joint_pos.RL[0] = -desired_joint_pos.RL[0]   

        rl_signal_out_msg = rl_signal_out()
        rl_signal_out_msg.desired_joint_positions = np.concatenate([desired_joint_pos.FL, desired_joint_pos.FR, desired_joint_pos.RL, desired_joint_pos.RR], axis=0).flatten()
        rl_signal_out_msg.kp = np.ones(12)*Kp
        rl_signal_out_msg.kd = np.ones(12)*Kd
        self.publisher.publish(rl_signal_out_msg)
        
        
        
        # Render the simulation -----------------------------------------------------------------------------------
        if USE_MUJOCO_RENDER:
            RENDER_FREQ = 30
            # Render only at a certain frequency -----------------------------------------------------------------
            if time.time() - self.last_render_time > 1.0 / RENDER_FREQ or self.env.step_num == 1:
                self.env.render()
                self.last_render_time = time.time()




#---------------------------
if __name__ == '__main__':
    print('Hello from basic-locomotion-dls-isaaclab ros node.')
    
    rospy.init_node('basic_locomotion_dls_isaaclab_node', anonymous=True)
    basic_locomotion_dls_isaaclab_node = Basic_Locomotion_DLS_Isaaclab_Node()
    rospy.spin()

    print("basic-locomotion-dls-isaaclab ros node is stopped")
    exit(0)