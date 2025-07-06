import readline
import readchar
import time

from gym_quadruped.utils.quadruped_utils import LegsAttr

import numpy as np
import copy

class Console():
    def __init__(self, controller_node):
        self.controller_node = controller_node

        self.isDown = True
        self.isActivated = False

        # Autocomplete setup
        self.commands = [
            "help", "ictp", "goUp", "goDown", "startGeneration", "ictp", "setKp", "setKd"
        ]
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")


    def complete(self, text, state):
        options = [cmd for cmd in self.commands if cmd.startswith(text)]
        if state < len(options):
            print(options[state])
            return options[state]
        else:
            return None


    def interactive_command_line(self, ):
        self.print_all_commands()
        while True:
            input_string = input(">>> ")
            try:
                if(input_string == "goUp"):
                    print("Going Up")
                    if(not self.isDown):
                        print("The robot is already up")
                        continue

                                        
                    start_time = time.time()
                    time_motion = 5.

                    temp = copy.deepcopy(self.controller_node.joint_positions)
                    initial_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller_node.env.mjModel.nu/4))) for _ in range(4)])
                    initial_joint_positions.FL = temp[0:3]
                    initial_joint_positions.FR = temp[3:6]
                    initial_joint_positions.RL = temp[6:9]
                    initial_joint_positions.RR = temp[9:12]

                    reference_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller_node.env.mjModel.nu/4))) for _ in range(4)])
                    reference_joint_positions.FL = np.array([0, 0.9, -1.8])
                    reference_joint_positions.FR = np.array([0, 0.9, -1.8])
                    reference_joint_positions.RL = np.array([0, 0.9, -1.6])
                    reference_joint_positions.RR = np.array([0, 0.9, -1.6])
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]

                        self.controller_node.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller_node.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller_node.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller_node.stand_up_and_down_actions.RR = interpolated_positions[3]

                        time.sleep(0.01)

                    self.isDown = False


                elif(input_string == "goDown"):
                    print("Going Down")
                    if(self.isDown):
                        print("The robot is already down")
                        continue

                    self.isDown = True

                    start_time = time.time()
                    time_motion = 5.

                    temp = copy.deepcopy(self.controller_node.joint_positions)
                    initial_joint_positions = LegsAttr(*[np.zeros((1, int(self.controller_node.env.mjModel.nu/4))) for _ in range(4)])
                    initial_joint_positions.FL = temp[0:3]
                    initial_joint_positions.FR = temp[3:6]
                    initial_joint_positions.RL = temp[6:9]
                    initial_joint_positions.RR = temp[9:12]

                    reference_joint_positions.FL = np.array([0.0, 1.21, -2.794])
                    reference_joint_positions.FR = np.array([0.0, 1.21, -2.794])
                    reference_joint_positions.RL = np.array([0.0, 1.21, -2.794])
                    reference_joint_positions.RR = np.array([0.0, 1.21, -2.794])
                    while(time.time() - start_time < time_motion):
                        time_diff = time.time() - start_time
                        alpha = time_diff / time_motion
                        interpolated_positions = [
                            (1 - alpha) * initial + alpha * reference
                            for initial, reference in zip(initial_joint_positions, reference_joint_positions)
                        ]
            
                        self.controller_node.stand_up_and_down_actions.FL = interpolated_positions[0]
                        self.controller_node.stand_up_and_down_actions.FR = interpolated_positions[1]
                        self.controller_node.stand_up_and_down_actions.RL = interpolated_positions[2]
                        self.controller_node.stand_up_and_down_actions.RR = interpolated_positions[3]

                        time.sleep(0.01)

                    
                elif(input_string == "startCollection"):
                    self.isActivated = not self.isActivated


                elif(input_string == "help"):
                    self.print_all_commands()


                elif(input_string == "setKp"):
                    print("Kp stand_up_and_down: ", self.controller_node.Kp_stand_up_and_down)
                    temp = input("Enter Kp: ")
                    if(temp != ""):
                        self.controller_node.Kp_stand_up_and_down = float(temp)
                    


                elif(input_string == "setKd"):
                    print("Kd stand_up_and_down: ", self.controller_node.Kd_stand_up_and_down)
                    temp = input("Enter Kd: ")
                    if(temp != ""):
                        self.controller_node.Kd_stand_up_and_down = float(temp)


            
            except Exception as e:
                print("Error: ", e)
                print("Invalid Command")
                self.print_all_commands()


    def print_all_commands(self):
        print("\nAvailable Commands")
        print("help: Display all available messages")
        print("ictp: Interactive Keyboard Control\n")