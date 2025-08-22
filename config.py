robot = 'aliengo'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.

    friction_static = 0.2
    friction_dynamic = 0.6


elif(robot == "go2"):
    Kp_walking = 20.
    Kd_walking = 1.5

    friction_static = 0.2
    friction_dynamic = 0.6    

elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5

    friction_static = 0.2
    friction_dynamic = 0.6

elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.

    friction_static = 0.2
    friction_dynamic = 0.6

else:
    raise ValueError(f"Robot {robot} not supported")


Kp_sampling_interval = 0.5
Kd_sampling_interval = 0.1


datasets_path = "./datasets/aliengo/from_reality"