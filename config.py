robot = 'aliengo'  # 'aliengo', 'go1', 'go2', 'b2', 'hyqreal1', 'hyqreal2', 'mini_cheetah' 

# ----------------------------------------------------------------------------------------------------------------
if(robot == "aliengo"):
    Kp_walking = 25.
    Kd_walking = 2.
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]


elif(robot == "go2"):
    Kp_walking = 20.
    Kd_walking = 1.5
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6   
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0] 

elif(robot == "b2"):
    Kp_walking = 20.
    Kd_walking = 1.5
    search_Kp_bounds = [-10., 10.]
    search_Kd_bounds = [-1., 2.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]

elif(robot == "hyqreal2"):
    Kp_walking = 175.
    Kd_walking = 20.
    search_Kp_bounds = [-50., 50.]
    search_Kd_bounds = [-5., 5.]

    friction_static = 0.2
    friction_dynamic = 0.6
    search_friction_static_bounds = [-0.2, 1.0]
    search_friction_dynamic_bounds = [-0.6, 1.0]

else:
    raise ValueError(f"Robot {robot} not supported")


Kp_sampling_interval = 0.1
Kd_sampling_interval = 0.1
friction_static_sampling_interval = 0.05
friction_dynamic_sampling_interval = 0.05

optimize_gain = True
optimize_friction = True


datasets_path = "./datasets/aliengo/from_reality"