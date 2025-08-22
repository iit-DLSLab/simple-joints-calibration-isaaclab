import numpy as np
import torch
import os


def load_datasets(datasets_path, expected_joint_order):
    """
    Load the dataset from the specified path.
    Returns:
        dataset_actual_joint_pos: Actual joint positions.
        dataset_actual_joint_vel: Actual joint velocities.
        dataset_desired_joint_pos: Desired joint positions.
        dataset_desired_joint_vel: Desired joint velocities.
        dataset_joint_names: Names of the joints in the dataset.
        dataset_fps: Frames per second of the dataset.
    """

    files = os.listdir(datasets_path)
    number_of_files = len(files)
    print(f"Number of trajectory in '{datasets_path}': {len(files)}")
    print("Files:", files)


    all_dataset_actual_joint_pos = None
    all_dataset_actual_joint_vel = None
    all_dataset_desired_joint_pos = None
    all_dataset_desired_joint_vel = None

    for i, file in enumerate(files):
        print(f"{i + 1}/{number_of_files}: {file}")


        data = np.load(datasets_path+"/"+file, allow_pickle=True).item()
        dataset_actual_joint_pos = data["actual_joints_position"]
        dataset_actual_joint_vel = data["actual_joints_velocity"]
        dataset_desired_joint_pos = data["desired_joints_position"]
        dataset_desired_joint_vel = data["desired_joints_velocity"]
        dataset_joint_names = data["joints_list"]
        dataset_fps = data["fps"]

        # reset environment
        timestep = 0


        # build index map for expected_joint_order
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_order:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in dataset_actual_joint_pos:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)
        dataset_actual_joint_pos = np.stack(jp_list, axis=0)

        # reorder & fill joint velocities
        jv_list: List[np.ndarray] = []
        for frame in dataset_actual_joint_vel:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jv_list.append(arr)
        dataset_actual_joint_vel = np.stack(jv_list, axis=0)

        # reorder & fill desired joint positions
        jp_list = []
        for frame in dataset_desired_joint_pos:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)
        dataset_desired_joint_pos = np.stack(jp_list, axis=0)

        # reorder & fill desired joint velocities
        jv_list = []
        for frame in dataset_desired_joint_vel:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jv_list.append(arr)
        dataset_desired_joint_vel = np.stack(jv_list, axis=0)


        # attach a termination frame to the end of each dataset
        termination_frame = np.zeros((len(expected_joint_order),), dtype=dataset_actual_joint_pos.dtype) - 10.
        dataset_actual_joint_pos = np.concatenate(
            (termination_frame[None, :], dataset_actual_joint_pos), axis=0)
        dataset_actual_joint_vel = np.concatenate(
            (termination_frame[None, :], dataset_actual_joint_vel), axis=0)
        dataset_desired_joint_pos = np.concatenate(
            (termination_frame[None, :], dataset_desired_joint_pos), axis=0)
        dataset_desired_joint_vel = np.concatenate(
            (termination_frame[None, :], dataset_desired_joint_vel), axis=0)
        
        # concatenate datasets
        if(all_dataset_actual_joint_pos is None):
            all_dataset_actual_joint_pos = dataset_actual_joint_pos
            all_dataset_actual_joint_vel = dataset_actual_joint_vel
            all_dataset_desired_joint_pos = dataset_desired_joint_pos
            all_dataset_desired_joint_vel = dataset_desired_joint_vel
        else:
            # concatenate along the first axis
            all_dataset_actual_joint_pos = np.concatenate(
                (all_dataset_actual_joint_pos, dataset_actual_joint_pos), axis=0)
            all_dataset_actual_joint_vel = np.concatenate(
                (all_dataset_actual_joint_vel, dataset_actual_joint_vel), axis=0)
            all_dataset_desired_joint_pos = np.concatenate(
                (all_dataset_desired_joint_pos, dataset_desired_joint_pos), axis=0)
            all_dataset_desired_joint_vel = np.concatenate(
                (all_dataset_desired_joint_vel, dataset_desired_joint_vel), axis=0)
        
    return {
        "all_dataset_actual_joint_pos": all_dataset_actual_joint_pos,
        "all_dataset_actual_joint_vel": all_dataset_actual_joint_vel,
        "all_dataset_desired_joint_pos": all_dataset_desired_joint_pos,
        "all_dataset_desired_joint_vel": all_dataset_desired_joint_vel,
        "dataset_fps": dataset_fps
    }


if __name__ == '__main__':
    load_datasets()
    print("Done loading datasets.")