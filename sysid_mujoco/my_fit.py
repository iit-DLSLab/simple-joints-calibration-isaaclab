from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sysid_mujoco.common import build_fixed_base_model_xml
from sysid_mujoco.common import build_actuator_gain_map
from sysid_mujoco.common import build_parameter_dict
from sysid_mujoco.common import chunk_processed_trajectory
from sysid_mujoco.common import get_actuated_joint_and_actuator_names
from sysid_mujoco.common import load_dataset_actuator_gains
from sysid_mujoco.common import load_processed_dataset
from sysid_mujoco.common import ProcessedTrajectory
from sysid_mujoco.common import processed_to_sysid_trajectory

import mujoco
import mujoco.rollout as rollout
from mujoco import sysid
import numpy as np
import matplotlib.pyplot as plt
import mediapy as media
from absl import logging
import config
import numpy as np


PIPER_TORQUE_SCALE = 4.
PIPER_TORQUE_SCALED_JOINTS = {"joint1", "joint2", "joint3"}


def default_converted_paths(robot: str) -> list[Path]:
    return sorted((REPO_ROOT / "sysid_mujoco" / "converted" / robot).glob("*.npz"))


def default_output_dir(robot: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "sysid_mujoco" / "results" / robot / timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-joint damping, armature, and frictionloss from "
            "quadruped datasets using MuJoCo sysid."
        )
    )
    parser.add_argument(
        "--robot",
        default=config.robot,
        help="Robot name. Defaults to config.robot.",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        type=Path,
        default=[Path(str(REPO_ROOT) + "/datasets/" + config.robot + "/traj_0.pt")],
        help="Raw repository datasets (.pt). If provided, conversion is done in-memory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save the fitted parameters and identified XML models.",
    )
    parser.add_argument(
        "--optimizer",
        choices=("mujoco", "scipy", "scipy_parallel_fd"),
        default="mujoco",
        help="Optimizer backend exposed by mujoco.sysid.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum optimizer iterations.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="If > 0, split each source trajectory into non-overlapping chunks of this size.",
    )
    parser.add_argument(
        "--damping-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.1, 3.0),
        help="Bounds for each joint damping parameter.",
    )
    parser.add_argument(
        "--armature-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.001, 0.6),
        help="Bounds for each joint armature parameter.",
    )
    parser.add_argument(
        "--frictionloss-bounds",
        nargs=2,
        type=float,
        metavar=("LOWER", "UPPER"),
        default=(0.01, 5.0),
        help="Bounds for each joint frictionloss parameter.",
    )
    return parser.parse_args()


def _is_piper_robot(robot: str) -> bool:
    return robot == "piper_l"


def _model_joint_names(model) -> list[str]:
    return [model.joint(joint_id).name for joint_id in range(model.njnt)]


def _scale_actuator_gains_for_fit(
    robot: str,
    joint_names: list[str],
    kp: np.ndarray,
    kd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scaled_kp = np.asarray(kp, dtype=np.float64).copy()
    scaled_kd = np.asarray(kd, dtype=np.float64).copy()
    if not _is_piper_robot(robot):
        return scaled_kp, scaled_kd

    for index, joint_name in enumerate(joint_names):
        if joint_name in PIPER_TORQUE_SCALED_JOINTS:
            scaled_kp[index] *= PIPER_TORQUE_SCALE
            scaled_kd[index] *= PIPER_TORQUE_SCALE
    return scaled_kp, scaled_kd


def _prepare_fixed_base_xml_for_fit(robot: str, model_xml: Path) -> Path:
    if not _is_piper_robot(robot):
        return model_xml

    tree = ET.parse(model_xml)
    root = tree.getroot()
    joint_names = {
        joint.get("name")
        for joint in root.iter("joint")
        if joint.get("name") is not None
    }
    if "joint8" not in joint_names:
        raise ValueError("Piper model must contain `joint8`.")

    sensor_element = root.find("sensor")
    if sensor_element is None:
        sensor_element = ET.SubElement(root, "sensor")

    for sensor in sensor_element.findall("jointpos"):
        if sensor.get("joint") == "joint8":
            return model_xml

    ET.SubElement(
        sensor_element,
        "jointpos",
        {
            "name": "joint8_pos",
            "joint": "joint8",
        },
    )
    tree.write(model_xml, encoding="utf-8", xml_declaration=True)
    return model_xml


def _insert_piper_mimic_joint8(
    values: np.ndarray,
    model_joint_names: list[str],
    value_name: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"`{value_name}` must be a 2-D array.")
    if "joint7" not in model_joint_names or "joint8" not in model_joint_names:
        raise ValueError("Piper model must contain both `joint7` and `joint8`.")
    if values.shape[1] == len(model_joint_names):
        return values

    source_joint_names = [name for name in model_joint_names if name != "joint8"]
    if values.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Piper `{value_name}` has {values.shape[1]} joints, expected "
            f"{len(source_joint_names)} without `joint8` or "
            f"{len(model_joint_names)} with `joint8`."
        )

    joint7_index = source_joint_names.index("joint7")
    joint8_index = model_joint_names.index("joint8")
    joint8_values = -values[:, joint7_index:joint7_index + 1]
    return np.concatenate(
        (
            values[:, :joint8_index],
            joint8_values,
            values[:, joint8_index:],
        ),
        axis=1,
    )


def _extend_piper_state_to_joint8(
    trajectory: ProcessedTrajectory,
    model,
) -> ProcessedTrajectory:
    model_joint_names = _model_joint_names(model)
    return replace(
        trajectory,
        measured_qpos=_insert_piper_mimic_joint8(
            trajectory.measured_qpos,
            model_joint_names,
            "measured_qpos",
        ),
        measured_qvel=_insert_piper_mimic_joint8(
            trajectory.measured_qvel,
            model_joint_names,
            "measured_qvel",
        ),
        desired_qpos=_insert_piper_mimic_joint8(
            trajectory.desired_qpos,
            model_joint_names,
            "desired_qpos",
        ),
        joint_names=model_joint_names,
    )


def build_model_sequences_from_source(
    sysid,
    mujoco,
    model,
    robot: str,
    dataset_paths: list[Path],
    chunk_size: int,
):
    measurement_ts = []
    control_ts = []
    initial_states = []

    for dataset_path in dataset_paths:

        processed = load_processed_dataset(
            dataset_path=dataset_path,
            model=model,
            mujoco=mujoco,
        )
        if _is_piper_robot(robot):
            processed = _extend_piper_state_to_joint8(processed, model)
        for chunk in chunk_processed_trajectory(processed, chunk_size):
            measurement_data, control_data, initial_state = processed_to_sysid_trajectory(sysid, model, chunk)
            measurement_ts.append(measurement_data)
            control_ts.append(control_data)
            initial_states.append(initial_state)

    return measurement_ts, control_ts, initial_states


def main() -> None:
    args = parse_args()
    if not args.dataset:
        raise ValueError("Use --dataset to pass a dataset")

    dataset_kp, dataset_kd = load_dataset_actuator_gains(args.dataset[0])


    fixed_base_xml = _prepare_fixed_base_xml_for_fit(
        args.robot,
        build_fixed_base_model_xml(args.robot),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_model = fixed_base_spec.compile()
    actuated_joint_names, _ = get_actuated_joint_and_actuator_names(mujoco, fixed_base_model)
    actuator_kp, actuator_kd = _scale_actuator_gains_for_fit(
        args.robot,
        actuated_joint_names,
        dataset_kp,
        dataset_kd,
    )
    fixed_base_xml = _prepare_fixed_base_xml_for_fit(
        args.robot,
        build_fixed_base_model_xml(
            args.robot,
            actuator_gains=build_actuator_gain_map(
                actuated_joint_names,
                actuator_kp,
                actuator_kd,
            ),
        ),
    )
    fixed_base_spec = mujoco.MjSpec.from_file(str(fixed_base_xml))
    fixed_base_model = fixed_base_spec.compile()

    measurement_ts, control_ts, initial_states = build_model_sequences_from_source(
        sysid=sysid,
        mujoco=mujoco,
        model=fixed_base_model,
        robot=args.robot,
        dataset_paths=args.dataset,
        chunk_size=args.chunk_size,
    )

    joint_names = [fixed_base_model.joint(i).name for i in range(fixed_base_model.njnt)]
    sequence_names = [f"sequence_{index:03d}" for index, _ in enumerate(measurement_ts)]
    model_sequences = [sysid.ModelSequences(
        args.robot,
        fixed_base_spec,
        sequence_names[i],
        initial_states[i],
        control_ts[i],
        measurement_ts[i],
    ) for i in range(len(measurement_ts))]

    bounds = {
        "damping": tuple(float(value) for value in args.damping_bounds),
        "armature": tuple(float(value) for value in args.armature_bounds),
        "frictionloss": tuple(float(value) for value in args.frictionloss_bounds),
    }

    params = build_parameter_dict(
        sysid=sysid,
        model=fixed_base_model,
        joint_names=joint_names,
        bounds=bounds,
    )
   
    residual_fn = residual_fn = sysid.build_residual_fn(models_sequences=model_sequences)

    opt_params, opt_result = sysid.optimize(
    initial_params=params,
    residual_fn=residual_fn,
    optimizer='mujoco'
    )



    output_dir = args.output_dir or default_output_dir(args.robot)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = sysid.default_report(
    models_sequences=model_sequences,
    initial_params=params,
    opt_params=opt_params,
    residual_fn=residual_fn,
    opt_result=opt_result,
    title=f"System Identification Report for {args.robot}",
    generate_videos=False,
    )
    def display_report(report, report_path: Path) -> Path:
        html = report.build()
        report_path.write_text(html, encoding="utf-8")
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display

            if getattr(get_ipython(), "kernel", None) is not None:
                display(HTML(html))
        except Exception:
            pass
        print(f"Report written to {report_path}")
        return report_path

    display_report(report, output_dir / "report.html")


if __name__ == "__main__":
    main()
