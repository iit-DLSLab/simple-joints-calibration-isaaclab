"""Plot desired and measured joint trajectories from a collected dataset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch


DATASETS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATASETS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_dataset = DATASETS_DIR / config.robot / "trajectory_1.pt"

    parser = argparse.ArgumentParser(
        description="Plot desired and real position trajectories in one subplot per joint."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=default_dataset,
        help=f"Dataset .pt file (default: {default_dataset})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional output image path, for example plots/trajectory.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open the plot window (useful together with --save).",
    )
    return parser.parse_args()


def to_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_trajectories(dataset_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        data = torch.load(dataset_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with PyTorch versions that do not have weights_only.
        data = torch.load(dataset_path, map_location="cpu")

    required_keys = ("time", "dof_pos", "des_dof_pos")
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(
            f"{dataset_path} is missing required keys: {', '.join(missing_keys)}"
        )

    time = to_numpy(data["time"]).squeeze()
    real_position = to_numpy(data["dof_pos"])
    desired_position = to_numpy(data["des_dof_pos"])

    if time.ndim != 1:
        raise ValueError(f"Expected time to be one-dimensional, got shape {time.shape}.")
    if real_position.ndim != 2 or desired_position.ndim != 2:
        raise ValueError(
            "Expected dof_pos and des_dof_pos to have shape (time, joints), "
            f"got {real_position.shape} and {desired_position.shape}."
        )
    if real_position.shape != desired_position.shape:
        raise ValueError(
            "Real and desired trajectories have different shapes: "
            f"{real_position.shape} and {desired_position.shape}."
        )
    if len(time) != real_position.shape[0]:
        raise ValueError(
            f"Time has {len(time)} samples but the trajectories have "
            f"{real_position.shape[0]}."
        )

    return time, real_position, desired_position


def plot_trajectories(
    time: np.ndarray,
    real_position: np.ndarray,
    desired_position: np.ndarray,
    dataset_path: Path,
):
    import matplotlib.pyplot as plt

    num_joints = real_position.shape[1]
    num_cols = min(3, num_joints)
    num_rows = math.ceil(num_joints / num_cols)
    figure, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5.2 * num_cols, 3.1 * num_rows),
        sharex=True,
        squeeze=False,
    )

    for joint_index, axis in enumerate(axes.flat):
        if joint_index >= num_joints:
            axis.set_visible(False)
            continue

        axis.plot(
            time,
            desired_position[:, joint_index],
            label="Desired",
            color="tab:orange",
            linestyle="--",
            linewidth=1.8,
        )
        axis.plot(
            time,
            real_position[:, joint_index],
            label="Real",
            color="tab:blue",
            linewidth=1.4,
        )
        axis.set_title(f"Joint {joint_index + 1}")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Position [rad]")
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.suptitle(f"Desired and real joint trajectories — {dataset_path.name}")
    figure.tight_layout()
    return figure


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    time, real_position, desired_position = load_trajectories(dataset_path)
    figure = plot_trajectories(
        time,
        real_position,
        desired_position,
        dataset_path,
    )

    if args.save is not None:
        output_path = args.save.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
