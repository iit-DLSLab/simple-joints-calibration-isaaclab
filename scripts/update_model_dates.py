#!/usr/bin/env python3
"""Update the robot-model modification dates in the repository README."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "robot_model"
README = ROOT / "README.md"
START_MARKER = "<!-- model-dates:start -->"
END_MARKER = "<!-- model-dates:end -->"
MODEL_SUFFIXES = {
    "XML": (".xml",),
    "USD": (".usd", ".usda", ".usdc"),
}


def last_change(paths: list[Path]) -> str:
    """Return the most recent commit date affecting any of the given paths."""
    if not paths:
        return "—"

    relative_paths = [str(path.relative_to(ROOT)) for path in paths]
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", *relative_paths],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or "—"


def build_table() -> str:
    rows = [
        START_MARKER,
        "| Robot | Ultima modifica XML | Ultima modifica USD |",
        "|:--|:--:|:--:|",
    ]

    for robot_dir in sorted(path for path in MODELS_DIR.iterdir() if path.is_dir()):
        dates = {}
        for model_type, suffixes in MODEL_SUFFIXES.items():
            paths = sorted(
                path
                for path in robot_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in suffixes
            )
            dates[model_type] = last_change(paths)

        if any(value != "—" for value in dates.values()):
            robot_link = f"[{robot_dir.name}](./robot_model/{robot_dir.name})"
            rows.append(f"| {robot_link} | {dates['XML']} | {dates['USD']} |")

    rows.append(END_MARKER)
    return "\n".join(rows)


def main() -> None:
    content = README.read_text(encoding="utf-8")
    if START_MARKER not in content or END_MARKER not in content:
        raise RuntimeError("Model-date markers are missing from README.md")

    before, remainder = content.split(START_MARKER, maxsplit=1)
    _, after = remainder.split(END_MARKER, maxsplit=1)
    README.write_text(f"{before}{build_table()}{after}", encoding="utf-8")


if __name__ == "__main__":
    main()
