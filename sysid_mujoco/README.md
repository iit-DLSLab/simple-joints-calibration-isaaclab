# MuJoCo SysID

This directory contains a MuJoCo-based system identification pipeline.

By default, it fits per-joint `damping`, `armature`, and `frictionloss` on the
recorded trajectory. From the repository root, run:

```bash
python sysid_mujoco/my_fit.py
```

Inertial identification is optional and has three hierarchical modes:

- `--identify-link-mass`: mass only (`InertiaType.Mass`).
- `--identify-center-of-mass`: mass and CoM (`InertiaType.MassIpos`).
- `--identify-inertia-tensor`: full inertia, including mass and CoM
  (`InertiaType.Pseudo`).

When multiple flags are supplied, the most complete requested mode is used.
For example, full inertial identification is enabled with:

```bash
python sysid_mujoco/my_fit.py \
  --identify-inertia-tensor
```

By default these parameters are added to every massive body downstream of an
articulated joint. Limit them to specific MuJoCo bodies with, for example:

```bash
python sysid_mujoco/my_fit.py \
  --identify-center-of-mass \
  --inertial-bodies link1 link2
```

The full-inertia mode uses MuJoCo SysID's pseudo-inertia Cholesky
parameterization, which guarantees physical consistency without singularities.
The optional bound controls are `--link-mass-scale-bounds`,
`--center-of-mass-offset-bounds`, `--inertia-tensor-scale-bounds`, and
`--inertia-tensor-shear-bounds`.

Notes:

- `my_fit.py` creates a fixed-base robot model and uses MuJoCo's integrated PD
  controller. It saves an HTML report under `sysid_mujoco/results/<robot>/`,
  where tracking and parameter values can be inspected.
