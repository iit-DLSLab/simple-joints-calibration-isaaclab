# Overwiew

A simple joint calibration routine for IsaacLab, to estimate friction parameters and PD gains. It provides scripts for data collection on the real robot (the robot should be in air with the base fixed). Random sampling provides the best parameters fitting the saved trajectories.

Work in progress, PRs are very welcome!

### Run a calibration in IsaacLab

```bash
python3 calibrate_isaaclab.py --task=Locomotion-Aliengo-Flat --num_envs=8192  --headless
```
