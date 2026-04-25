# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

```cmd
conda create -n climb python=3.10
conda activate climb
conda install numpy pybullet gymnasium stable-baselines3 wandb --channel conda-forge
pip install stable-baselines3[extra]
```

Training uses CUDA when available (`torch.cuda.is_available()`), otherwise CPU. `wandb` is used for run tracking — `wandb login` must be configured before running `train.py` / `autotrain.py`.

## Common commands

Train PPO from scratch with N parallel workers (uses `SubprocVecEnv` with `start_method="spawn"` — must run under `if __name__ == '__main__':`, which `train.py` already does):

```cmd
python train.py HumanoidClimb-v0 PPO -w 4 -t
```

Continue training from an existing checkpoint:

```cmd
python train.py HumanoidClimb-v0 PPO -w 4 -t -f path/to/model.zip
```

Test a trained single-stance model in the GUI (uses `STANCE_14_1` hardcoded in `train.py`):

```cmd
python train.py HumanoidClimb-v0 PPO -s path/to/model.zip
```

Run the full multi-stance climb demo (chains `STANCE_1`..`STANCE_4` policies sequentially with GUI controls r/space/q):

```cmd
python main.py        # single PPO model loaded from humanoid_climb/models/
python humanoid_climb/climb.py   # chains 4 PPO policies, one per stance
```

Generate end-of-stance state files (`.npz` saved under `humanoid_climb/states/`) by rolling out a trained model:

```cmd
python collect_states.py   # edit STANCE / MODEL_FILE constants at top first
```

Inspect joint/part names from `humanoid_symmetric.xml` (PyBullet GUI):

```cmd
python joint_test.py
```

There are no tests, linters, or build steps in this repo.

## Architecture

This is a hierarchical climbing-RL setup: one PPO policy is trained per stance transition (a "stance" = which target hold each of the 4 effectors is gripping), then policies are chained at evaluation time.

### Data flow

1. **`config.json`** is the single source of truth for the simulation: assets (URDF/MJCF paths), climber joint forces and collision groups, holds (3D positions on the wall), and a `stance_path` (sequence of `desired_holds` / `force_attach` / `ignore_holds`). `ClimbingConfig` (`humanoid_climb/climbing_config.py`) loads it and resolves asset references.
2. **`HumanoidClimbEnv`** (`humanoid_climb/env/humanoid_climb_env.py`, registered as `HumanoidClimb-v0` in `humanoid_climb/__init__.py`) consumes the config, builds the PyBullet scene (plane, wall, holds, humanoid), and exposes a 21-dim action / 306-dim observation gym env. The motion path is read from the config's `stance_path`; each step advances `desired_stance_index` when `current_stance == desired_stance`.
3. **`Humanoid`** (`humanoid_climb/assets/humanoid.py`) wraps the MJCF robot. The 21-dim action is split: first 17 are joint torques (scaled by `power * joint_forces[name]`), last 4 are grasp signals (one per effector — left/right hand/foot). Grasp > 0 triggers `attach()`, which creates a PyBullet `JOINT_POINT2POINT` constraint to the nearest hold within penetration distance (uses `getClosestPoints` with 0.0 threshold). `force_attach` constraints can be overridden per-stance via `action_override`/`force_attach`, and excluded holds via `exclude_targets`/`ignore_holds`.
4. **`Asset`** (`humanoid_climb/assets/asset.py`) is a thin wrapper for static URDF/MJCF objects (wall, holds, plane). `robot_util.addToScene` (PyBullet's standard helper) populates `parts`/`joints`/`ordered_joints`.

### Stance system

Two parallel definitions exist for stances — they are NOT auto-synced:

- **`config.json` → `stance_path`**: used by `HumanoidClimbEnv` directly when constructed with a `config=` kwarg (the path used by `main.py`, `train.py`, `climb.py`, `autotrain.py`).
- **`humanoid_climb/stances/__init__.py`**: hand-defined `Stance` objects (`STANCE_1`..`STANCE_5`, plus many commented-out `STANCE_6`..`STANCE_14_1`). `Stance.get_args()` returns kwargs (`motion_path`, `state_file`, `action_override`, `motion_exclude_targets`) for an alternate env-construction path used by `collect_states.py` and the `--test` branch of `train.py`. Call `stances.set_root_path("./humanoid_climb")` before `get_args()` so `state_file` resolves correctly.

Hold IDs in `stances/__init__.py` are **integers** (e.g. `[10, 9, -1, -1]`), while `config.json` uses **string keys** (e.g. `["hold_10", "hold_9", -1, -1]`). When editing the motion path, update whichever source the entry-point script you're running uses.

### Reward functions

`HumanoidClimbEnv` contains several reward functions; the one called from `step()` is hardcoded (currently `calculate_reward_negative_distance`). Switch by editing the call site in `step()`. `calculate_reward_eq1` is the original paper reward; `calculate_improved_reward` and the commented `calculate_advanced_dyno_reward` are dyno-movement experiments (see recent commits on the `humanoid` branch).

### Pretrained models

`humanoid_climb/models/*.zip` are PPO checkpoints named `{stance_index}_{hold_lh}_{hold_rh}_{hold_lf}_{hold_rf}.zip` (`n` = no hold, `-1`). `climb.py` chains them; `main.py` loads one. They were trained against the integer-stance system in `stances/__init__.py`, not the current `config.json`.

### Observation shape

`_get_obs` produces 306 floats: per-joint `(worldPos, worldOri, localInertialPos, linearVel, angVel)` for all ordered joints, plus per-effector target position + distance, current/desired stance hold IDs, per-effector reached flags, `best_dist_to_stance`, and floor/wall contact bits. Changing the joint count, effector count, or this layout requires updating `observation_space` shape.

## Project history & research context

Full dissertation text is gitignored at `_private/dissertation.txt` (extracted from `C:\Sid\Aston\Dissertation\Final Dissertation\CS4700_230271746_Dissertation.pdf`). Read it directly when you need detail beyond this summary.

**Lineage**: this repo is a fork of Dylan Goes' "Climbing Motion Discovery" (Aston 2023). The 2024 dissertation by Siddharth Sharma (supervisor Dr. Martin Rudorfer) extended it specifically to study **dyno movement** — a four-limb dynamic leap where all end-effectors detach from the wall mid-flight. The agent does not currently perform a real dyno; the recurring failure mode is the agent **briefly touching the floor with one leg to bounce off and gain upward momentum** (Run 5, §6.2.5).

### Reward equations as the dissertation defined them

Naming kept consistent with §3.5 / §5 of the report so equation references map cleanly:

- **Eq 1** — `R_distance = clip(-1 * Σ current_dist_away, -2, ∞)` — original distance-only reward.
- **Eq 2** — `R_velocity = max(0, V_z) * k`, `k = 4`. Rewards upward COM velocity only.
- **Eq 3** — `R_slouch = max(0, |θ_target| - |θ_slouch - θ_target|) * k`, `k = 0.5`, `θ_target = -π/6` (30° backward lean).
- **Eq 4** — `R_floor = +0.1 if airborne, -5 if on floor`.
- **Eq 5** — `R_stance = +1000` on reaching desired stance, else `0`.
- **Eq 6** — `R = R_distance + R_velocity + R_slouch + R_floor + R_stance`.
- **Eq 7** — `F_wall = Σ (|F_normal,i| - T_impact)` for each contact whose normal force exceeds threshold `T_impact` (set to 100 in Run 5).
- **Eq 8** — `R_total = Eq6 - 0.1 * F_wall`. **This is the dissertation's final reward.**

### Code-versus-dissertation drift (important)

What the current `humanoid_climb_env.py` actually contains differs from the report:

- `calculate_reward_negative_distance` (the active one in `step()`) implements only Eq 1 + a *modified* floor penalty `(max_ep_steps - steps) * -2` — not Eq 4. No stance bonus, no velocity term, no slouch term, no wall impact.
- `calculate_reward_eq1` adds a `+3000` stance bonus, not the dissertation's `+1000`.
- `calculate_improved_reward` is close to Eq 6 *minus the stance reward*, with `R_velocity` scaling factor `2` instead of `4`. The slouch term uses `target_slouch = -π/6` correctly.
- `calculate_advanced_dyno_reward` is commented out and was never functional anyway — it called `self.climber.robot_body.speed()` / `current_orientation()` which don't exist (the helpers `Humanoid.speed()` / `get_orientation()` were added later and operate on `self.robot`, not `robot_body`).
- **Eq 7/8 (wall normal-force impact penalty) is not implemented at all.** The wall is only used as a binary `is_touching_body(self.wall.id)` flag in the observation — no per-contact `getContactPoints` aggregation, no threshold logic.
- `hold_18` was relocated from `[0.4, 0.3, 3.05]` to `[0.4, 0, 2.9]` — the dyno target is now centered (matches `["hold_18", "hold_18", -1, -1]` for stance 4, both hands aiming for one center hold).

If you're trying to **reproduce the dissertation's Run 5 baseline**, none of the three reward functions in the file match it. You'd need to add `R_distance + R_velocity(k=4) + R_slouch(k=0.5) + R_floor + R_stance(+1000) - 0.1 * F_wall` and implement `F_wall` from `getContactPoints(robot, wall)` normal forces.

### Five experimental runs (one-line each)

All used 12 workers, PyBullet, PPO. Success rate stayed ≈0% throughout except Run 3.

1. **Run 1** — Eq 1, dyno-only env, 10M steps, default hyperparams. Success 0%. Agent learned to *survive*, not climb.
2. **Run 2** — Eq 1, full 20-hold env, 25M steps. Success 0%. Partial jumps but no real transition.
3. **Run 3** — Eq 6 + hip 600 / knee 400, 25M steps. Success ~4% — but driven by *uncontrolled blast-off* into the wall and accidental proximity to target, not real dyno.
4. **Run 4** — Eq 8, hip 400 / knee 300, `ent_coef=0.01, n_steps=4096, lr=1e-4`, 25M steps. Success 0%. Left leg got *stuck attached* to its hold during the leap.
5. **Run 5** — Eq 8, default hyperparams, `T_impact=100`, 25M steps. Success 0%. Left leg briefly touched the *floor* mid-leap to bounce off — the canonical failure case.

### Self-identified limitations (§6.3 + §7 future work)

- **Action override mechanism** (`force_attach` in `config.json` / `Humanoid.apply_action`) **manually controls grasp/release** for some stances. The report flags this as a primary limitation — the agent cannot autonomously decide *when* to release all four limbs, which is exactly what a dyno requires.
- Distance-only reward gives no credit for momentum coordination.
- Wall-impact penalty (even the dissertation's version) doesn't capture *timing* — only magnitude.
- Only vertical stance transitions tested; no lateral / diagonal.

### Key references for future reward / planner design

- **Naderi (2020), "Discovering and Synthesizing Humanoid Climbing Movements"** (Aalto PhD) — the most directly relevant prior work. Combines high-level graph-based planning with low-level sampling optimization, uses NN to predict movement-success rates for leaps. This is the template for hierarchical climbing.
- Goes (2024) — the prior Aston dissertation this repo forks.
- Jaramillo-Martínez et al. (2024) — reward shaping for path planning.
- Pan & Bao (2019) — driving-style reward terms (smoothness, safety).

## Gotchas

- `train.py` mixes tabs and spaces — `make_env` is space-indented, the surrounding code is tab-indented. Match the surrounding style when editing.
- `SubprocVecEnv` is created with `start_method="spawn"`; on Windows the training scripts must stay guarded by `if __name__ == '__main__':`.
- `humanoid_symmetric.xml` is loaded with `URDF_USE_SELF_COLLISION`, and collision filter groups/masks are set per-body-part from `config.json` to prevent specific limb-pair self-collisions — changing the collision_groups values can cause silent self-tangling.
- `Humanoid.set_state` / `initialise_from_state` are present but `init_from_state` is not currently wired up in `reset()` (the call is commented out).
- `notes.txt` records project-specific quirks (grasp must be > 0 to attach; stances must match effector count; etc.).
