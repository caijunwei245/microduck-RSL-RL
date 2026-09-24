# Deploying the MicroDuck policy into Isaac Lab — step-by-step

**Scope warning first.** This repository is an **mjlab (MuJoCo Warp) + rsl_rl** stack; it
contains no Isaac Lab code (verified: zero `isaac` references outside `logs/`).  Isaac Lab is
a different simulator (PhysX + USD) with different contact, actuator and randomization
semantics, so "deploying to Isaac Lab" is a **port**, not a copy.  Two different goals, very
different cost:

| goal | what it means | effort | verdict |
|---|---|---|---|
| **(A) Run the trained policy inside Isaac Lab** (sim-to-sim evaluation, extra sensors, parallel rendering) | import the MJCF as USD, rebuild the 61D observation, emulate the actuator, step the ONNX at 50 Hz | days; ~90% environment plumbing, ~10% policy | **recommended** |
| **(B) Port the training task to Isaac Lab and retrain** | re-implement rewards/commands/DR/curricula on Isaac Lab's manager API, recalibrate the actuator and contact model | weeks, and it invalidates the MuJoCo-side sim2real calibration this whole workflow rests on | only if you intend to leave mjlab |

Everything below is **(A)**.  Nothing here is verified against a running Isaac Lab — it is
the concrete plan derived from the actual contracts in this repo, with the numbers you
should reproduce at each step.

---

## 0. The contracts you must preserve (read off the exported ONNX)

From the deployment candidate `logs/scr_comb_64250.onnx` (metadata_props):

```
input      obs      (1, 61)      float32
output     actions  (1, 14)      float32
action_scale        1.0
joint_names         left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
                    neck_pitch, head_pitch, head_yaw, head_roll,
                    right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
default_joint_pos   0.000, -0.087, -0.458, -0.005, 0.453, 0.349, 0.349, 0.000, 0.000,
                    0.000, 0.087, 0.458, 0.005, -0.453
observation_names   base_ang_vel, projected_gravity, joint_pos, joint_vel, actions,
                    command, head_command, body_command
command_names       twist, head_pose, body_pose
joint_stiffness     1.0 x14      joint_damping 0.0 x14     (action-space placeholders)
```

The 61D observation is, in this exact order (no per-term scaling in the built cfg — see
`cfg.observations["actor"].terms`):

| slice | term | definition | training noise (do NOT replicate at eval) |
|---|---|---|---|
| 0:3 | `base_ang_vel` | trunk **body-frame** angular velocity | U(-0.03, 0.03) |
| 3:6 | `projected_gravity` | gravity direction in the body frame | U(-0.01, 0.01) |
| 6:20 | `joint_pos` | joint positions **relative to `default_joint_pos`** | U(-0.001, 0.001) |
| 20:34 | `joint_vel` | joint velocities | U(-0.25, 0.25) |
| 34:48 | `actions` | the **previous raw policy output** (zeros at t=0) | none |
| 48:51 | `command` | twist `[vx, vy, wz]` in the body frame | none |
| 51:55 | `head_command` | 4D head-pose delta command (zeros) | none |
| 55:61 | `body_command` | 6D body-pose delta command (zeros) | none |

**The observation normalizer is baked into the ONNX** (`src/mjlab_microduck/export.py` calls
`runner.export_policy_to_onnx`, which emits `actor(normalizer(obs))`).  Feed **raw** values;
never normalize again, and never hand-convert a checkpoint.

**Actions** are `JointPositionActionCfg(scale=1.0)` on top of `default_joint_pos`, i.e. the
14 outputs are **radian offsets from HOME**; the actuator (below) turns them into joint
torques.  Control rate **50 Hz** (0.02 s); training physics dt 0.002 s (decimation 10); the
real-body server uses 0.005 s because the actuator fit was calibrated there (AGENTS.md).

---

## 1. Prerequisites

- An Isaac Lab install (Isaac Sim + `isaaclab` python package) with a working `AppLauncher`.
- The policy ONNX: `logs/scr_comb_64250.onnx` (candidate) or any `logs/*.onnx` you exported.
  Copy it somewhere Isaac Lab can read; nothing else from `logs/` is needed.
- `onnxruntime` (CPU is fine — the policy is a 3-layer MLP) inside the same env, or run the
  policy in a separate process and pipe actions over a socket.
- The robot asset source: `src/mjlab_microduck/robot/microduck/robot_groundcontact.xml`
  (+ its `additional.xml`/`joints_properties.xml` includes) or a URDF produced by the same
  onshape-to-robot config (`config_mjcf_*.json`).
- The actuator model source: `src/mjlab_microduck/actuator/friction_dr_bam.py` (this is what
  you are going to reimplement; see §3).
- Reference numbers to check yourself against, all from the 140 s deployment rehearsal in
  mjlab/MuJoCo with the same ONNX (see `logs/deployment_handoff.md`):

| test | mjlab value (candidate 64250) |
|---|---|
| in-place turn, cmd wz = 0.50 | 0.656 rad/s (131 % of the required 0.5) |
| forward, cmd vx = +0.30 | +0.218 m/s (73 %) |
| forward, cmd vx = +0.20 / +0.40 | +0.147 (73 %) / +0.268 (67 %) |
| forward dead zone | cmd +0.10 → 0.000 m/s |
| yaw dead zone | cmd wz ≤ 0.35 → 0-38 % |
| zero command (idle) | 0.000 rad/s, 0.000 m/s, trunk z 116.7 mm |
| same policy with PD actuators (`--no-bam`) | 0.765 rad/s (153 %) |

---

## 2. Asset conversion (MJCF → USD)

1. Convert with Isaac Lab's MJCF importer (in recent versions
   `scripts/tools/convert_mjcf.py`; check your version's tooling) or, if you prefer, import the
   URDF that the same onshape-to-robot config produces.  Do it **once** and commit the USD.
2. Verify, in this order — each of these has bitten this project before:
   - **joint names and order** must equal the `joint_names` list in §0 (if the importer
     reorders or renames, fix the mapping in the observation assembly, never by editing the
     ONNX);
   - **joint limits/ranges**: read them from the compiled MuJoCo model
     (`left_hip_yaw [-0.436, 0.524]`, `right_hip_yaw [-0.524, 0.436]`, hip_roll ±0.384,
     knee/ankle ±1.571) and compare with the USD;
   - **masses/inertias/CoM**: total mass ≈ 0.8 kg; the head CoM work is hard-won (AGENTS.md
     warns that a wrong CoM degraded every long run for months);
   - **standing height**: holding `default_joint_pos` with a stiff PD should put the trunk at
     ~113-117 mm (measured: trunk z 116.5 mm while walking at cmd 0.30, 115.2 mm during the
     in-place turn).  A settle test that only checks "not fallen" is not enough — check tilt;
   - **foot geometry and friction**: the training range is mu ∈ (0.7, 1.3), nominal ~1.0, and
     the measured response is strongly friction-dependent (at mu 1.3 the candidate turns
     125 %, at mu ≥ 1.6 the gait collapses).  Set the friction and the **combine mode**
     explicitly; PhysX's default combine mode differs from MuJoCo's.
3. Do **not** import the BAM actuator or the DR definitions from the MJCF as ground truth —
   the actuator is a custom python kernel, not an XML feature.

---

## 3. Actuator emulation — the fidelity-critical step

In mjlab the joints are driven by `FrictionDRBamActuator` (`actuator/friction_dr_bam.py`):
a **voltage-controlled XL330 model** with `kp_fw = 200`, a per-env battery voltage
(6.5-8.2 V) with load-dependent sag (`V_drop = gain·Σ|τ|`, floor 6.0 V), forcerange ±0.963 Nm,
armature 1.81e-3, load-dependent friction, and **actuation delays of 3-6 physics steps**.
Isaac Lab has no equivalent.

**Path A — port the model (best fidelity).**  Subclass Isaac Lab's actuator base and copy the
kernel's math: position error → torque via the firmware loop, torque → current → voltage,
voltage sag, clip to forcerange, add armature; keep the delay as a ring buffer of actions.
The file is ~300 lines and self-contained; this is the only way to keep the sim2real
calibration that the deployment numbers rest on.

**Path B — approximate with a PD actuator (fast, lossy).**  For a voltage-mode joint,
`τ ≈ (kt/R)·(kp_fw·(q_des − q) − kt·ω)`, so an equivalent implicit PD is

```
stiffness = kt · kp_fw / R = 0.366 · 200 / 2.8114 ≈ 26.0 Nm/rad
damping   = kt² / R        = 0.366² / 2.8114      ≈ 0.0477 Nm·s/rad
effort_limit = 0.963 Nm, armature = 1.81e-3
```

(the kt/R/vin values are printed at rehearsal startup: `kt=0.3660 R=2.8114 vin=7.40V
kp_fw=200`).  You lose the voltage sag, the load-dependent friction and the 3-6 step delay —
those are exactly what the mjlab `--no-bam` (PD) rehearsal isolates: the same checkpoint
measures **0.765 rad/s with PD against 0.656 with BAM**, so treat that pair as your
calibration bracket: an Isaac Lab PD arm should land nearer 0.765, and a faithful BAM port
nearer 0.656.

---

## 4. Observation assembly (per control step, 50 Hz)

```python
obs = np.concatenate([
    ang_vel_b,                      # 3  trunk body-frame angular velocity
    quat_rotate_inverse(quat, [0,0,-1]),  # 3  projected gravity
    joint_pos - DEFAULT_JOINT_POS,  # 14 (DEFAULT_JOINT_POS from the ONNX metadata)
    joint_vel,                      # 14
    last_action,                    # 14 zeros at t=0
    [vx_cmd, vy_cmd, wz_cmd],       # 3  body-frame twist (0, 0, 0.5 for the turn test)
    np.zeros(4),                    # 4  head_command
    np.zeros(6),                    # 6  body_command
]).astype(np.float32)               # -> (61,) -> ONNX input shape (1, 61)
```
Then `action = session.run(["actions"], {"obs": obs[None]})[0][0]` (14,), set the joint
position targets to `default_joint_pos + action`, hold for `0.02 / physics_dt` physics steps
(10 at 0.002 s, 4 at 0.005 s), and feed `action` back as `last_action` on the next step.

Notes that matter:
- **No observation noise and no delays** at evaluation — those exist only in training DR.
- **Reset semantics**: on reset, zero the action history and take the obs after the reset
  state is written, exactly as the mjlab rehearsal does (`scripts/infer_policy.py` is the
  reference implementation of this loop; port its order of operations, not just its math).
- Spawn the robot at the same initial pose the rehearsal uses, otherwise the first second
  differs and you will chase a phantom discrepancy.

---

## 5. Verification protocol (what to measure, and what "pass" means)

Run the same four tests the deployment rehearsal runs, with the same 140 s horizon and the
same upright filter (discard samples with trunk z < 90 mm — a fallen robot must not flatter
the turn metric):

1. **in-place turn**: cmd (0, 0, 0.50).  Compare the mean body-frame yaw rate with 0.656.
2. **forward**: cmd (+0.30, 0, 0).  Compare with +0.218 m/s.
3. **idle**: cmd (0, 0, 0).  Must be ~0.000/0.000 and standing; if it drifts, the obs or the
   default pose is wrong, not the policy.
4. **command response shape**: wz ∈ {0.2, 0.35, 0.5, 0.7, 1.0} and vx ∈ {0.1, 0.2, 0.3, 0.4}.
   The *shape* (dead zone below ~0.35 rad/s and ~0.15 m/s, then roughly proportional) should
   survive the port even if the levels shift.

Report the result as a **sim-to-sim gap**, not as a pass/fail: expect the level to move.  Then
attribute it in this order (each is a known lever with a measured magnitude in mjlab):
friction and its combine mode (up to ±25 % on the turn), actuator model (BAM vs PD is a
17 % bracket on this checkpoint), physics dt (0.002 vs 0.005), contact stiffness/solref,
armature.  Only after those are matched does a residual gap say something about the policy.

---

## 6. If the real goal is (B), retraining in Isaac Lab

Then port, in this order, and expect the deployment numbers to move:
the 61D observation group (keep the layout — it is the hot-swap contract), the action term
(`scale=1.0`, default-offset), the command term (`twist` with the turn-in-place and
sustained-turn buckets, `head_pose`, `body_pose`), the reward stack from
`microduck_velocity_env_cfg.py` + `mdp.py` (⚠ the sign conventions in AGENTS.md: mjlab-base
costs return ≥ 0 with negative weights, the microduck `*_penalty` helpers self-negate and
take POSITIVE weights), the DR/event set, the curricula, the `nan_state` termination, and the
bounded-adaptive-LR patch (Patch 5 in `mdp.py`).  Keep `export.py`'s ONNX contract so
policies remain hot-swappable in the runtime.

Two things do **not** transfer and must be re-derived on the PhysX side: the contact
parameters (solref/friction/armature were tuned against MuJoCo's solver) and the actuator fit
(the real XL330 recordings were replayed through the BAM kernel, see
`scripts/validate_bam_testbench.py`).  Plan a calibration pass with the same XL330 test-bench
data before trusting any sim2real conclusion.

---

## 7. Honest status of this document

- Every **contract, number and file path** above is read from this repo in this session
  (ONNX metadata, `microduck_velocity_env_cfg.py`, `microduck_constants.py`, `export.py`,
  the rehearsal logs).
- **No step was executed against Isaac Lab** — there is no Isaac Lab in this environment, so
  the Isaac-Lab-side API names (importer script paths, actuator base classes, `AppLauncher`
  usage) are described by concept and **must be checked against your installed version**.
- The PD-equivalence numbers in §3 are derived from the printed BAM constants
  (`kt=0.366, R=2.8114, kp_fw=200`); they are the algebra of the kernel's linear regime, not a
  measured Isaac Lab result.
