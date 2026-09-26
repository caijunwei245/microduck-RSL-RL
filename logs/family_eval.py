"""Deterministic, training-side evaluation of one checkpoint — works for ANY MicroDuck task.

This is the generic sibling of logs/kick_reset_probe.py: the kick probe could measure a ball
because that task has a physical score; every other family has to be judged by the env's own
signals. So this reports, over N parallel envs run deterministically for a whole episode:

  * mean episode return and length (and the fraction of envs still alive at the end),
  * every termination reason the env fired (time_out vs task-specific falls/failures),
  * every Episode_Metrics/* the task logs (these are the task's own competence numbers:
    velocity tracking error, pose error, height, phase progress...),
  * the same Episode_Reward/* decomposition training prints, so a checkpoint can be read
    against its own training curve.

Run one checkpoint:
    CUDA_VISIBLE_DEVICES=0 uv run python logs/family_eval.py <task_id> <checkpoint.pt> [envs] [steps] [seed]
Prints one JSON object on the last line. Rank families on DISTRIBUTIONS of these rows
(several checkpoints), never on a single one.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

import mjlab_microduck.tasks  # noqa: F401  (registrations + patches)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls


class Actor:
    """Hand-loaded policy: its own mlp weights AND its own obs normalizer (see main() for why)."""

    def __init__(self, ckpt: str, clip: float | None = None, device: str = "cuda:0"):
        # `device` is a parameter, not a module constant: family_eval keeps it local to main()
        d = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = d["obs_normalizer._mean"].to(device)
        self.std = d["obs_normalizer._std"].to(device)
        self.layers = []
        i = 0
        while f"mlp.{i}.weight" in d:
            self.layers.append((d[f"mlp.{i}.weight"].to(device), d[f"mlp.{i}.bias"].to(device)))
            i += 2
        assert self.layers, f"no mlp layers in {ckpt}"
        self.clip = clip

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / (self.std + 1e-8)
        for w, b in self.layers[:-1]:
            x = F.elu(x @ w.T + b)
        w, b = self.layers[-1]
        out = x @ w.T + b
        if self.clip is not None:
            out = torch.clamp(out, -self.clip, self.clip)
        return out

    def __call__(self, obs):        # the eval calls policy(obs)
        return self.act(obs)


def as_actor(o):
    """The wrapper hands back a TensorDict with actor/critic groups; a hand-loaded MLP wants the
    actor block only."""
    if isinstance(o, tuple):
        o = o[0]
    if hasattr(o, "keys"):
        for k in ("actor", "policy", "actor_obs"):
            try:
                if k in o:
                    return o[k]
            except Exception:  # noqa: BLE001
                pass
    return o


def main() -> int:
    task = sys.argv[1]
    ckpt = sys.argv[2]
    n_envs = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 0          # 0 = one full episode
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    device = "cuda:0"

    torch.manual_seed(seed)
    env_cfg = load_env_cfg(task, play=False)
    env_cfg.scene.num_envs = n_envs
    # Force a SINGLE spawn bucket (SPAWN_BUCKET=face_down|face_up|sitting|standing) so the recovery
    # metric cannot be contaminated by bucket labels inferred from the first trunk height: with one
    # bucket, "stood up" can only mean a genuine low->high transition.
    _bucket = os.environ.get("SPAWN_BUCKET")
    if _bucket:
        _probs = {"face_down": (1, 0, 0, 0), "face_up": (0, 1, 0, 0),
                  "sitting": (0, 0, 1, 0), "standing": (0, 0, 0, 1)}[_bucket]
        # The spawn probabilities are DRIVEN by a curriculum term (prone_init_prob and friends) that
        # re-writes them at runtime, so overriding the event cfg alone is a no-op (measured: three
        # "forced" buckets returned identical results). Drop those curriculum terms from the cfg
        # before the env is built - that is the clean entry point, no env internals touched.
        for _cname in list(env_cfg.curriculum):
            if "init_prob" in _cname or "spawn" in _cname:
                env_cfg.curriculum.pop(_cname)
                print(f"SPAWN_BUCKET: dropped curriculum {_cname!r}")
        for _name in ("set_ground_state", "set_random_ground_state"):
            _ev = env_cfg.events.get(_name)
            if _ev is not None:
                _ev.params.update(dict(zip(
                    ("face_down_prob", "face_up_prob", "sitting_prob", "standing_prob"), _probs)))
                print(f"SPAWN_BUCKET={_bucket} forced on {_name}")
                break
    if os.environ.get("SPAWN_FLOOR"):
        # A floor pin trips the FALL termination the moment it is applied (the robot IS down), the env
        # recycles the episode, and the measurement silently becomes "what the env's own spawn mix
        # does" - measured on VelStand: fell_over 64/64, final z 75.0 mm, i.e. nothing moved. The
        # StandUp env removes this term for exactly this reason; an evaluation that pins the robot on
        # purpose must do the same. EVALUATION-ONLY: nothing here changes training.
        for _t in ("fell_over",):
            if _t in env_cfg.terminations:
                env_cfg.terminations.pop(_t)
                print(f"SPAWN_FLOOR: dropped termination {_t!r} (it fires on the pin and recycles the episode)")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if steps <= 0:
        steps = int(round(env_cfg.episode_length_s / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)))

    from mjlab.rl import RslRlVecEnvWrapper

    agent_cfg = load_rl_cfg(task)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # HAND-LOADED ACTOR, no rsl_rl runner (2026-09-25). Upstream's new `distill` runner fetches an
    # EXPERT checkpoint through `wandb.Api()` for some tasks (VelStand), so building a runner there
    # raises without an API key and took this evaluator out for that whole family. An evaluation needs
    # the ACTOR only, and the actor carries its own obs normalizer - which is exactly what the ONNX
    # export bakes in, so this path is the deployment arithmetic rather than a reimplementation.
    policy = Actor(ckpt, clip=getattr(agent_cfg, "clip_actions", None), device=device)

    term_names = list(env.termination_manager.active_terms)
    term_counts = dict.fromkeys(term_names, 0)
    returns = torch.zeros(n_envs, device=device)
    lengths = torch.zeros(n_envs, device=device)
    obs = as_actor(wrapped.reset() if hasattr(wrapped, "reset") else env.reset())
    if isinstance(obs, tuple):
        obs = obs[0]

    # Optional FIXED command (FIXED_VX=0.3): hold the twist command instead of sampling, so a
    # policy can be asked "do you walk at this speed AT ALL" separately from "in the rehearsal".
    import os as _os
    fixed_vx = float(_os.environ.get("FIXED_VX", "nan"))

    # SPAWN_FLOOR=prone|supine: write every env's root state to a PURE floor pose AFTER the reset,
    # bypassing the spawn events and the curriculum that drives them. Both of those were already
    # tried and failed (an event-cfg override is a no-op because `prone_init_prob`-style curriculum
    # terms re-write it every interval; dropping the curriculum term crashed the env build). The
    # StandUp env deliberately spawns some envs "partway along the roll" as a built-in reverse
    # curriculum, which is exactly the tail that made the earlier recovery numbers unreadable - this
    # removes it by construction, so "stood up" can only mean a real floor-to-stand transition.
    _floor = _os.environ.get("SPAWN_FLOOR")
    if _floor in ("prone", "supine", "prone_event", "supine_event"):
        import math as _m
        if _floor.endswith("_event"):
            # Replay the env's OWN spawn function for one orientation, mixture forced to 1.0.
            # This is the apples-to-apples twin of the hand-written pin below: the built-in
            # "prone" bucket recovers 14 % while the hand-written pin recovered 0/256, and the
            # only way to tell a real spawn-distribution effect from a bug in my pin is to
            # replay the event itself (same random yaw, same joint handling, same z band).
            import mjlab_microduck.tasks.mdp as _mdp
            _all = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
            _pd = _floor.startswith("prone")
            _mdp.set_random_ground_state(
                env, _all,
                face_down_prob=1.0 if _pd else 0.0,
                face_up_prob=0.0 if _pd else 1.0,
                sitting_prob=0.0,
                standing_prob=0.0,
                prone_z_min=0.05,
                prone_z_max=0.09,
                face_up_roll_max=_m.radians(90.0),
            )
            env.sim.forward()
            _ra = env.scene["robot"].indexing.free_joint_q_adr
            _ra = int(_ra[0]) if hasattr(_ra, "__getitem__") else int(_ra)
            _z = env.sim.data.qpos[:, _ra + 2]
            print(
                f"SPAWN_FLOOR={_floor}: event replayed on all {env.num_envs} envs; trunk z "
                f"min/med/max = {float(_z.min()) * 1000:.0f}/{float(_z.median()) * 1000:.0f}/"
                f"{float(_z.max()) * 1000:.0f} mm"
            )
        else:
            _robot = env.scene["robot"]
            _adr = _robot.indexing.free_joint_q_adr
            _adr = int(_adr[0]) if hasattr(_adr, "__getitem__") else int(_adr)
            _z0 = 0.075 if _floor == "prone" else 0.048
            _pitch = _m.radians(90.0) if _floor == "prone" else _m.radians(-90.0)
            _q = env.sim.data.qpos
            _q[:, _adr + 2] = _z0
            _q[:, _adr + 3:_adr + 7] = torch.tensor(
                [_m.cos(_pitch / 2.0), 0.0, _m.sin(_pitch / 2.0), 0.0], device=_q.device
            )
            try:
                _va = _robot.indexing.free_joint_v_adr
                _va = int(_va[0]) if hasattr(_va, "__getitem__") else int(_va)
                env.sim.data.qvel[:, _va:_va + 6] = 0.0
            except Exception:  # noqa: BLE001
                pass
            env.sim.forward()
            print(f"SPAWN_FLOOR={_floor}: trunk z={_z0 * 1000:.0f} mm, pitch={_m.degrees(_pitch):+.0f} deg")

    # Per-env bookkeeping: do NOT break on the first termination (that made ep_len mean
    # "step at which the FIRST env failed"). Track each env's own termination step and the pose
    # it held immediately before it, then report a distribution -- the only way to tell
    # "consolidated" from "stalled" on a pose task.
    alive = torch.ones(n_envs, dtype=torch.bool, device=device)
    ep_rew = torch.zeros(n_envs, device=device)
    first_stand = torch.zeros(n_envs, dtype=torch.long, device=device)   # 0 = never stood   # per-env total reward, to test whether the
                                                  # failing spawn buckets still collect reward
    term_step = torch.zeros(n_envs, dtype=torch.long, device=device)
    pose_z = torch.full((n_envs,), float("nan"), device=device)
    pose_g = torch.full((n_envs,), float("nan"), device=device)
    # Body-speed / wheel-slip readout (2026-09-22). The wheeled families' stacks contain no
    # body-displacement term and their `Metrics/twist/error_vel_xy` is identically 0 (it measures
    # the LATERAL error and lin_vel_y is pinned to 0), so "did the body actually move?" had no
    # training-side readout at all -- the whole burnout diagnosis rested on the rehearsal's
    # achieved_fwd_med. This accumulates forward body speed and the wheel-implied surface speed so
    # the question is answerable in one eval, on any task that has the four passive wheels.
    _vf_sum = torch.zeros((), device=device)
    _ws_sum = torch.zeros((), device=device)
    _speed_n = 0
    _ws_n = 0
    _wheel_ids = None
    _wheel_missing = False
    # Per-env episode min/max trunk z. The `final_*` fields are the pose at the LAST step, which is
    # useless for a PHASE-DRIVEN task (crouch/ground_pick/spin): all envs start at phase 0
    # (`randomize_phase=False`) and advance in sync, so the final pose is one instant of the cycle.
    # Whether the cycle actually goes DOWN and comes back UP is what has to be measured.
    z_min = torch.full((n_envs,), float("inf"), device=device)
    z_max = torch.full((n_envs,), float("-inf"), device=device)
    # SUSTAINED recovery bookkeeping (the criterion fix). The instantaneous `first_stand` below
    # fires on a SINGLE frame, so a spawn or contact bounce that is momentarily high and upright
    # reads as a recovery. The sustained version requires the pose to hold for hold_steps
    # consecutive frames AND the env to have been genuinely low (<95 mm) BEFORE the window opens,
    # which is exactly what a spawn transient cannot fake. Both numbers are reported side by side
    # so the size of the contamination is visible instead of argued about.
    first_stand_sust = torch.zeros(n_envs, dtype=torch.long, device=device)   # 0 = never
    hold_run = torch.zeros(n_envs, dtype=torch.long, device=device)
    min_z_sofar = torch.full((n_envs,), 1e9, device=device)
    hold_steps = int(_os.environ.get("RECOVERY_HOLD_STEPS", "25"))   # 25 steps = 0.5 s at 50 Hz

    last_pose = {}
    with torch.no_grad():
        for step_i in range(steps):
            # Snapshot the reward manager's PER-ENV episode sums before the step: it zeroes them
            # on reset, so a post-step read returns zeros (the same trap as the post-reset pose).
            _ts = getattr(getattr(env, "reward_manager", None), "_episode_sums", None)
            if isinstance(_ts, dict) and _ts:
                prev_sums = {k: v.clone() for k, v in _ts.items()}
            try:                      # pose BEFORE the step: survives the auto-reset
                robot = env.scene["robot"]
                z_now = robot.data.root_link_pos_w[:, 2] * 1000.0
                g_now = robot.data.projected_gravity_b[:, 2]
                pose_z[alive] = z_now[alive]
                pose_g[alive] = g_now[alive]
                z_min = torch.where(alive, torch.minimum(z_min, z_now), z_min)
                z_max = torch.where(alive, torch.maximum(z_max, z_now), z_max)
                try:      # burnout readout: forward body speed vs wheel-implied surface speed
                    # Body speed is meaningful for EVERY family, so it is accumulated even when the
                    # robot has no passive wheels (the walk models) - only the wheel/slip half needs
                    # them. Getting this wrong made the readout silently None on VelStand.
                    _vf_sum = _vf_sum + torch.nan_to_num(
                        robot.data.root_link_lin_vel_b[:, 0], nan=0.0
                    ).mean()
                    _speed_n += 1
                    if _wheel_ids is None and not _wheel_missing:
                        try:
                            _wheel_ids = [
                                robot.find_joints(_n)[0][0]
                                for _n in (
                                    "passive_LF_?wheel", "passive_LR_?wheel",
                                    "passive_RF_?wheel", "passive_RR_?wheel",
                                )
                            ]
                        except Exception:  # noqa: BLE001
                            _wheel_missing = True      # walk model: no wheels, skip the slip half
                    if _wheel_ids is not None:
                        _om = sum(robot.data.joint_vel[:, _i] for _i in _wheel_ids) / 4.0
                        _ws_sum = _ws_sum + torch.nan_to_num(_om * 0.0175, nan=0.0).mean()
                        _ws_n += 1
                except Exception:  # noqa: BLE001
                    pass
                if step_i == 0:
                    spawn_z = z_now.clone()
                    min_z_sofar = z_now.clone()
                    # Spawn-pose diagnostics: the built-in "prone" bucket recovers 14 % while a
                    # hand-written prone pin at the same height recovered 0/256, so record WHAT
                    # the recovering envs were actually spawned as (tilt + joint vector) --
                    # otherwise the two numbers cannot be reconciled without guessing.
                    spawn_tilt_deg = torch.rad2deg(
                        torch.arccos(torch.clamp(-g_now, -1.0, 1.0))
                    )
                    try:
                        spawn_jq = robot.data.joint_pos.clone()
                    except Exception:  # noqa: BLE001
                        spawn_jq = None
                try:      # time-to-stand: first step where the env is upright AND at height
                    up_now = (z_now >= 105.0) & (g_now <= -0.9) & (first_stand == 0) & alive
                    first_stand[up_now] = step_i + 1
                except Exception:  # noqa: BLE001
                    pass
                try:      # sustained twin: same pose, held hold_steps in a row, after a real low
                    q_now = (z_now >= 105.0) & (g_now <= -0.9) & alive
                    hold_run = torch.where(
                        q_now, hold_run + 1, torch.zeros_like(hold_run)
                    )
                    opens = (
                        (hold_run >= hold_steps)
                        & (first_stand_sust == 0)
                        & (min_z_sofar < 95.0)
                    )
                    # record the step the WINDOW opened (1-based), not the step it completed
                    first_stand_sust[opens] = step_i + 1 - (hold_steps - 1)
                    min_z_sofar = torch.where(
                        (first_stand_sust == 0) & alive,
                        torch.minimum(min_z_sofar, z_now),
                        min_z_sofar,
                    )
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
            if fixed_vx == fixed_vx:            # not NaN
                try:
                    term = env.command_manager.get_term("twist")
                    term.command[:, 0] = fixed_vx
                    term.command[:, 1] = 0.0
                    term.command[:, 2] = 0.0
                except Exception:  # noqa: BLE001
                    pass
            actions = policy(as_actor(obs))
            out = env.step(actions)
            obs, _rew, terminated, truncated, _info = out
            done = (terminated | truncated).float()
            lengths += 1.0
            for name in term_names:
                fired = env.termination_manager.get_term(name)
                term_counts[name] += int(fired.sum().item())
            ep_rew += _rew
            done_b = terminated | truncated
            newly = done_b & alive
            term_step[newly] = step_i + 1
            alive &= ~done_b

    # End-of-episode POSE. For pose/episodic families (standup, sitstand, pick, roulade)
    # survival says nothing: an untrained policy can lie on the floor for the whole episode
    # without ever tripping a termination (found on standup's checkpoint 0). Trunk height and
    # uprightness are the honest "did it reach the pose" signals.
    def _deciles(x):
        x = x[~torch.isnan(x)]
        if x.numel() == 0:
            return {}
        q = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=x.device)
        v = torch.quantile(x, q)
        return {k: round(float(a), 1) for k, a in zip(("p10", "p25", "p50", "p75", "p90"), v)}

    steps_ran = int(term_step[term_step > 0].max().item()) if (term_step > 0).any() else int(lengths.max().item())
    standing = (pose_z >= 105.0) & (pose_g <= -0.9)          # documented thresholds

    # Strict recovery: spawned LOW (trunk < 95 mm) and reached the standing pose later in the SAME
    # episode. Labels come from the measured spawn pose, not from a guessed bucket, and the
    # transition requirement excludes envs that were already up at spawn -- the two contaminations
    # (bucket mislabelling, and post-reset spawn states) are both removed here.
    recovery = {}
    if "spawn_z" in dir():
        low = spawn_z < 95.0
        for lo, hi, name in [(-1e9, 60.0, "spawn<60mm (supine)"), (60.0, 85.0, "spawn 60-85mm (prone)"),
                             (85.0, 95.0, "spawn 85-95mm (sitting)")]:
            m = low & (spawn_z >= lo) & (spawn_z < hi)
            n = int(m.sum().item())
            if n == 0:
                continue
            ts = first_stand[m]
            ok = ts > 0
            ts_s = first_stand_sust[m]
            ok_s = ts_s > 0
            recovery[name] = {
                "n": n,
                "recovery_rate": round(float(ok.float().mean().item()), 3),
                "recovery_rate_sustained": round(float(ok_s.float().mean().item()), 3),
                "sustained_hold_steps": hold_steps,
                "t_stand_p50_steps": int(ts[ok].median().item()) if ok.any() else None,
                "t_stand_sust_p50_steps": int(ts_s[ok_s].median().item()) if ok_s.any() else None,
                "le_2s": round(float(((ts > 0) & (ts <= 100)).float().mean().item()), 3),
                "le_4s": round(float(((ts > 0) & (ts <= 200)).float().mean().item()), 3),
                "le_2s_sustained": round(float(((ts_s > 0) & (ts_s <= 100)).float().mean().item()), 3),
                "le_4s_sustained": round(float(((ts_s > 0) & (ts_s <= 200)).float().mean().item()), 3),
            }
            if "spawn_tilt_deg" in dir():
                if ok.any() and (~ok).any():
                    recovery[name]["spawn_tilt_deg_med_recovered"] = round(
                        float(spawn_tilt_deg[m][ok].median().item()), 1
                    )
                    recovery[name]["spawn_tilt_deg_med_not"] = round(
                        float(spawn_tilt_deg[m][~ok].median().item()), 1
                    )
                else:
                    recovery[name]["spawn_tilt_deg_med"] = round(
                        float(spawn_tilt_deg[m].median().item()), 1
                    )
                if "spawn_jq" in dir() and spawn_jq is not None and ok.any() and (~ok).any():
                    recovery[name]["spawn_joint_l2_rec_vs_not"] = round(
                        float((spawn_jq[m][ok].mean(0) - spawn_jq[m][~ok].mean(0)).norm().item()), 3
                    )
        recovery["ALL low spawns"] = {
            "n": int(low.sum().item()),
            "recovery_rate": round(float((first_stand[low] > 0).float().mean().item()), 3)
            if int(low.sum().item()) else None,
            "recovery_rate_sustained": round(
                float((first_stand_sust[low] > 0).float().mean().item()), 3
            ) if int(low.sum().item()) else None,
            "sustained_hold_steps": hold_steps,
        }
        # ORIENTATION-keyed recovery -- the durable fix. The z-based buckets above silently mix
        # the sitting keyframe (upright, z 50-90 mm) with true floor poses (90 deg tilt): that is
        # how a "14 % prone recovery" survived four reward interventions while being entirely made
        # of envs spawned at 3-8 deg of tilt. Key the buckets by SPAWN TILT, which cannot lie.
        if "spawn_tilt_deg" in dir():
            _ob = {}
            for lo, hi, nm in [
                (60.0, 1e9, "floor (tilt>=60deg)"),
                (25.0, 60.0, "slouch (25-60deg)"),
                (-1e9, 25.0, "upright-low (<25deg)"),
            ]:
                mm = low & (spawn_tilt_deg >= lo) & (spawn_tilt_deg < hi)
                nn = int(mm.sum().item())
                if nn == 0:
                    continue
                _ob[nm] = {
                    "n": nn,
                    "recovery_rate": round(
                        float((first_stand[mm] > 0).float().mean().item()), 3
                    ),
                    "recovery_rate_sustained": round(
                        float((first_stand_sust[mm] > 0).float().mean().item()), 3
                    ),
                }
            recovery["BY ORIENTATION (tilt-keyed)"] = _ob

    # Per-term x per-bucket attribution: read the reward manager's PER-ENV episode sums rather than
    # re-calling the reward functions (calling them again would double-update stateful terms like
    # air-time accumulators). This is the audit that has to answer "which terms still pay a policy
    # that never stands".
    term_sums = prev_sums if "prev_sums" in dir() else None
    _rm = getattr(env, "reward_manager", None)
    for _attr in ("_episode_sums", "episode_sums", "_term_sums"):
        _cand = getattr(_rm, _attr, None)
        if isinstance(_cand, dict) and _cand:
            term_sums = _cand
            break
    bucket_terms = {}
    if term_sums is not None and "spawn_z" in dir():
        for lo, hi, name in [(-1e9, 60.0, "spawn<60mm"), (60.0, 85.0, "spawn 60-85mm"),
                             (85.0, 105.0, "spawn 85-105mm"), (105.0, 1e9, "spawn>=105mm")]:
            m = (spawn_z >= lo) & (spawn_z < hi)
            if int(m.sum().item()) == 0:
                continue
            vals = {}
            for tname, ts in term_sums.items():
                try:
                    vals[tname] = round(float(ts[m].mean().item()) / max(steps_ran, 1), 3)
                except Exception:  # noqa: BLE001
                    pass
            top = dict(sorted(vals.items(), key=lambda kv: -abs(kv[1]))[:8])
            bucket_terms[name] = top
    buckets = {}
    if "spawn_z" in dir():
        edges = [(-1e9, 60.0, "spawn<60mm (supine/low)"), (60.0, 85.0, "spawn 60-85mm (prone)"),
                 (85.0, 105.0, "spawn 85-105mm (sitting)"), (105.0, 1e9, "spawn>=105mm (standing)")]
        for lo, hi, name in edges:
            m = (spawn_z >= lo) & (spawn_z < hi)
            if int(m.sum().item()) > 0:
                ts = first_stand[m]
                stood = ts[ts > 0]
                buckets[name] = {"n": int(m.sum().item()),
                                 "stand_rate": round(float((ts > 0).float().mean().item()), 3),
                                 "t_stand_p50_steps": int(stood.median().item()) if stood.numel() else None,
                                 "stand_le_1s": round(float(((ts > 0) & (ts <= 50)).float().mean().item()), 3),
                                 "stand_le_2s": round(float(((ts > 0) & (ts <= 100)).float().mean().item()), 3),
                                 "stand_le_4s": round(float(((ts > 0) & (ts <= 200)).float().mean().item()), 3),
                                 "standing_frac": round(float(standing[m].float().mean().item()), 3),
                                 "mean_reward": round(float(ep_rew[m].mean().item()), 2)}
    final = {
        "final_trunk_z_mm_median": round(float(pose_z[~torch.isnan(pose_z)].median().item()), 1)
        if (~torch.isnan(pose_z)).any() else None,
        "final_trunk_z_mm_deciles": _deciles(pose_z),
        "final_gravity_z_median": round(float(pose_g[~torch.isnan(pose_g)].median().item()), 3)
        if (~torch.isnan(pose_g)).any() else None,
        "standing_frac": round(float(standing.float().mean().item()), 3),
        "mean_reward": round(float(ep_rew.mean().item()), 2),
        # Phase-driven-task readout: did the cycle go down and come back up?
        "trunk_z_min_mm_median": round(float(z_min[~torch.isinf(z_min)].median().item()), 1)
        if (~torch.isinf(z_min)).any() else None,
        "trunk_z_max_mm_median": round(float(z_max[~torch.isinf(z_max)].median().item()), 1)
        if (~torch.isinf(z_max)).any() else None,
        "trunk_z_span_mm_median": round(
            float((z_max - z_min)[~torch.isinf(z_min)].median().item()), 1
        ) if (~torch.isinf(z_min)).any() else None,
        # Burnout readout: a wheeled policy that spins its wheels while the body stays put shows
        # wheel_surface >> fwd_speed. Only populated on tasks that have the four passive wheels.
        "mean_fwd_speed_mps": round(float(_vf_sum.item()) / _speed_n, 4) if _speed_n else None,        "mean_wheel_surface_speed_mps": round(float(_ws_sum.item()) / _ws_n, 4)
        if _ws_n else None,
        "mean_slip_mps": round(
            (float(_ws_sum.item()) / _ws_n) - (float(_vf_sum.item()) / _speed_n), 4
        ) if (_ws_n and _speed_n) else None,
        "speed_samples": _speed_n,
        "by_spawn_bucket": buckets,
        "by_spawn_bucket_terms": bucket_terms,
        "recovery": recovery,
        "alive_at_end": int(alive.sum().item()),
        "per_env_ep_len_median": int(term_step[term_step > 0].median().item())
        if (term_step > 0).any() else int(lengths.max().item()),
    }

    def _num(v):
        try:
            v = v.mean() if hasattr(v, "mean") else v
            return round(float(v), 4)
        except Exception:  # noqa: BLE001
            return None

    extras = getattr(env, "extras", None) or {}
    raw_log = extras.get("log", {}) if isinstance(extras, dict) else {}
    metrics = {k: _num(v) for k, v in raw_log.items()}
    metrics = {k: v for k, v in metrics.items() if v is not None}
    row = {
        **final,
        "task": task,
        "checkpoint": Path(ckpt).name,
        "envs": n_envs,
        "steps_run": int(lengths.max().item()),
        "episode_len_mean": round(float(lengths.mean().item()), 1),
        "alive_at_end": round(float((lengths >= steps).float().mean().item()), 3),
        "terminations": {k: v for k, v in term_counts.items()},
        "metrics": {k: round(v, 4) for k, v in sorted(metrics.items())},
    }
    env.close()
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
