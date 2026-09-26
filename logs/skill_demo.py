"""Demonstrate every skill in simulation: 5 rounds each, one env, explicit reset per round.

`family_eval.py` answers aggregate questions (distributions over 256 envs); this answers the
question a human asks next: "show me each skill working, five times". Each round is a fresh episode
with a recorded trace, and each skill has an explicit SUCCESS CRITERION so the output is a verdict
per round rather than a number to interpret:

    kind            criterion
    walk            mean forward speed >= 0.15 m/s at cmd 0.3 AND ends upright (g <= -0.9)
    ball            the ball is displaced forward by >= 0.05 m at some point
    cycle_low       trunk dips below `low_mm` AND returns to >= 105 mm (sit / crouch-and-return)
    crouch_cycle    trunk span >= 40 mm (a commanded crouch cycle happened at all)
    roller_stand    ends at >= 130 mm of trunk on the wheels (the 138 mm roller stand) and never fell
    floor_flip      from a FORCED pure floor spawn, holds trunk >= 105 mm AND g <= -0.9 for >= 0.5 s
    spin            mean |yaw rate| >= 0.30 rad/s while staying upright
    roulade         after the roll, ends upright (>= 105 mm and g <= -0.9)

Usage:  uv run python logs/skill_demo.py [--rounds 5] [--only substring]
"""

from __future__ import annotations

import argparse
import glob
import os as _os_main
import math
import os
from dataclasses import asdict

import torch

import torch.nn.functional as F

import mjlab_microduck.tasks  # noqa: F401  (registrations + patches)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

DEVICE = "cuda:0"
HOLD = 25          # 0.5 s at 50 Hz, matching the sustained recovery criterion
TURN_CMD = 0.5     # rad/s, overridden by --turn: the "requirement" the turn rows are judged against

# (name, task, checkpoint glob, kind, cmd_x, cmd_yaw, force_floor, low_mm)
SKILLS = [
    ("velocity walk",      "Mjlab-Velocity-Flat-MicroDuck",        "logs/rsl_rl/velocity/*dc_long_0919_0125/model_*.pt",         "walk",         0.3,  0.0,  None,   None),
    # "velocity turn" and "walk+turn" are appended in main() so --turn can set their target
    ("ball_kick right",    "Mjlab-BallKick-Flat-MicroDuck",        "logs/rsl_rl/ball_kick_right/*kick_r4/model_*.pt",            "ball",         0.0,  0.0,  None,   None),
    ("sitstand",           "Mjlab-SitStand-Flat-MicroDuck",        "logs/rsl_rl/microduck_sitstand/*/model_*.pt",                "cycle_low",    0.0,  0.0,  None,   75.0),
    ("ground_pick",        "Mjlab-GroundPick-Flat-MicroDuck",      "logs/rsl_rl/ground_pick/*/model_*.pt",                       "cycle_low",    0.0,  0.0,  None,   90.0),
    ("rollers (fast)",     "Mjlab-Velocity-Flat-MicroDuck-Rollers","logs/rsl_rl/velocity_rollers/*rollers_noskate/model_*.pt",   "walk",         0.3,  0.0,  None,   None),
    ("swizzle",            "Mjlab-Velocity-Swizzle-MicroDuck",     "logs/rsl_rl/velocity_swizzle/*swizzle_rolling18/model_*.pt", "walk",         0.3,  0.0,  None,   None),
    ("roller_slope",       "Mjlab-RollerSlope-Flat-MicroDuck",     "logs/rsl_rl/roller_slope/*/model_*.pt",                      "slope_descent",0.0,  0.0,  None,   None),
    ("roller_standup",     "Mjlab-RollerStandUp-Flat-MicroDuck",   "logs/rsl_rl/roller_standup/*roller_standup_stall/model_*.pt",     "roller_stand", 0.0,  0.0,  None,   None),
    ("roller_crouch",      "Mjlab-RollerCrouch-Flat-MicroDuck",    "logs/rsl_rl/roller_crouch/*/model_*.pt",                     "crouch_cycle", 0.0,  0.0,  None,   None),
    ("standup floor flip", "Mjlab-StandUp-Flat-MicroDuck",         "logs/rsl_rl/microduck_stand/*standup_tiltstall/model_*.pt",  "floor_flip",   0.0,  0.0, "prone", None),
    # velstand floor flip: RE-ENABLED 2026-09-25 after both blockers were diagnosed and fixed (the
    # runner dependency, and the pin tripping `fell_over` - see the drop above). Corrected rate: the
    # policy flips from a pure prone pin in **81 %** of envs (52/64, sustained), not the 99 % the
    # contaminated measurement reported.
    ("velstand floor flip","Mjlab-VelStand-Flat-MicroDuck",        "logs/rsl_rl/velstand/*velstand_fromstandup/model_*.pt",      "floor_flip",   0.0,  0.0, "prone", None),
    ("spin",               "Mjlab-Spin-Flat-MicroDuck",            "logs/rsl_rl/spin/*_spin_warmstand/model_*.pt",                              "spin",         0.0,  0.0,  None,   None),
    ("roulade",            "Mjlab-Roulade-Flat-MicroDuck",         "logs/rsl_rl/microduck_roulade/*roulade_finish/model_*.pt",   "roulade",      0.0,  0.0,  None,   None),
]


def _annotate(img, text: str):
    """Burn a label into the frame so the video shows WHICH skill and WHICH round is on screen."""
    if not text:
        return img
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 8 * len(text) + 8, 20], fill=(0, 0, 0))
        dr.text((4, 4), text, fill=(255, 255, 0))
        return __import__("numpy").asarray(im)
    except Exception:  # noqa: BLE001
        return img


def newest(pattern: str) -> str | None:
    hits = glob.glob(pattern)
    if not hits:
        return None
    def it(p):
        try:
            return int(p.rsplit("model_", 1)[1].split(".")[0])
        except Exception:  # noqa: BLE001
            return -1
    return max(hits, key=it)


def force_floor(env, which: str) -> None:
    """Write the root state of a PURE floor pose (same pin as family_eval's SPAWN_FLOOR)."""
    robot = env.scene["robot"]
    adr = robot.indexing.free_joint_q_adr
    adr = int(adr[0]) if hasattr(adr, "__getitem__") else int(adr)
    z0 = 0.075 if which == "prone" else 0.048
    pitch = math.radians(90.0 if which == "prone" else -90.0)
    q = env.sim.data.qpos
    q[:, adr + 2] = z0
    q[:, adr + 3:adr + 7] = torch.tensor(
        [math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0], device=q.device
    )
    try:
        va = robot.indexing.free_joint_v_adr
        va = int(va[0]) if hasattr(va, "__getitem__") else int(va)
        env.sim.data.qvel[:, va:va + 6] = 0.0
    except Exception:  # noqa: BLE001
        pass
    env.sim.forward()


def run_round(env, wrapped, policy, spec, max_steps: int, frames=None,
              stride: int = 2, label: str = "", log_cmd: bool = False) -> dict:
    name, task, _glob, kind, cmd_x, cmd_yaw, floor, _low = spec
    obs = wrapped.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    if floor:
        force_floor(env, floor)

    robot = env.scene["robot"]
    ball = None
    try:
        ball = env.scene["ball"]
    except Exception:  # noqa: BLE001
        ball = None
    ball_x0 = float(ball.data.root_link_pos_w[0, 0]) if ball is not None else None
    ball_max = 0.0

    zs, gs, vs, ys = [], [], [], []
    cmd_trace = []
    fell = False
    steps = 0
    sums = {}
    upright_steps = 0
    with torch.no_grad():
        for _ in range(max_steps):
            if log_cmd:
                try:
                    cmd_trace.append(tuple(float(v) for v in env.command_manager.get_term("twist").command[0]))
                except Exception:  # noqa: BLE001
                    pass
            if kind == "floor_flip":
                # A recovery policy is driven with a ZERO command (stand still) - that is the
                # deployment scenario and what the rehearsal does. Without this the row silently
                # inherited the task's own command sampling: VelStand is a WALKING task, so the
                # policy was told to walk forward while lying on the floor and never got up
                # (measured: hold=0, z=75 mm, while the same checkpoint scores 127/128 in the
                # family evaluation, which does not command anything either).
                try:
                    term = env.command_manager.get_term("twist")
                    term.command[:, :3] = 0.0
                except Exception:  # noqa: BLE001
                    pass
            if cmd_x or cmd_yaw:
                try:
                    term = env.command_manager.get_term("twist")
                    term.command[:, 0] = cmd_x
                    term.command[:, 1] = 0.0
                    term.command[:, 2] = cmd_yaw
                except Exception:  # noqa: BLE001
                    pass
            z = float(robot.data.root_link_pos_w[0, 2]) * 1000.0
            g = float(robot.data.projected_gravity_b[0, 2])
            v = float(robot.data.root_link_lin_vel_b[0, 0])
            y = float(robot.data.root_link_ang_vel_b[0, 2]) if hasattr(robot.data, "root_link_ang_vel_b") else 0.0
            zs.append(z); gs.append(g); vs.append(v); ys.append(y)
            # pose-based fall flag: independent of which termination terms a task happens to keep
            # (`fell_over` is curriculum-disabled in several tasks, and querying it raises KeyError)
            if z < 90.0 and g > -0.5:
                fell = True
            if g <= -0.9:
                upright_steps += 1
            _es = getattr(getattr(env, "reward_manager", None), "_episode_sums", None)
            if isinstance(_es, dict) and _es:
                sums = {k: float(v[0]) for k, v in _es.items()}
            if ball is not None:
                ball_max = max(ball_max, float(ball.data.root_link_pos_w[0, 0]) - ball_x0)
            out = env.step(policy(as_actor(obs)))
            obs, _r, term_b, trunc_b, _i = out
            steps += 1
            if frames is not None and (steps % stride == 0):
                try:
                    img = env.render()
                except Exception:  # noqa: BLE001
                    img = None
                if img is not None:
                    frames.append(_annotate(img, label))
            if "fell_over" in env.termination_manager.active_terms:
                if bool(env.termination_manager.get_term("fell_over")[0]):
                    fell = True
            if bool((term_b | trunc_b)[0]):
                break

    # hold run of (z >= 105 and g <= -0.9)
    hold = run = 0
    for z, g in zip(zs, gs):
        run = run + 1 if (z >= 105.0 and g <= -0.9) else 0
        hold = max(hold, run)
    # longest run at/above a threshold (used by the stand-up tasks, whose episode may end mid-rise)
    def longest_run(pred):
        best = run = 0
        for ok_i in pred:
            run = run + 1 if ok_i else 0
            best = max(best, run)
        return best

    tail = slice(max(1, len(zs) * 2 // 3), None)          # steady-state window
    v_ss = sum(vs[tail]) / max(1, len(vs[tail]))
    yaw_ss = sum(abs(y) for y in ys[tail]) / max(1, len(ys[tail]))
    res = dict(z0=zs[0], z_min=min(zs), z_max=max(zs), z_last=zs[-1], g_last=gs[-1],
               v_mean=v_ss, yaw_abs_mean=yaw_ss, upright_frac=upright_steps / max(1, len(zs)),
               descent=zs[0] - min(zs), held_high=longest_run([z >= 130.0 for z in zs]),
               hold=hold, steps=steps, fell=fell, ball=ball_max, sums=sums)
    if cmd_trace:
        res["cmd0_min"] = min(c[0] for c in cmd_trace)
        res["cmd0_max"] = max(c[0] for c in cmd_trace)
        res["cmd1_min"] = min(c[1] for c in cmd_trace)
        res["cmd1_max"] = max(c[1] for c in cmd_trace)
    zf, gf = zs[-1], gs[-1]
    if kind == "walk":
        ok = res["v_mean"] >= 0.15 and gf <= -0.9 and not fell
    elif kind == "turn":
        # judge against the COMMANDED target (the requirement), and separately against the absolute
        # 0.3 rad/s bar the user asked about - the two answer different questions.
        ok = res["yaw_abs_mean"] >= TURN_CMD and not fell
        res["gain"] = res["yaw_abs_mean"] / TURN_CMD
        res["ok_abs03"] = res["yaw_abs_mean"] >= 0.30 and not fell
    elif kind == "ball":
        ok = ball_max >= 0.05
    elif kind == "cycle_low":
        # PERIODIC task: the episode may end mid-cycle, so the skill is "both extremes reached",
        # not "ends high" (a 5 s phase period against a 6 s episode guarantees an arbitrary end).
        ok = res["z_min"] <= spec[7] and res["z_max"] >= 105.0
    elif kind == "crouch_cycle":
        ok = (res["z_max"] - res["z_min"]) >= 40.0 and not fell
    elif kind == "roller_stand":
        # reach-and-hold the roller stand HEIGHT (a forward-leaning posture: 138 mm of trunk at a
        # tilt that would fail an uprightness gate), and never fall. Rise can take most of the episode.
        # NO fall gate here: tasks that SPAWN low (prone start) satisfy the pose-based fall test on
        # step 0, so gating on it failed rounds that rose to 133 mm and held for 285 steps.
        ok = res["held_high"] >= 10
    elif kind == "slope_descent":
        # the slope task is gravity-driven on a DESCENDING ramp, so absolute z is meaningless
        # (measured -95 mm below the origin): the skill is "roll down while staying on the wheels"
        ok = res["descent"] >= 100.0 and res["upright_frac"] >= 0.8
    elif kind == "floor_flip":
        ok = hold >= HOLD
    elif kind == "spin":
        ok = res["yaw_abs_mean"] >= 0.30 and gf <= -0.9      # spin AND stay on your feet
    elif kind == "roulade":
        # `reward_manager._episode_sums` is keyed by BARE term name (no "Episode_Reward/" prefix -
        # that prefix only exists in mjlab's metrics dict). Getting this wrong read 0.0 and failed
        # every round of a policy that does end upright.
        def _sum(name):
            return res["sums"].get(name, res["sums"].get(f"Episode_Reward/{name}", 0.0))
        rolled = _sum("roulade_progress") > 0.0
        ok = zf >= 105.0 and gf <= -0.9 and rolled
        res["roll"] = _sum("roulade_progress")
    else:
        ok = False
    res["ok"] = ok
    res["ok"] = ok
    res["sum_keys"] = sorted(sums)[:6]
    return res


def detail(kind: str, r: dict) -> str:
    if kind == "ball":
        return f"ball+{r['ball']:.3f} m"
    if kind == "walk":
        return f"v={r['v_mean']:+.3f} m/s z={r['z_last']:.0f} g={r['g_last']:+.2f}"
    if kind == "turn":
        # z/g/`fell` are here because a turn round can miss the criterion two very different ways -
        # "stood still" (upright, no rotation) or "fell over" - and the two need opposite fixes.
        return (f"|yaw|={r['yaw_abs_mean']:.3f} gain={r.get('gain', 0):.2f} "
                f"abs0.3={'OK' if r.get('ok_abs03') else 'XX'} "
                f"z={r['z_last']:.0f} g={r['g_last']:+.2f}{' FELL' if r.get('fell') else ''}")
    if kind == "spin":
        return f"|yaw|={r['yaw_abs_mean']:.2f} rad/s g={r['g_last']:+.2f}"
    if kind == "floor_flip":
        return f"hold={r['hold']:2d} z={r['z_last']:.0f} g={r['g_last']:+.2f}"
    if kind == "slope_descent":
        return f"desc={r['descent']:.0f} mm upright={r['upright_frac']:.2f}"
    if kind == "roller_stand":
        return f"held_high={r['held_high']:3d} z_max={r['z_max']:.0f} upright={r['upright_frac']:.2f}"
    if kind == "cycle_low" and r.get("cmd0_min") is not None:
        return (f"z {r['z_min']:.0f}->{r['z_last']:.0f} span={r['z_max']-r['z_min']:.0f} "
                f"cmd cos[{r['cmd0_min']:+.2f},{r['cmd0_max']:+.2f}] "
                f"sin[{r['cmd1_min']:+.2f},{r['cmd1_max']:+.2f}]")
    if kind == "roulade":
        return (f"z {r['z_min']:.0f}->{r['z_last']:.0f} g={r['g_last']:+.2f} "
                f"roll={r.get('roll', 0.0):.2f}")
    return f"z {r['z_min']:.0f}->{r['z_last']:.0f} span={r['z_max']-r['z_min']:.0f} g={r['g_last']:+.2f}"


def as_actor(o):
    """rsl_rl's wrapper hands back a TensorDict with actor/critic groups; a hand-loaded MLP wants the
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


class Actor:
    """Hand-loaded policy: its own mlp weights AND its own obs normalizer.

    Replaces rsl_rl's runner for the demonstration, for three reasons that all bit us in practice:
      * upstream's new `distill` runner fetches an EXPERT checkpoint through `wandb.Api()` for some
        tasks (VelStand), so building that runner needs a wandb API key the demo should not require;
      * the critic's privileged-obs width differs between families, so a cross-family checkpoint
        cannot be loaded by a task-configured runner;
      * an evaluation needs the ACTOR only, and its normalizer is part of it (it is what the ONNX
        export bakes in) - loading the two by hand is exactly the arithmetic that ships.
    """

    def __init__(self, ckpt: str, clip: float | None = None):
        d = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.mean = d["obs_normalizer._mean"].to(DEVICE)
        self.std = d["obs_normalizer._std"].to(DEVICE)
        self.layers = []
        i = 0
        while f"mlp.{i}.weight" in d:
            self.layers.append((d[f"mlp.{i}.weight"].to(DEVICE), d[f"mlp.{i}.bias"].to(DEVICE)))
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

    def __call__(self, obs):        # the demo calls policy(obs)
        return self.act(obs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--turn", type=float, default=0.5,
                    help="commanded yaw rate for the turn rows (rad/s)")
    ap.add_argument("--video", action="store_true",
                    help="record one video per skill (all rounds back to back, offscreen renderer)")
    ap.add_argument("--play-cfg", action="store_true",
                    help="load the PLAY cfg (domain randomization off) - a DIAGNOSTIC: if the failing "
                         "rounds recover without DR, the residual is a DR-tail robustness gap, not a "
                         "policy-shape problem. Never used for the numbers quoted in the docs.")
    ap.add_argument("--video-dir", default="logs/demo_videos")
    ap.add_argument("--video-fps", type=int, default=25)
    ap.add_argument("--video-stride", type=int, default=2,
                    help="record every Nth control step (2 => 25 fps from a 50 Hz policy)")
    ap.add_argument("--video-scale", type=int, default=1)
    ap.add_argument("--cam-distance", type=float, default=1.3)
    ap.add_argument("--log-cmd", action="store_true",
                    help="record the twist command each step and report its range in the detail "
                         "line - answers 'did the phase ever command the sit?' for sitstand")
    ap.add_argument("--seed", type=int, default=0,
                    help="env seed; sweep it to separate spawn sensitivity from seed noise (A3)")
    ap.add_argument("--ckpt", default=None,
                    help="gate a SPECIFIC checkpoint instead of the newest match of each row's glob")
    ap.add_argument("--turn-ckpt", default=os.environ.get(
                        "MICRODUCK_TURN_CKPT", "logs/rsl_rl/velocity/*dc_long_0919_0125/model_*.pt"),
                    help="checkpoint glob for the two TURN rows. Default = the deployed walking "
                         "candidate; point it at the wobble-free turn policy to show the fixed turn "
                         "without touching the other rows.")
    args = ap.parse_args()

    global TURN_CMD
    TURN_CMD = args.turn
    torch.manual_seed(args.seed)
    SKILLS.insert(1, ("velocity turn", "Mjlab-Velocity-Flat-MicroDuck",
                      args.turn_ckpt, "turn", 0.0, TURN_CMD, None, None))
    SKILLS.insert(2, ("walk+turn", "Mjlab-Velocity-Flat-MicroDuck",
                      args.turn_ckpt, "turn", 0.3, TURN_CMD, None, None))

    print(f"{'skill':22s} {'task env':34s} rounds")
    print("-" * 110)
    summary = []
    for spec in SKILLS:
        name, task, pattern, kind = spec[0], spec[1], spec[2], spec[3]
        if args.only and args.only.lower() not in name.lower():
            continue
        ckpt = args.ckpt if args.ckpt else newest(pattern)
        if ckpt is None:
            print(f"{name:22s} no checkpoint for {pattern}")
            summary.append((name, None, 0, args.rounds))
            continue
        torch.manual_seed(args.seed)
        env_cfg = load_env_cfg(task, play=args.play_cfg)
        env_cfg.scene.num_envs = 1
        if args.video:
            env_cfg.viewer.width = 640 * args.video_scale
            env_cfg.viewer.height = 480 * args.video_scale
            # a 25 cm robot at the default 5 m is a speck: frame it for a human viewer
            env_cfg.viewer.distance = args.cam_distance
            env_cfg.viewer.elevation = -20.0
            env_cfg.viewer.azimuth = 120.0
        if spec[6]:        # a floor pin is requested for this row
            # Same trap family_eval hit (2026-09-25): the robot IS on the floor, so a fall termination
            # fires the moment the pin is applied, the env recycles the episode, and the round quietly
            # measures the env's own spawn mix. StandUp removes this term by design; an evaluation that
            # pins on purpose must do the same. EVALUATION-ONLY.
            for _t in ("fell_over",):
                if _t in env_cfg.terminations:
                    env_cfg.terminations.pop(_t)
                    print(f"  {spec[0]}: dropped termination {_t!r} (fires on the pin)")
        env = ManagerBasedRlEnv(
            cfg=env_cfg, device=DEVICE,
            render_mode="rgb_array" if args.video else None,
        )
        max_steps = int(round(env_cfg.episode_length_s / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)))
        from mjlab.rl import RslRlVecEnvWrapper

        agent_cfg = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        policy = Actor(ckpt, clip=getattr(agent_cfg, "clip_actions", None))

        marks, lines = [], []
        frames = [] if args.video else None
        for r_i in range(args.rounds):
            res = run_round(env, wrapped, policy, spec, max_steps, frames=frames,
                            stride=args.video_stride, log_cmd=args.log_cmd,
                            label=f"{name}  round {r_i + 1}/{args.rounds}")
            marks.append("OK " if res["ok"] else "XX ")
            lines.append(f"{r_i + 1}:{marks[-1]}{detail(kind, res)}")
        if args.video and frames:
            import imageio.v2 as _iio
            _os_main.makedirs(args.video_dir, exist_ok=True)
            vpath = _os_main.path.join(
                args.video_dir, name.replace(" ", "_").replace("+", "_") + ".mp4")
            _iio.mimsave(vpath, frames, fps=args.video_fps, macro_block_size=None)
            print(f"{'':22s} video -> {vpath} ({len(frames)} frames)")
        n_ok = sum(1 for m in marks if m.strip() == "OK")
        summary.append((name, ckpt.split("/")[-2][20:], n_ok, args.rounds))
        print(f"{name:22s} {ckpt.split('/')[-2][20:]:34s} " + " | ".join(lines))
        del env
    print("-" * 110)
    for name, run, n_ok, total in summary:
        tag = "ALL PASS" if n_ok == total else ("FAIL" if n_ok == 0 else "PARTIAL")
        print(f"{name:22s} {str(n_ok) + '/' + str(total):>5s}  {tag}   ({run})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
