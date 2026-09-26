"""4096 MicroDucks driven by a trained walk policy, rendered as one 64x64 array.

Physics + policy: mjlab's MuJoCo Warp env, 4096 independent worlds on the GPU.
Rendering: a plain XML-loaded model in warp, fed mjlab's per-world qpos each frame.

Two non-obvious requirements, both found by measurement rather than reading docs:

* mjlab's compiled spec renders as bare floor in warp's rasteriser even though CPU
  mujoco draws it and its mesh/material data is byte-identical to the XML model's.
  Rendering from `scene.xml` instead sidesteps that entirely.
* warp culls geometry far from the origin, and mjlab lays its envs out on a +-16 m
  terrain grid -- so every duck fell outside the renderer's reach. Zeroing each
  world's root xy is visually lossless (each world is its own scene with an infinite
  floor) and puts every duck back in frame.

  uv run python scripts/render_4096_mosaic.py --probe          # one frame -> PNG
  uv run python scripts/render_4096_mosaic.py --frames 150     # -> mp4
"""

import argparse
import os
import subprocess
import time
from dataclasses import asdict

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjwarp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"
CKPT = "logs/rsl_rl/velocity/2026-09-23_19-49-11_walk_sustained/model_69996.pt"
RENDER_XML = "src/mjlab_microduck/robot/microduck/scene.xml"

# Third-person camera, fixed in WORLD axes relative to the duck. World-fixed (not
# duck-fixed) so all 4096 share one viewpoint and the array reads as a grid.
CAM_OFFSET = np.array([0.45, -0.45, 0.25])


def look_at(cam_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation whose columns are the camera axes, looking from cam_pos to target.

    MuJoCo cameras look down local -Z, so +Z points from target back to camera.
    """
    z = cam_pos - target
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    n = np.linalg.norm(x)
    x = np.array([1.0, 0.0, 0.0]) if n < 1e-6 else x / n
    return np.stack([x, np.cross(z, x), z], axis=1)


class ArrayRenderer:
    """Renders mjlab's 4096 worlds through a separate, XML-loaded warp model."""

    def __init__(self, sim, nworld: int, tile: int, offset: np.ndarray = CAM_OFFSET):
        self.sim = sim
        self.nworld = nworld
        self.offset = offset
        rmjm = mujoco.MjModel.from_xml_path(RENDER_XML)
        self.trunk = next(
            i for i in range(rmjm.nbody)
            if (mujoco.mj_id2name(rmjm, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")
        )
        self.wm = mjwarp.put_model(rmjm)
        self.wd = mjwarp.put_data(rmjm, mujoco.MjData(rmjm), nworld=nworld)
        self.rc = mjwarp.create_render_context(
            rmjm, nworld=nworld, cam_res=(tile, tile), render_rgb=True,
            render_depth=False, render_seg=False, use_textures=True, use_shadows=False,
        )
        self.out = wp.zeros((nworld, tile, tile), dtype=wp.vec3)

    def render(self) -> np.ndarray:
        q = self.sim.wp_data.qpos.numpy().copy()
        q[:, 0:2] = 0.0                      # see module docstring: warp culls far geometry
        wp.copy(self.wd.qpos, wp.array(q, dtype=self.wd.qpos.dtype, device=self.wd.qpos.device))
        mjwarp.forward(self.wm, self.wd)
        wp.synchronize()
        p = self.wd.xpos.numpy()[:, self.trunk, :].astype(np.float64)
        pos = np.zeros((self.nworld, 1, 3), np.float32)
        rot = np.zeros((self.nworld, 1, 3, 3), np.float32)
        for i in range(self.nworld):
            cam_pos = p[i] + self.offset
            pos[i, 0] = cam_pos
            rot[i, 0] = look_at(cam_pos, p[i])
        wp.copy(self.wd.cam_xpos, wp.array(pos, dtype=wp.vec3, device=self.wd.cam_xpos.device))
        wp.copy(self.wd.cam_xmat, wp.array(rot, dtype=wp.mat33, device=self.wd.cam_xmat.device))
        wp.synchronize()
        mjwarp.render(self.wm, self.wd, self.rc)
        mjwarp.get_rgb(self.rc, 0, self.out)
        wp.synchronize()
        a = self.out.numpy()
        return a if a.dtype == np.uint8 else np.clip(a * 255.0, 0, 255).astype(np.uint8)


def make_mosaic(tiles: np.ndarray, grid: int) -> np.ndarray:
    n, h, w, c = tiles.shape
    assert n == grid * grid, (n, grid)
    return tiles.reshape(grid, grid, h, w, c).transpose(0, 2, 1, 3, 4).reshape(grid * h, grid * w, c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--every", type=int, default=2, help="sim steps per written frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--out", type=str, default="renders/microduck_4096_mosaic.mp4")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    grid = int(round(args.envs ** 0.5))
    assert grid * grid == args.envs, f"{args.envs} is not a perfect square"

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = args.envs
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device="cuda:0")
    runner.load(CKPT, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
    policy = runner.get_inference_policy(device="cuda:0")
    sim = env.unwrapped.sim
    rnd = ArrayRenderer(sim, args.envs, args.tile)
    print(f"[setup] {args.envs} worlds, tile={args.tile}, mosaic={grid*args.tile}x{grid*args.tile}", flush=True)

    obs, _ = env.reset()
    for _ in range(10):                       # settle so frame 0 is a walking duck
        with torch.inference_mode():
            obs, _r, _d, _e = env.step(policy(obs))

    if args.probe:
        t0 = time.perf_counter()
        tiles = rnd.render()
        mos = make_mosaic(tiles, grid)
        import imageio.v2 as iio
        iio.imwrite("renders/probe_mosaic_full.png", mos)
        iio.imwrite("renders/probe_tile0.png", tiles[0])
        k = max(1, mos.shape[0] // 1600)
        iio.imwrite("renders/probe_mosaic_small.png", np.ascontiguousarray(mos[::k, ::k]))
        print(f"[probe] one mosaic frame in {time.perf_counter()-t0:.2f}s", flush=True)
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    W = grid * args.tile
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{W}", "-r", str(args.fps), "-i", "-",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
        stdin=subprocess.PIPE,
    )

    t0 = time.perf_counter()
    step_dt = env.unwrapped.step_dt
    for i in range(args.frames):
        for _ in range(args.every):
            with torch.inference_mode():
                obs, _r, _d, _e = env.step(policy(obs))
        mos = make_mosaic(rnd.render(), grid)
        ff.stdin.write(mos.tobytes())
        if i == 0 or (i + 1) % 20 == 0:
            el = time.perf_counter() - t0
            print(f"[frame] {i+1}/{args.frames}  {(i+1)/el:.2f} frame/s  "
                  f"(sim {(i+1)*args.every*step_dt/el:.2f}x realtime)", flush=True)
    ff.stdin.close()
    ff.wait()
    dt = time.perf_counter() - t0
    print(f"[done] {args.frames} frames in {dt:.1f}s -> {args.out}", flush=True)
    print(f"[done] {W}x{W} mosaic, {args.frames/dt:.2f} frame/s, "
          f"peak vram {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
