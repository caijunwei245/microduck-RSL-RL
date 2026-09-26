"""Aggregate the pre-registered turn sweep into one table and apply the decision rules.

Reads `logs/turn_eval/<arm>_turn<T>_seed<S>.out` (produced by `logs/turn_queue.sh` from
`logs/skill_demo.py --rounds 5 --only turn`) and prints, per arm and commanded rate:

    rounds      how many of the 30 rounds cleared the script's 0.30 rad/s absolute bar
    median      median ACHIEVED |yaw| over those rounds - the number the decision rules use
    gain        median achieved / commanded

`velocity turn` is the in-place row (the dead zone lives here); `walk+turn` is the same yaw rate
commanded while walking, which tracks it well - the contrast is the point, so both are shown.
Decision rules are quoted from `logs/turn_deadzone_plan.md`, written before the runs.
"""

from __future__ import annotations

import glob
import os
import re
import statistics

EVAL_DIR = "logs/turn_eval"
ARMS = [
    ("base", "deployed 68k policy (dc_long_0919_0125)"),
    ("turn05_warm", "deployment ckpt + WARM_START, range 0.5, pin 0.3, 2000 it"),
    ("turn03_scratch", "from scratch, range 0.3, pin 0.3, 5000 it"),
    ("turn00_scratch", "from scratch, CONTROL (no pin), 5000 it"),
    ("turn05_scratch", "from scratch, range 0.5, pin 0.5, 5000 it"),
]
TURN_CMD = ["0.5", "0.3"]
ROW_RE = re.compile(r"^(velocity turn|walk\+turn)\s+\S+\s+(.*)$")
ROUND_RE = re.compile(r"\|yaw\|=([0-9.]+) gain=([0-9.]+) abs0\.3=(\w+)")


def read_rounds(arm: str, turn: str) -> dict[str, list[tuple[float, float, bool]]]:
    out: dict[str, list[tuple[float, float, bool]]] = {"velocity turn": [], "walk+turn": []}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, f"{arm}_turn{turn}_seed*.out"))):
        with open(path, errors="replace") as fh:
            for line in fh:
                m = ROW_RE.match(line.strip())
                if not m:
                    continue
                row = "velocity turn" if m.group(1) == "velocity turn" else "walk+turn"
                # One match per round, so the 0.30 absolute bar is read per round rather than once
                # per line (a line reports all 5 rounds of that seed on one row).
                for yaw, gain, bar in ROUND_RE.findall(m.group(2)):
                    out[row].append((float(yaw), float(gain), bar == "OK"))
    return out


def main() -> int:
    print()
    print("############ DECISION TABLE (median over 3 seeds x 5 rounds) ############")
    print(f"{'arm':16s} {'cmd':>4s} {'row':14s} {'n':>3s} {'median|yaw|':>11s} {'gain':>5s} {'>=0.30':>7s}")
    got: dict[tuple[str, str, str], float] = {}
    for arm, _desc in ARMS:
        for turn in TURN_CMD:
            data = read_rounds(arm, turn)
            for row in ("velocity turn", "walk+turn"):
                vals = data[row]
                if not vals:
                    print(f"{arm:16s} {turn:>4s} {row:14s} {0:>3d} {'-':>11s} {'-':>5s} {'-':>7s}")
                    continue
                yaws = [v[0] for v in vals]
                med = statistics.median(yaws)
                ok = sum(1 for v in vals if v[2])
                got[(arm, turn, row)] = med
                print(f"{arm:16s} {turn:>4s} {row:14s} {len(vals):>3d} {med:>11.3f} "
                      f"{med / float(turn):>5.2f} {ok:>3d}/{len(vals):<3d}")

    print()
    print("############ DECISION RULES (from logs/turn_deadzone_plan.md, written in advance) ############")
    pin = got.get(("turn03_scratch", "0.3", "velocity turn"))
    ctl = got.get(("turn00_scratch", "0.3", "velocity turn"))
    warm = got.get(("turn05_warm", "0.5", "velocity turn"))
    base = got.get(("base", "0.5", "velocity turn"))

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    print(f"in-place @ cmd 0.3 : pin-from-scratch {fmt(pin)}   control {fmt(ctl)}")
    print(f"in-place @ cmd 0.5 : deployed {fmt(base)}   warm+pin {fmt(warm)}")
    if pin is None or ctl is None:
        print("VERDICT: incomplete - some arms produced no turn data (see the per-seed lines above).")
    elif pin >= 0.20 and ctl <= 0.10:
        print("VERDICT: rule 1 - H1 (optimization artifact). Pinning from iteration 0 opens the dead")
        print("         zone; port the pin into the deployed recipe and re-gate the turn rows.")
    elif ctl >= 0.20:
        print("VERDICT: rule 3 - a fresh run suffices; the pin is irrelevant and the deployed 68k")
        print("         policy is the odd one out. The answer is retrain, not pin.")
    elif pin <= 0.10:
        print("VERDICT: rule 2 - H2 survives. This gait cannot hold a small in-place yaw rate even")
        print("         when trained on it from step 0: a DESIGN constraint, not a training bug.")
        print("         The runtime must command >= 0.4 rad/s or use bang-bang heading control.")
    else:
        print(f"VERDICT: partial - the pin moved the rate to {fmt(pin)} (control {fmt(ctl)}), between")
        print("         the rule-1 and rule-2 thresholds; report as a trend, not a fix.")

    print()
    print("Read the walk row of each arm (`logs/turn_eval/<arm>_walk.out`) before trusting any turn")
    print("number: a gait that cannot walk at 0.15 m/s has no meaningful yaw rate either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
