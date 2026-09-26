"""Aggregate the turn sweeps into one table and apply the pre-registered decision rules.

Reads every `logs/turn_eval/<arm>_turn<T>_seed<S>.out` (produced by `logs/turn_queue.sh` /
`logs/turn_ship.sh` from `logs/skill_demo.py --rounds 5 --only turn`) and prints, per arm and
commanded rate:

    n           rounds found (3 seeds x 5 rounds = 15 when complete)
    median|yaw| median ACHIEVED |yaw| over those rounds - the number the decision rules use
    gain        median achieved / commanded
    >=0.30      rounds clearing the script's 0.30 rad/s absolute bar

`velocity turn` is the in-place row (the dead zone lives here); `walk+turn` is the same yaw rate
commanded while walking, which tracks it well - the contrast is the point, so both are shown.

Arms are DISCOVERED from the files on disk, so a sweep that adds arms needs no edit here; add a
human-readable label to LABELS when you add one. Decision rules are quoted from
`logs/turn_deadzone_plan.md` and `logs/turn_ship_plan.md`, both written before their runs.
"""

from __future__ import annotations

import glob
import os
import re
import statistics

EVAL_DIR = "logs/turn_eval"
TURN_CMD = ["0.5", "0.3"]

# Display order + labels. Unknown arms are appended alphabetically with a blank label.
LABELS: list[tuple[str, str]] = [
    ("base", "deployed 68k policy (dc_long_0919_0125)"),
    ("turn05_warm", "deployed + WARM_START, pin 0.5, 2000 it"),
    ("turn00_scratch", "from scratch, CONTROL (no pin), 5000 it"),
    ("turn03_scratch", "from scratch, pin 0.3, 5000 it"),
    ("turn03_woboff", "deployed + WARM_START + pin 0.3 + WOBBLE OFF, 2000 it"),
    ("turn03_woboff5k", "same, continued to 5000 it"),
    ("turn00_ship", "deployed + WARM_START + WOBBLE OFF, no pin, 5000 it"),
    ("ref_wobbleoff", "REFERENCE: wobble-free arm of the 2026-09-17 matched pair"),
]
LABEL_OF = dict(LABELS)

ROW_RE = re.compile(r"^(velocity turn|walk\+turn)\s+\S+\s+(.*)$")
ROUND_RE = re.compile(r"\|yaw\|=([0-9.]+) gain=([0-9.]+) abs0\.3=(\w+)")


def discover_arms() -> list[str]:
    found = set()
    for path in glob.glob(os.path.join(EVAL_DIR, "*_turn*_seed*.out")):
        base = os.path.basename(path)
        arm = re.sub(r"_turn[0-9.]+_seed[0-9]+\.out$", "", base)
        if arm != base:
            found.add(arm)
    ordered = [a for a, _ in LABELS if a in found]
    return ordered + sorted(found - set(ordered))


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
    arms = discover_arms()
    if not arms:
        print(f"no sweep files in {EVAL_DIR}/ - nothing to report")
        return 1

    print()
    print("############ DECISION TABLE (median over 3 seeds x 5 rounds) ############")
    print(f"{'arm':17s} {'cmd':>4s} {'row':14s} {'n':>3s} {'median|yaw|':>11s} {'gain':>5s} {'>=0.30':>7s}")
    got: dict[tuple[str, str, str], float] = {}
    for arm in arms:
        for turn in TURN_CMD:
            data = read_rounds(arm, turn)
            for row in ("velocity turn", "walk+turn"):
                vals = data[row]
                if not vals:
                    print(f"{arm:17s} {turn:>4s} {row:14s} {0:>3d} {'-':>11s} {'-':>5s} {'-':>7s}")
                    continue
                med = statistics.median(v[0] for v in vals)
                ok = sum(1 for v in vals if v[2])
                got[(arm, turn, row)] = med
                print(f"{arm:17s} {turn:>4s} {row:14s} {len(vals):>3d} {med:>11.3f} "
                      f"{med / float(turn):>5.2f} {ok:>3d}/{len(vals):<3d}")
    print()
    print("labels:")
    for arm in arms:
        print(f"  {arm:17s} {LABEL_OF.get(arm, '')}")

    def fmt(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.3f}"

    g = lambda a, t, r="velocity turn": got.get((a, t, r))  # noqa: E731
    print()
    print("############ DECISION RULES (written before the runs they judge) ############")
    print("in-place achieved |yaw|:")
    print(f"  cmd 0.3 : deployed {fmt(g('base', '0.3'))}  control {fmt(g('turn00_scratch', '0.3'))}  "
          f"pin-scratch {fmt(g('turn03_scratch', '0.3'))}  pin+woboff {fmt(g('turn03_woboff', '0.3'))}  "
          f"ship(woboff,no pin) {fmt(g('turn00_ship', '0.3'))}  ref {fmt(g('ref_wobbleoff', '0.3'))}")
    print(f"  cmd 0.5 : deployed {fmt(g('base', '0.5'))}  warm+pin {fmt(g('turn05_warm', '0.5'))}  "
          f"pin+woboff {fmt(g('turn03_woboff', '0.5'))}/{fmt(g('turn03_woboff5k', '0.5'))}  "
          f"ship(woboff,no pin) {fmt(g('turn00_ship', '0.5'))}  ref {fmt(g('ref_wobbleoff', '0.5'))}")
    print()

    ship = g("turn00_ship", "0.3")
    ship5 = g("turn00_ship", "0.5")
    pin5k = g("turn03_woboff5k", "0.3") or g("turn03_woboff", "0.3")
    ref = g("ref_wobbleoff", "0.3")

    if ship is not None and ship >= 0.24:
        print("VERDICT: ship rule 1 - the wobble-free arm tracks 0.3 rad/s WITH THE STOCK COMMAND MIX,")
        print("         so the pin is not needed: the fix is 'remove the angular-wobble tax', nothing")
        print("         else. Re-gate the turn rows on that checkpoint.")
    elif ship is not None and ship <= 0.18 and (pin5k or 0) >= 0.255:
        print("VERDICT: ship rule 2 - the pin IS still needed once the tax is gone (untaxed-but-unpinned")
        print("         falls short of the pinned arm): keep BOTH levers in the deployed recipe.")
    elif ship is not None:
        print(f"VERDICT: between the ship rules - untaxed/no-pin {fmt(ship)}, pinned {fmt(pin5k)}.")
        print("         Report as a trend; neither lever alone is sufficient or dominant.")
    else:
        print("VERDICT: the ship arms produced no data (see the per-seed lines above).")

    if ship is not None and ship5 is not None and ship >= 0.27 and ship5 >= 0.45:
        print("         ship rule 3 also met (gain >= 0.9 at both commands): promote the checkpoint to")
        print("         the rotation's turn rows and re-run logs/acceptance_gate.sh on them.")
    if ref is not None:
        print(f"         reference ceiling at cmd 0.3 (fully wobble-free older recipe): {fmt(ref)} -")
        print("         the gap between the ship arm and this is what more iterations could buy.")
    print()
    print("Read the walk row of each arm (`logs/turn_eval/<arm>_walk.out`) before trusting any turn")
    print("number: a gait that cannot walk at 0.15 m/s has no meaningful yaw rate either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
