# Turn fix: consolidation run (pre-registration, 2026-09-26)

Follow-up to `logs/turn_deadzone_verdict.md`. The mechanism is settled (the `angular_wobble` tax, not
the gait and not the optimization history); the question now is **what the shippable recipe is**.

## Arms (2 GPUs, both from the deployed walking checkpoint `dc_long_0919_0125/model_67997.pt`)

| arm | recipe | iters | why |
|---|---|---|---|
| `turn03_woboff` (continued) | WARM_START + pin 0.3 + `MICRODUCK_ANGULAR_WOBBLE=0`, resumed from its own `model_1999.pt` | 2000 -> 5000 | does more time close the gap to the 1.03-gain reference, or does 2k iterations already saturate? |
| `turn00_ship` | WARM_START + `MICRODUCK_ANGULAR_WOBBLE=0`, **stock command mix, no pin**, new run name | 5000 | is the pin needed at all once the tax is gone? If not, the fix is a one-line recipe change with no new config surface |

Same code, one env-var/knob apart, both warm-started from the same checkpoint with
`MICRODUCK_WARM_START=1` (curricula restart, so neither inherits the fully-hardened schedule). The
third and fourth cells of the 2x2 already exist from the completed sweep: `base` (tax on, no pin) and
`turn05_warm` (tax on, pin).

## Readout (fixed now)

The same `logs/skill_demo.py --rounds 5 --only turn --turn T --seed S`, `T ∈ {0.3, 0.5}`,
`S ∈ {0, 1, 2}`, plus the walk row per arm — **a turning policy that cannot walk is not a candidate**.
Reported as median achieved |yaw| and gain, aggregated by `logs/turn_report.py`.

## Decision rules (written before the data)

1. `turn00_ship` (wobble-free, **no pin**) reaches gain >= 0.80 at cmd 0.3 → **drop the pin**: ship
   "remove the tax, change nothing else", which needs no command-distribution change.
2. `turn00_ship` stays <= 0.60 while the continued pin arm reaches >= 0.85 → **keep both levers**
   (untax + pin); the pin goes into the deployed recipe as a permanent feature.
3. Either arm reaching gain >= 0.90 at **both** 0.3 and 0.5 with the walk row intact → promote that
   checkpoint to the turn rows of the 15-row rotation and re-run `logs/acceptance_gate.sh` on them.
4. Walking regression guard: if the promoted arm's walk row drops below 0.20 m/s at cmd 0.3 (deployed:
   0.252-0.258), the tax removal costs the gait and must be replaced by a *gated* weight
   (`w(|cmd_yaw|)`: full above ~0.25 rad/s, ~0 below) rather than removed globally — that arm is the
   next experiment, not this one.

## Cost

2 x 5,000 iterations at ~1.1 s/iteration, one per GPU -> ~95 min wall clock, then ~30 min of sweep.
