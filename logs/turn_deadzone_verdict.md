# Turn dead zone: verdict (2026-09-26)

Pre-registration and mid-flight amendment: `logs/turn_deadzone_plan.md`. Raw sweep:
`logs/turn_scratch_results.txt`; per-round files in `logs/turn_eval/`. Diagnostic that redirected the
experiment: `logs/wobble_deadzone.txt`. All numbers below are medians over **3 seeds x 5 rounds**.

## The answer

The sub-0.25 rad/s in-place turn dead zone is **neither a gait limit nor an optimization-history
artifact**: it is the `angular_wobble` cost, which prices the instantaneous roll/pitch rate — exactly
what the scuffing pivot-step a slow in-place turn requires produces. It suppresses the low-command
region so hard that the reward's argmax there is "barely turn", and no amount of *pinning* the command
recovers it while the tax is in place.

In-place achieved |yaw| (median of 15 rounds), commanded in the left column:

| arm | init / recipe | cmd 0.5 | cmd 0.3 |
|---|---|---|---|
| `base` | deployed 68k policy | 0.341 (gain 0.68) | 0.081 (0.27) |
| `turn00_scratch` | from scratch, **no pin** (control) | 0.285 (0.57) | 0.038 (0.13) |
| `turn05_warm` | deployed + WARM_START, **pin 0.5** | 0.393 (0.79) | 0.136 (0.45) |
| `turn03_scratch` | **from scratch**, pin 0.3 | 0.410 (0.82) | 0.173 (0.58) |
| `turn03_woboff` | deployed + WARM_START, pin 0.3, **wobble OFF**, 2k it | **0.471 (0.94)** | **0.256 (0.85)** |
| `ref_wobbleoff` | 2026-09-17 yaw-only recipe, wobble OFF, no pin (reference) | 0.554 (1.11) | 0.309 (1.03) |

Three levers, measured separately:

1. **Removing the wobble tax is the dominant lever.** At cmd 0.3: 0.173 (pin, tax on) -> 0.256 (pin,
   tax off) on comparable recipes, and the fully wobble-free reference reaches **1.03 gain** — i.e. a
   wobble-free recipe can *track* 0.3 rad/s, so the earlier "the gait cannot turn slowly" reading is
   wrong.
2. **The pin helps, but only as a second-order lever.** From scratch it lifts the control 0.038 -> 0.173
   (4.5x) and on the deployed policy 0.081 -> 0.136; on its own it never reaches tracking. Populating a
   command region is necessary but not sufficient when another term taxes the behaviour that region
   requires — which is exactly why step 3 of `optimization_plan.md` (pin only, 3k iters) came back
   negative.
3. **A fresh optimization does not find the exit by itself.** `turn03_scratch` (pin, tax on, 5k iters
   from scratch) lands at 0.173 — better than the control, still 0.58 gain. The control
   (`turn00_scratch`, no pin) is the worst arm in the table at 0.038, i.e. the stock command mix barely
   trains low-rate turning at all.

**Decision rule 0 fired** (`logs/turn_scratch_results.txt`), which was written before the sweep ran as
the amendment's first rule.

## What this changes

* **For the runtime**: the dead zone was never a reason to require |yaw| >= 0.4 rad/s or to use
  bang-bang heading control. A policy trained without the tax tracks 0.3 rad/s at gain ~1.0, so the
  runtime's heading loop can keep commanding small rates.
* **For the recipe**: `MICRODUCK_ANGULAR_WOBBLE=0` is now a candidate change to the *shipping* walking
  recipe, not an experiment. The 2026-09-17 sweep that chose the -1.5 default measured only the
  linear-tracking trade-off (`err_xy` 0.2010 vs 0.2000 vs 0.2068) and the trunk-wobble trade-off; it
  never measured the low-command turn region, which is where the tax is decisive.
* **For the ledger**: step 3 of `optimization_plan.md` and "Still open" in `GOAL_SUMMARY.md` are
  answered; the turn rows of the 15-row rotation are the last thing to re-gate.

## The residual floor

At cmd **0.2** the tax does *not* explain the failure: the matched pair measured 0.022-0.032 (wobble
on) vs 0.008-0.012 (wobble off) — both fail, and the wobble-free arm is if anything worse. Whether the
wobble-free reference reaches 0.2 was not measured (it was outside the sweep's two commands); treat
"cannot hold 0.2 rad/s" as an open, separate question rather than a solved one.

## Next (pre-registered in `logs/turn_ship_plan.md`)

Consolidate the fix into a shippable recipe: continue `turn03_woboff` to 5,000 iterations, and run the
matched **wobble-free, no-pin** arm to test whether the pin is needed at all once the tax is gone.
