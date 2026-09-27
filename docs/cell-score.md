# The controlled cell score

This note explains how `adjusted_strength` scores a seat in a controlled game against its matched VPAI baseline, why the score is no longer the plain logit difference, and how the default `cell_gain_bend: 0.5` was chosen. Uncontrolled games are not affected. They still use the civilization-level adjustment from the original CivBench (`civ_adjust: "ols_logit"`).

For the config fields, see [configs/benchmark.md](../configs/benchmark.md), section 5.1. The code is `cell_score` in [bench/adjust/strength.py](../bench/adjust/strength.py).

## The problem

In a controlled game, each seat is compared with Vanilla VPAI playing the same seed and seat. Call the seat's late-game win probability `x`, and call VPAI's baseline in that cell `b`: the inverse logit of VPAI's mean `logit_strength` over that cell's replays. The score is 0.5 when the seat plays level with VPAI.

The previous score was the logit difference, `inv_logit(logit x - logit b)`. This compares odds ratios, so the same ratio scores the same in every cell:

| VPAI `b` | Seat `x` | Logit difference |
| --- | --- | --- |
| 1% | 2% | 0.669 |
| 10% | 20% | 0.692 |
| 0.2% | 1% | 0.834 |

Many cells are weak for VPAI. In the baseline experiment, a quarter of VPAI's seats have a win probability under 0.8%, and 9 of the 24 cells have `b` under 2%. In those cells, a model that moves from almost no chance to slightly more than almost no chance is scored as a clear win. The logit curve for `b < 0.5` also rises steeply just above `b` and then flattens, so these small gains get most of the available score. Models that happen to sit in weak cells are inflated.

## The formula

The new score keeps the reference point (b, 0.5) and the end points (0, 0) and (1, 1). It uses two pieces that meet at `b`:

| Side | Formula |
| --- | --- |
| Gain, `x >= b` | `h = (x - b) / (1 - b)`, then `y = 0.5 + 0.5 h / (h + r (1 - h))` |
| Loss, `x < b` | `s = x / b`, then `y = 0.5 s / (s + q (1 - s))` |

The two parameters are:

- `r = (2b)^k` when `b < 0.5`, and `r = 2b` when `b >= 0.5`. Here `k` is `cell_gain_bend`.
- `q = b / (r (1 - b))`, which makes the two pieces meet with the same slope, so the curve has no corner at `b`.

What `k` does when `b < 0.5`:

| `k` | Gain side |
| --- | --- |
| 1 | Today's logit arch. The whole curve equals the logit difference exactly. |
| 0.5 | An arch halfway between, on the log scale, since `r = sqrt(2b)`. This is the default. |
| 0 | A straight line from (b, 0.5) to (1, 1). |

The gain side never sags below the straight line and never rises faster than the logit arch. When `b >= 0.5`, every `k` gives the logit difference, because the logit curve already behaves well there.

Worked cases, with the same `x` and `b` as above plus some losses:

| VPAI `b` | Seat `x` | `k = 1` (previous) | `k = 0.5` (default) | `k = 0` |
| --- | --- | --- | --- | --- |
| 1% | 2% | 0.669 | 0.534 | 0.505 |
| 10% | 20% | 0.692 | 0.609 | 0.556 |
| 0.2% | 1% | 0.834 | 0.557 | 0.504 |
| 5% | 15% | 0.770 | 0.636 | 0.553 |
| 10% | 55% | 0.917 | 0.845 | 0.750 |
| 5% | 2.5% | 0.328 | 0.429 | 0.475 |
| 5% | 0.5% | 0.087 | 0.200 | 0.339 |
| 30% | 15% | 0.292 | 0.322 | 0.350 |
| 63% | 80% | 0.701 | 0.701 | 0.701 |

With `k = 0.5`, doubling a 10% baseline now scores more than doubling a 1% baseline. In weak cells, the smooth join also softens the loss side, so a seat that falls to half of a tiny baseline is not scored as a heavy defeat.

The seat's `x` is `inv_logit(logit_strength)`, which keeps the existing clip of `1e-5`. `cell_logit_advantage` still records the plain logit difference `logit_strength - cell_baseline`.

## Choosing `k`

The default was chosen by sweeping `k` from 0 to 1 in steps of 0.1 on the September 2026 panel. That panel had 5,760 controlled seats in 24 cells, with the explicit baseline experiment and 32 non-VPAI player types. The loss side stayed on the smooth join throughout. Harsher loss sides were tried in an earlier prototype. They pulled VPAI's own mean score well below 0.5 and lowered agreement with outcomes, so they were set aside.

Each `k` was scored on five measures:

| Measure | What it asks | Ideal |
| --- | --- | --- |
| Split-half reliability | Split the games in half at random (300 splits). How well do the player-type rankings from one half match the other? | High |
| Cross-half validity | Do rankings from one half predict actual win rates in the other half? This avoids the shared noise of scoring and winning the same games. | High |
| Continuity | How well does the ranking agree with the original civ-only CivBench adjustment on the same seats? | High |
| Cell tilt | Among the 28 player types that played every cell, how much higher do they score in the weakest third of cells than in the strongest third? | 0 |
| VPAI mean | VPAI's own average score | 0.5 |

Results:

| `k` | Reliability | Validity | Continuity | Cell tilt | VPAI mean |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.318 | 0.349 | 0.813 | 0.167 | 0.481 |
| 0.2 | 0.361 | **0.354** | 0.841 | 0.185 | 0.483 |
| 0.4 | 0.388 | 0.347 | 0.871 | 0.209 | 0.485 |
| **0.5** | 0.396 | 0.341 | **0.883** | 0.222 | 0.487 |
| 0.6 | 0.399 | 0.334 | 0.871 | 0.237 | 0.489 |
| 0.7 | **0.400** | 0.325 | 0.869 | 0.253 | 0.490 |
| 1 (previous) | 0.377 | 0.292 | 0.863 | 0.302 | 0.497 |

For reference, raw P(win) scores 0.399 on reliability, 0.318 on validity, and -0.140 on cell tilt. The civ-only method scores 0.385, 0.317, and 0.011.

The measures pull in different directions:

- **Validity** favors a straighter gain side. It is nearly flat from `k = 0` to `0.5` (0.341 to 0.354) and falls off above that.
- **Reliability** favors more bend, peaking around `k = 0.6` to `0.7`. With a straight gain side, weak cells squeeze every score toward 0.5 and carry little ranking information.
- **Continuity** peaks at `k = 0.5`. In a game bootstrap, the best `k` for continuity was usually between 0.5 and 0.8.

`k = 0.5` is best on continuity, within 0.004 of the best on reliability, and within 0.013 of the best on validity. It is the only value that is best on one measure and that close on the other two. Every `k` from 0.4 to 0.6 beats the previous logit difference on all three, and 0.5 is the simplest point in that range, since `r = sqrt(2b)`.

These results are one panel's evidence, not a proof. The validity differences between nearby `k` are well within bootstrap noise. If the panel changes a lot, rerun the sweep before changing the default.

## Known limit: cell tilt

At every `k`, models still score noticeably higher in VPAI's weakest cells than in its strongest ones. Lower `k` shrinks the tilt (0.302 at `k = 1`, 0.222 at `0.5`, 0.167 at `0`), but no value removes it. The civ-only method has almost none (0.011).

The tilt comes from the baseline, not from the curve. VPAI's own strength varies across cells much more than the models' strength does. The standard deviation of the cell mean logit is 1.88 for VPAI and 1.35 for the models, and the models' cell mean moves only 0.55 per unit of VPAI's. In the starts where VPAI almost never wins, the models in the same seat do relatively better. Any score anchored to VPAI's per-cell result therefore rewards weak cells. Fixing that would mean changing how the baseline is formed, for example by shrinking each cell's baseline toward the civilization effect. `cell_gain_bend` does not address it.

## Settings

`cell_gain_bend` lives in the strength stage's `params`:

```jsonc
"params": {
  "block": "auto",
  "cell_gain_bend": 0.5   // numeric in [0, 1]; 1 = previous logit difference
}
```

Set it to `1` to reproduce reports made before this change. The bootstrap resampling in `ratings.*` and the Relative view of the performance curves use the same setting, so their confidence intervals and curves are scaled like the point estimate.
