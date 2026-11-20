# GTA VI Trophy Rarity Forecast — make.fun gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ranger-putih/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ranger-putih/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A program that forecasts, for every Grand Theft Auto VI trophy on
PlayStation 5, the percent of Exophase-tracked players who will hold it on
2026-11-26T00:00:00Z — as five quantiles (5%, 25%, 50%, 75%, 95%) per
trophy — from the house's recorded reads of the Exophase trophy page up to
the case checkpoint. Built for the
[make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## Why not persistence

Exophase refreshes its counts on its own schedule, hours apart. On launch
night the page sat unchanged at 638 players tracked, then refreshed in one
jump to 3,868 players — and every rate but the one at 100% fell, because
the players who joined in the meantime hold far fewer trophies. A forecast
that says "rates stay where they are" is biased high after every refresh,
and its narrow spread misses completely (the house baseline scores ~1.51
pinball loss on the practice cases, with 90% coverage ~0). This program
forecasts the dilution instead of ignoring it.

## How it works

`main.py` (Python 3 stdlib only, no network, deterministic) reads one case
file per run from `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`):

1. **Newcomer rates from the jumps.** For every pair of consecutive reads
   whose tracked-player count grew (N2 > N1), the joining players must
   account for the extra holders: `r_new = (N2*r2 - N1*r1) / (N2 - N1)`
   per trophy. The estimates are pooled across all jumps, weighted by the
   number of newcomers and by recency (24 h half-life), and shrunk toward
   the trophy's own current rate.
2. **Tracked players at the target.** `log(players)` is fitted against
   `log(1 + hours since release)` by least squares and read at the target
   hour — sublinear in log time, so it saturates instead of exploding —
   with the extrapolated growth damped by 25% and capped at 25× the last
   count.
3. **Median rate.** Holders are conserved across the dilution:
   `r_target = (N_last * r_old + (N_target - N_last) * r_new) / N_target`,
   where `r_old` is the last rate drifted up slightly for the trophies the
   existing players keep earning (larger for common early trophies, near
   zero for the platinum). A trophy at 100% stays at 100%.
4. **Flat history.** When the history shows no jump at all (the practice
   cases), the program uses a prior calibrated on the one refresh the
   competition rules publish for launch night: players 638 → 3,868
   (×6.06), trophies earned 943 → 4,276. The newcomers earned 3,333
   trophies; the trophy at 100% accounts for one each, leaving 3.19
   percentage points of newcomer rate spread over the other trophies in
   proportion to their current rates.
5. **Spread.** Quantiles are built on the logit scale with a spread
   deliberately wider than the house baseline's (about 1.0 logit units
   when no jump has been seen yet), combining the random-walk term, a
   refresh term, and the observed scatter of the jump-implied newcomer
   rates.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each run takes well under a second per case on one CPU, writes at most
65,536 bytes per output, and two runs on the same case are byte-identical.

## Files

- `main.py` — the forecast program (the entrypoint).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet for this entry.
