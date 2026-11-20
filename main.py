"""Forecast GTA VI trophy earn rates on Exophase at the case target.

Entry for make.fun gta6-trophy-rarity-forecast-01. Python 3 stdlib only,
no network, deterministic: the same case always yields the same bytes.

The forecast problem
--------------------
Exophase prints, for each trophy, the percent of the players it tracks who
hold that trophy. It refreshes those counts on its own schedule, hours
apart, not continuously. On launch night the page sat unchanged (638
players tracked, every rate fixed) and then refreshed in one jump to 3,868
players, with every rate but the one at 100% falling: the players who
joined in the meantime hold far fewer trophies, so each refresh *dilutes*
the rates. A persistence forecast ("rates stay where they are") is
therefore biased high after every refresh, and its narrow spread misses
completely. This program forecasts the dilution instead of ignoring it.

Model (three pieces, all estimated from the case file itself)
------------------------------------------------------------
1. Newcomer rates from the jumps. For every pair of consecutive reads
   whose tracked-player count grew (N2 > N1), the players who joined
   between the reads must account for the extra holders:

       r_new = (N2 * r2 - N1 * r1) / (N2 - N1)

   per trophy, clamped to 0..100. The estimates are pooled across all
   jumps, weighted by the number of newcomers (Delta N) and by recency
   (half-life 24 h: later joiners are more like the future joiners), and
   shrunk toward the trophy's own current rate so a thin history cannot
   drag a trophy to an extreme.

2. How many players will be tracked at the target. log(players) is fitted
   against log(1 + hours since release) by least squares over the reads
   and read at the target hour. The fit is sublinear in log time, so it
   saturates instead of exploding; the extrapolated log-growth is damped
   by 25% and the result is capped at 25x the last count, never below it.
   With fewer than two distinct player counts the fit cannot run (see 4).

3. Median rate at the target. Holders are conserved across the dilution:

       r_target = (N_last * r_old + (N_target - N_last) * r_new) / N_target

   where r_old is the last rate drifted up a little for the trophies the
   existing players keep earning while they play (a small logit drift,
   larger for common early trophies, near zero for the platinum and for
   late-game rarities). A trophy at 100% stays at 100%.

4. Flat history: the refresh that has not happened yet. When the history
   shows no jump at all (the practice cases: every read identical), the
   program cannot estimate newcomer rates from jumps, so it uses a prior
   calibrated on the one refresh the competition rules publish for launch
   night: players tracked 638 -> 3,868 (factor 6.06) and trophies earned
   943 -> 4,276. The 3,230 newcomers earned 3,333 trophies; the trophy
   at 100% accounts for one per newcomer, leaving 3.19 percentage points
   of newcomer rate spread over the other trophies in proportion to their
   current rates (newcomers hold mostly the earliest trophies, which is
   what the current rates proxy). The same prior says the tracked count
   grows by that factor before the target.

Spread. Quantiles are built on the logit scale around the median with a
normal spread that is deliberately wider than the house baseline's
(0.5 * sqrt(days) was so narrow the practice refresh fell outside its
90% band for every trophy). The spread combines the baseline random-walk
term, a refresh term (1.0 logit when no jump has been seen yet, ~0.45
plus the observed scatter of the jump-implied newcomer rates when jumps
exist), and is clamped to [0.55, 1.4] logit units.

Running: python3 main.py reads every *.in in $MAKEFUN_INPUT_DIR (else
/input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels above.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- Calibration constants (see the module docstring) ---------------------
# Launch-night refresh published in the competition rules: tracked players
# 638 -> 3868, trophies earned 943 -> 4276.
FLAT_GROWTH = 3868.0 / 638.0            # 6.0627: how much the count jumps
FLAT_RESIDUAL_PP = 3.19                 # newcomer rate points left after
                                        # the one trophy every newcomer holds
# Shrinkage for pooled newcomer rates, in units of players.
SHRINK_PLAYERS = 200.0
# Recency half-life for weighting jumps, in hours.
JUMP_HALF_LIFE_H = 24.0
# Baseline random-walk spread, logit units per sqrt(day).
BASE_SIGMA = 0.5
# Extra spread when no jump has been observed yet (flat history).
FLAT_SIGMA = 1.0
# Extra spread when jumps exist: base term plus observed scatter.
JUMP_SIGMA_BASE = 0.45
# Small logit drift per day for trophies the current players keep earning.
DRIFT_LOGIT_PER_DAY = 0.05
# Damping applied to extrapolated log-growth of the tracked count.
GROWTH_DAMP = 0.75
# Hard cap on projected tracked players, as a multiple of the last count.
GROWTH_CAP = 25.0


def clamp_pct(x):
    """Clamp a percent into the range the logit can take (0.01..99.99)."""
    return min(99.99, max(0.01, x))


def logit_pct(rate):
    """logit of a percent rate, with the rate clamped first."""
    p = clamp_pct(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(z):
    """Percent whose logit is z; z is clamped so exp() never overflows."""
    z = min(20.0, max(-20.0, z))
    return 100.0 / (1.0 + math.exp(-z))


def round4(x):
    """Round to 4 decimals, half up (as the starter rounds it).

    Whole numbers come back as ints, so the JSON writes 100, not 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp from a case file."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    """Hours from the release day's start to a timestamp, floored at 0."""
    return max(0.0, (moment - RELEASE).total_seconds() / 3600.0)


def last_known_rates(history, count):
    """Each trophy's rate in the last read that reported it (None if never)."""
    rates = [None] * count
    for read in reversed(history):
        row = read.get("rates") or []
        for i in range(count):
            if rates[i] is None and i < len(row) and row[i] is not None:
                rates[i] = float(row[i])
        if all(v is not None for v in rates):
            break
    return [0.0 if v is None else v for v in rates]


def collect_jumps(history, count):
    """One entry per consecutive pair of reads whose player count grew.

    Each entry is (read time, newcomers, [implied newcomer rate per trophy
    or None where a read lacked the trophy]). Pairs with a tiny growth are
    skipped: a handful of newcomers cannot identify a rate.
    """
    jumps = []
    for prev, cur in zip(history, history[1:]):
        n1 = prev.get("players_tracked")
        n2 = cur.get("players_tracked")
        if n1 is None or n2 is None:
            continue
        newcomers = n2 - n1
        if newcomers < 20:
            continue
        r1 = prev.get("rates") or []
        r2 = cur.get("rates") or []
        implied = []
        for i in range(count):
            a = r1[i] if i < len(r1) else None
            b = r2[i] if i < len(r2) else None
            if a is None or b is None:
                implied.append(None)
            else:
                implied.append(min(100.0, max(0.0, (n2 * b - n1 * a) / newcomers)))
        jumps.append((parse_time(cur["at"]), newcomers, implied))
    return jumps


def pool_newcomer_rates(jumps, last_rates, last_time):
    """Pool the jumps into one newcomer rate per trophy.

    Weighted by newcomers and by recency, shrunk toward the trophy's own
    current rate. Also returns the pooled scatter of the implied rates on
    the logit scale, which feeds the spread: jumps that disagree mean the
    future refresh is less predictable.
    """
    count = len(last_rates)
    pooled = []
    deviations = []
    for i in range(count):
        num = SHRINK_PLAYERS * last_rates[i]
        den = SHRINK_PLAYERS
        for when, newcomers, implied in jumps:
            value = implied[i]
            if value is None:
                continue
            age_h = max(0.0, (last_time - when).total_seconds() / 3600.0)
            weight = newcomers * (0.5 ** (age_h / JUMP_HALF_LIFE_H))
            num += weight * value
            den += weight
        centre = num / den if den else last_rates[i]
        pooled.append(centre)
        for when, newcomers, implied in jumps:
            value = implied[i]
            if value is None:
                continue
            age_h = max(0.0, (last_time - when).total_seconds() / 3600.0)
            weight = newcomers * (0.5 ** (age_h / JUMP_HALF_LIFE_H))
            deviations.append((weight, abs(logit_pct(value) - logit_pct(centre))))
    scatter = 0.0
    if deviations:
        total_w = sum(w for w, _ in deviations)
        scatter = sum(w * d for w, d in deviations) / total_w if total_w else 0.0
    return pooled, scatter


def flat_newcomer_rates(last_rates):
    """Newcomer-rate prior for a history with no jump (see docstring).

    The trophy everyone holds (100%) is held by every newcomer too; the
    remaining 3.19 points of newcomer rate are shared over the other
    trophies in proportion to their current rates.
    """
    others = [r for r in last_rates if r < 99.995]
    total = sum(others)
    prior = []
    for rate in last_rates:
        if rate >= 99.995:
            prior.append(100.0)
        elif total > 0:
            prior.append(FLAT_RESIDUAL_PP * rate / total)
        else:
            prior.append(0.0)
    return prior


def project_players(history, target_time, last_count):
    """Project the tracked-player count at the target.

    Returns (projected count, True when the history was flat and the
    launch-night prior was used instead of a fit).
    """
    # One point per distinct player count, at the first read that showed it:
    # between refreshes the page repeats the same count for hours, and
    # letting those repeats vote would let a long stale stretch outweigh
    # the refreshes that actually trace the growth.
    seen_counts = {}
    for read in history:
        n = read.get("players_tracked")
        if n is None or n <= 0:
            continue
        if n not in seen_counts:
            seen_counts[n] = parse_time(read["at"])
    points = [(math.log1p(hours_since_release(when)), math.log(n))
              for n, when in seen_counts.items()]
    points.sort()
    distinct_counts = set(seen_counts)
    if len(distinct_counts) < 2 or len(points) < 2:
        return last_count * FLAT_GROWTH, True

    # Least squares for log N = a + b * log(1 + hours since release).
    n_pts = len(points)
    mean_x = sum(p[0] for p in points) / n_pts
    mean_y = sum(p[1] for p in points) / n_pts
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx if sxx > 0 else 0.0
    slope = min(1.2, max(0.0, slope))
    intercept = mean_y - slope * mean_x
    target_x = math.log1p(hours_since_release(target_time))
    fitted = math.exp(intercept + slope * target_x)
    # Damp the extrapolated growth: the log-log line is a trend, not a law.
    if fitted > last_count:
        projected = last_count * math.exp(GROWTH_DAMP * math.log(fitted / last_count))
    else:
        projected = fitted
    projected = min(projected, last_count * GROWTH_CAP)
    return max(float(last_count), projected), False


def drift_logit(rate, tier, horizon_days):
    """Small upward logit drift for trophies current players keep earning.

    Common early trophies move most; the platinum and late-game rarities
    barely move. A trophy at 100% never moves (handled by the caller).
    """
    if rate >= 99.995:
        return 0.0
    ease = min(1.0, max(0.05, rate / 5.0))
    tier_factor = {"platinum": 0.1, "gold": 0.4, "silver": 0.7}.get(tier, 1.0)
    return min(0.25, DRIFT_LOGIT_PER_DAY * horizon_days * ease * tier_factor)


def forecast_case(case, name):
    """Forecast one case: {"trophies": [{"id": n, "q": [5 quantiles]}]}."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    count = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target_time = parse_time(case["target"])
    last_read = history[-1]
    last_count = last_read.get("players_tracked") or 0
    last_time = parse_time(last_read["at"])
    last_rates = last_known_rates(history, count)

    jumps = collect_jumps(history, count)
    total_newcomers = sum(n for _, n, _ in jumps)
    has_jumps = total_newcomers >= 100

    if has_jumps:
        newcomer_rates, scatter = pool_newcomer_rates(jumps, last_rates, last_time)
    else:
        newcomer_rates = flat_newcomer_rates(last_rates)
        scatter = 0.0
    target_count, used_flat_prior = project_players(history, target_time, last_count)

    base_sigma = BASE_SIGMA * math.sqrt(horizon_days)
    if used_flat_prior:
        sigma = math.sqrt(base_sigma ** 2 + FLAT_SIGMA ** 2)
    else:
        extra = JUMP_SIGMA_BASE + 0.4 * min(scatter, 1.0)
        sigma = math.sqrt(base_sigma ** 2 + extra ** 2)
    sigma = min(1.4, max(0.55, sigma))

    rows = []
    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate >= 99.995:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        drifted = inv_logit_pct(logit_pct(rate) + drift_logit(rate, trophy.get("tier"), horizon_days))
        if target_count > last_count:
            median = (last_count * drifted + (target_count - last_count) * newcomer_rates[i]) / target_count
        else:
            median = drifted
        median = clamp_pct(median)
        centre = logit_pct(median)
        quantiles = [round4(clamp_pct(inv_logit_pct(centre + z * sigma))) for z in Z]
        # Rounding can only break monotonicity by a hair; enforce the order.
        for j in range(1, 5):
            if quantiles[j] < quantiles[j - 1]:
                quantiles[j] = quantiles[j - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def persistence_fallback(case):
    """The house baseline (last rate, narrow spread): never crash a case."""
    last = case["history"][-1]
    spread = BASE_SIGMA * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if last.get("rates") and i < len(last["rates"]) else 0.0
        if rate is None:
            rate = 0.0
        centre = logit_pct(rate)
        rows.append({"id": trophy["id"],
                     "q": [round4(clamp_pct(inv_logit_pct(centre + z * spread))) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = persistence_fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
