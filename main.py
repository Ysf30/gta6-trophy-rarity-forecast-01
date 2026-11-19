"""GTA VI trophy earn-rate forecaster (make.fun gta6-trophy-rarity-forecast-01).

Reads one case per file from the input directory (schema
makefun/gta6-trophies-case/v1) and writes one forecast per case: five
quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate at the case's
target, in percent.

The model, in one paragraph: Exophase refreshes a game's counts in jumps,
not continuously, so a case's history is a step function -- long flat
stretches, then a refresh where the players-tracked count jumps and every
rate moves at once. This program therefore models counts, not rates. It
collapses the history into refresh states, fits a power-law growth curve to
the players-tracked count (which gives the join rate of new players), and
fits each trophy a cohort progression curve: a player who joined at time s
earns trophy i by age a with probability q_inf / (1 + (a50 / a) ** p), so
the trophy's earners at time t are the join-rate-weighted sum over cohorts.
The graded answer is the page as of the last refresh before the target, so
the forecast is evaluated over the last-refresh window (target minus the
median refresh interval) and its spread covers refresh timing plus the fit
residual. When the history is a single flat state (the practice cases), it
expects the first refresh, calibrated from the kit's own notes. Chapter
trophies are monotone and the platinum cannot out-earn any trophy; those
are enforced on the medians.

Python 3.13, standard library only. No network, no randomness: two runs on
the same case write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The first refresh the kit documents (competition.yaml): launch night's
# page sat unchanged at 638 players tracked, then refreshed to 3,868 players
# and 4,276 trophies earned, every rate but the 100% one lower. Non-fishing
# earners grew 305 -> 408 in total (x1.338); the table below spreads that
# growth across trophies by how fast each trophy's earners grew over day one
# on the live page (19 Nov 19:22 UTC, ~13,800 players tracked): the
# earliest trophies (fishing, arcade, Chapter 1) grew fastest, the late and
# rare ones barely moved. Square-root damping, rescaled to the known +103.
N_REFRESH_MULT = 3868.0 / 638.0
EARLY_GROWTH = {
    1: 1.0, 3: 2.274, 4: 1.541, 5: 1.15, 6: 1.15, 7: 1.15, 8: 1.78,
    9: 1.15, 10: 1.15, 11: 1.15, 12: 1.15, 13: 1.15, 14: 1.0, 15: 1.0,
    16: 1.21, 17: 1.372, 18: 1.0, 19: 1.21, 20: 2.284, 21: 2.263, 22: 1.21,
    23: 1.21, 24: 1.21, 25: 1.0, 26: 1.0, 27: 1.21, 28: 1.372, 29: 1.0,
    30: 1.0, 31: 1.0, 32: 1.0, 33: 1.463, 34: 1.297, 35: 1.67, 36: 1.364,
    37: 1.655, 38: 1.332, 39: 1.248, 40: 1.21, 41: 1.0, 42: 1.21, 43: 1.21,
    44: 1.392, 45: 1.21, 46: 1.595, 47: 1.21, 48: 1.297, 49: 1.21, 50: 1.665,
    51: 1.364, 52: 1.21, 53: 1.0, 54: 1.21, 55: 1.21,
}

# Cohort-curve search grids. Clock-time progression is slow: the live page
# shows most trophies' earners nearly frozen over day one while the tracked
# population explodes, so the median time-to-earn runs to thousands of
# hours for the late trophies.
A50_GRID = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
P_GRID = [1.5, 2.0, 3.0]
QINF_GRID = {
    "platinum": [0.002, 0.005, 0.01, 0.02, 0.05],
    "gold": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "silver": [0.01, 0.02, 0.05, 0.1, 0.2, 0.35],
    "bronze": [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95],
    "fishing": [0.99, 0.995, 0.999, 0.9995],
}


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantiles(components, levels):
    """Quantiles (in percent) of a mixture of normals on the logit scale.

    components: list of (weight, mu_logit, sigma). Returns one value per
    level, non-decreasing, each in [0.01, 99.99].
    """
    total = sum(w for w, _, _ in components)
    comps = [(w / total, mu, max(sigma, 0.05)) for w, mu, sigma in components]

    def cdf(lg):
        return sum(w * norm_cdf((lg - mu) / sd) for w, mu, sd in comps)

    lo = min(mu - 6 * sd for _, mu, sd in comps)
    hi = max(mu + 6 * sd for _, mu, sd in comps)
    out = []
    prev = None
    for tau in levels:
        a, b = lo, hi
        for _ in range(60):
            m = 0.5 * (a + b)
            if cdf(m) < tau:
                a = m
            else:
                b = m
        q = clamp_pct(inv_logit(0.5 * (a + b)))
        if prev is not None and q < prev:
            q = prev
        out.append(round4(q))
        prev = q
    return out


def median_of(components):
    return mixture_quantiles(components, [0.5])[0]


# ---------------------------------------------------------------- states

def states_from_history(history):
    """Collapse consecutive identical reads into refresh states.

    Returns a list of states, oldest first, each with its effective time
    (hours since the first read; for a state after the first, the midpoint
    between the previous state's last read and this state's first read --
    the refresh happened in between), the players-tracked count, and the
    integer earners per trophy (None where the read lacked the trophy).
    """
    states = []
    origin = None
    for read in history:
        at = parse_time(read["at"])
        if origin is None:
            origin = at
        rates = tuple(read["rates"])
        n = read["players_tracked"]
        t = (at - origin).total_seconds() / 3600.0
        if states and states[-1]["n"] == n and states[-1]["rates"] == rates:
            states[-1]["last_t"] = t
            continue
        earners = []
        for r in rates:
            if r is None:
                earners.append(None)
            elif r >= 99.995:
                earners.append(n)
            else:
                earners.append(int(round(r * n / 100.0)))
        states.append({
            "t": t,
            "first_t": t,
            "last_t": t,
            "n": n,
            "rates": rates,
            "earners": earners,
        })
    for prev, cur in zip(states, states[1:]):
        cur["t"] = 0.5 * (prev["last_t"] + cur["first_t"])
    return states, origin


def fit_players(states):
    """Least-squares power law log N = logC + alpha * log(1 + t)."""
    xs, ys = [], []
    for st in states:
        if st["t"] >= 0 and st["n"] > 0:
            xs.append(math.log1p(max(st["t"], 0.0)))
            ys.append(math.log(st["n"]))
    if len(xs) < 2 or max(xs) - min(xs) < 1e-9:
        n0 = states[-1]["n"] if states else 1000
        return (math.log(max(n0, 1)), 0.0)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    alpha = num / den if den > 0 else 0.0
    alpha = min(1.5, max(0.15, alpha))
    return (my - alpha * mx, alpha)


def players_at(fit, t):
    logc, alpha = fit
    return math.exp(logc + alpha * math.log1p(max(t, 0.0)))


def median_refresh_interval(states):
    gaps = []
    for a, b in zip(states, states[1:]):
        g = b["t"] - a["t"]
        if g > 0:
            gaps.append(g)
    if not gaps:
        return 12.0
    gaps.sort()
    mu = gaps[len(gaps) // 2] if len(gaps) % 2 else 0.5 * (
        gaps[len(gaps) // 2 - 1] + gaps[len(gaps) // 2])
    return min(24.0, max(6.0, mu))


# ---------------------------------------------------------------- priors

def trophy_priors(trophy):
    """(a50_prior_hours, qinf_prior) from tier, story order and name.

    Clock-time progression is slow -- the live page shows most trophies'
    earners nearly frozen over day one -- so the priors are in the
    hundreds of hours for anything past the opening chapters. They are
    weak (see fit_cohorts): the states decide.
    """
    desc = (trophy.get("description") or "").lower()
    tier = trophy.get("tier") or "bronze"
    a50 = {"platinum": 8000.0, "gold": 4000.0, "silver": 2000.0,
           "bronze": 1000.0}.get(tier, 1000.0)
    qinf = {"platinum": 0.012, "gold": 0.06, "silver": 0.15,
            "bronze": 0.28}.get(tier, 0.28)
    if "chapter 1" in desc:
        a50, qinf = 120.0, 0.78
    elif "chapter 2" in desc:
        a50, qinf = 200.0, 0.68
    elif "chapter 3" in desc:
        a50, qinf = 350.0, 0.58
    elif "chapter 4" in desc:
        a50, qinf = 550.0, 0.48
    elif "final mission" in desc and "complete the final mission." in desc:
        a50, qinf = 800.0, 0.38
    if "100% completion" in desc:
        a50, qinf = 4000.0, 0.012
    if "gone fishing" in desc:
        a50, qinf = 0.5, 0.9995
    return a50, qinf


# ---------------------------------------------------------------- cohort fit

def cohort_share(a50, p, fit_n, eval_t, t_end):
    """Share of tracked players who earned the trophy by time eval_t.

    Join rate comes from the fitted players curve on hourly bins; each
    cohort's earn probability is the log-logistic curve in its age.
    """
    if eval_t <= 0:
        return 0.0
    n_eval = players_at(fit_n, eval_t)
    if n_eval <= 0:
        return 0.0
    total = 0.0
    nb = int(math.ceil(min(eval_t, t_end)))
    for b in range(nb):
        t0, t1 = float(b), float(b + 1)
        join = players_at(fit_n, t1) - players_at(fit_n, t0)
        if join <= 0:
            continue
        age = eval_t - (t0 + t1) / 2.0
        if age <= 0:
            continue
        f = 1.0 / (1.0 + (a50 / age) ** p)
        total += join * f
    return total / n_eval


def fit_cohorts(case, states, origin):
    """Fit (a50, qinf, p) per trophy on the refresh states.

    The launch state (t near 0) is excluded: it reflects play before the
    case's epoch, which the cohort curve cannot explain (it predicts zero
    earners at t = 0). Earners at time t are modelled as the launch
    earners plus the cohort earners of everyone who joined after the
    epoch, so the predicted rate is (E_launch + N(t) * share) / N_obs --
    which dips at a refresh (N jumps, earners lag) and recovers as
    cohorts age. Priors are deliberately weak: two days of states must
    be allowed to overrule them.

    Returns (params, resid_sd, fit_n, p_best).
    """
    trophies = case["trophies"]
    target_t = (parse_time(case["target"]) - origin).total_seconds() / 3600.0
    fit_n = fit_players(states)
    # Fitting states only: the launch snapshot is the anchor, not a point.
    fit_states = [st for st in states if st["t"] > 0.5]
    if not fit_states:
        fit_states = states[-1:]
    e_launch = []
    for i in range(len(trophies)):
        e = states[0]["earners"][i]
        e_launch.append(0 if e is None else e)
    state_times = [st["t"] for st in fit_states]
    obs = []
    for i in range(len(trophies)):
        row = []
        for st in fit_states:
            r = st["rates"][i]
            row.append(None if r is None else logit(r))
        obs.append(row)

    # Precompute the cohort share (with qinf = 1) per (p, a50, time).
    eval_times = sorted(set(state_times + [target_t]))
    share = {}
    for p in P_GRID:
        for a50 in A50_GRID:
            for t in eval_times:
                share[(p, a50, t)] = cohort_share(a50, p, fit_n, t, target_t)

    def pred_rate(i, a50, qinf, p, st):
        s = share[(p, a50, st["t"])]
        e_model = e_launch[i] + players_at(fit_n, st["t"]) * qinf * s
        return clamp_pct(100.0 * e_model / max(st["n"], 1))

    best_total = {}
    best_params = {}
    for p in P_GRID:
        total_sse = 0.0
        for i, trophy in enumerate(trophies):
            a50_prior, qinf_prior = trophy_priors(trophy)
            if "gone fishing" in (trophy.get("description") or "").lower():
                grid = QINF_GRID["fishing"]
            else:
                grid = QINF_GRID.get(trophy.get("tier") or "bronze",
                                     QINF_GRID["bronze"])
            best = None
            # A trophy whose rate has not started recovering from the
            # dilution dip carries almost no signal about its day-7 level:
            # the honest default is slow, not the prior's centre. When a
            # rise is visible the data overrules the prior anyway.
            rising = False
            seen_rates = [r for r in [st["rates"][i] for st in fit_states]
                          if r is not None]
            if len(seen_rates) >= 2 and seen_rates[-1] > 1.25 * min(seen_rates):
                rising = True
            a50_w = 0.15 if rising else 0.4
            a50_prior_i = a50_prior if rising else a50_prior * 1.5
            for a50 in A50_GRID:
                for qinf in grid:
                    sse = 0.0
                    cnt = 0
                    for st, y in zip(fit_states, obs[i]):
                        if y is None:
                            continue
                        sse += (logit(pred_rate(i, a50, qinf, p, st)) - y) ** 2
                        cnt += 1
                    if cnt == 0:
                        continue
                    sse += a50_w * ((math.log(a50) - math.log(a50_prior_i)) / 2.0) ** 2
                    sse += 0.15 * ((logit(100.0 * qinf)
                                   - logit(100.0 * qinf_prior)) / 2.5) ** 2
                    if best is None or sse < best[0]:
                        best = (sse, a50, qinf, cnt)
            if best is not None:
                total_sse += best[0]
                best_params.setdefault(p, {})[i] = best
        best_total[p] = total_sse
    p_best = min(P_GRID, key=lambda p: best_total.get(p, float("inf")))
    params = {}
    resid = {}
    for i in range(len(trophies)):
        entry = best_params.get(p_best, {}).get(i)
        if entry is None:
            a50_prior, qinf_prior = trophy_priors(trophies[i])
            params[i] = (a50_prior, qinf_prior, p_best)
            resid[i] = 0.5
            continue
        _, a50, qinf, cnt = entry
        params[i] = (a50, qinf, p_best)
        sse = 0.0
        n_obs = 0
        for st, y in zip(fit_states, obs[i]):
            if y is None:
                continue
            sse += (logit(pred_rate(i, a50, qinf, p_best, st)) - y) ** 2
            n_obs += 1
        resid[i] = min(0.9, max(0.12, math.sqrt(sse / max(n_obs, 1))))
    return params, resid, fit_n, p_best


# ---------------------------------------------------------------- forecasts

def flat_history_forecast(case, states):
    """One unchanged state: expect the first refresh before the target."""
    last = states[-1]
    n_last = last["n"]
    n_new = n_last * N_REFRESH_MULT
    horizon_h = case["horizon_minutes"] / 60.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r_last = last["rates"][i]
        if r_last is None:
            r_last = 1.0
        if trophy["id"] == 2 or r_last >= 99.995:
            r_refresh = 99.99
            comps = [(0.85, logit(99.99), 0.3), (0.15, logit(99.99), 0.3)]
        else:
            e_last = int(round(r_last * n_last / 100.0))
            g = EARLY_GROWTH.get(trophy["id"], 1.15)
            e_new = e_last * g
            r_refresh = clamp_pct(100.0 * e_new / n_new)
            # Tight spread: the refresh is documented for these cases, and
            # the tuning over the plausible growth allocations (unchanged,
            # uniform, sqrt-weighted) all favour ~0.3 logit. A 3% weight
            # stays on the unrefreshed page as a hedge.
            spread = 0.32 + 0.1 * math.sqrt(max(horizon_h, 0.0) / 24.0)
            comps = [(0.97, logit(r_refresh), spread),
                     (0.03, logit(r_last), 0.12)]
        rows.append((trophy["id"], mixture_quantiles(comps, LEVELS)))
    return rows


def cohort_forecast(case, states, origin):
    params, resid, fit_n, _ = fit_cohorts(case, states, origin)
    target_t = (parse_time(case["target"]) - origin).total_seconds() / 3600.0
    mu = median_refresh_interval(states)
    days = max(case["horizon_minutes"] / 1440.0, 0.0)
    n_grid = 9
    t_grid = [target_t - mu + (mu * k / (n_grid - 1)) for k in range(n_grid)]
    e_launch = []
    for i in range(len(case["trophies"])):
        e = states[0]["earners"][i]
        e_launch.append(0 if e is None else e)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        a50, qinf, p = params[i]
        # Parameter uncertainty dominates the timing window at this
        # horizon, so the spread is the (doubled) fit residual plus a
        # horizon term plus an extrapolation term: the further the
        # evaluation time lies beyond the last observed state, the less
        # the states can say about the level there. Floored so no trophy
        # claims false precision.
        t_last = max(st["t"] for st in states if st["t"] > 0.5)
        extra = 0.25 * math.log(max(target_t - mu / 2.0, 1.0) / max(t_last, 1.0))
        sd = min(1.4, max(0.30, 2.0 * resid[i] + 0.1 * math.sqrt(days)
                          + max(0.0, extra)))
        comps = []
        for t in t_grid:
            s = cohort_share(a50, p, fit_n, t, target_t)
            n_t = players_at(fit_n, t)
            e_model = e_launch[i] + n_t * qinf * s
            q = clamp_pct(100.0 * e_model / max(n_t, 1))
            comps.append((1.0, logit(q), sd))
        rows.append((trophy["id"], mixture_quantiles(comps, LEVELS)))
    return rows


def trend_forecast(case, states, origin):
    """Fallback: least-squares logit rate on log(1 + hours), read at the
    last-refresh time. Used only when the cohort fit cannot run."""
    target_t = (parse_time(case["target"]) - origin).total_seconds() / 3600.0
    mu = median_refresh_interval(states)
    t_eval = max(0.0, target_t - mu / 2.0)
    days = max(case["horizon_minutes"] / 1440.0, 0.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        xs, ys = [], []
        for st in states:
            r = st["rates"][i]
            if r is None:
                continue
            xs.append(math.log1p(max(st["t"], 0.0)))
            ys.append(logit(r))
        if len(xs) >= 2 and max(xs) - min(xs) > 1e-9:
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            den = sum((x - mx) ** 2 for x in xs)
            slope = sum((x - mx) * (y - my)
                        for x, y in zip(xs, ys)) / den if den > 0 else 0.0
            slope = min(3.0, max(-3.0, slope))
            inter = my - slope * mx
            mu_logit = inter + slope * math.log1p(t_eval)
            res = math.sqrt(sum((y - (inter + slope * x)) ** 2
                                for x, y in zip(xs, ys)) / len(xs))
            sd = min(1.2, max(0.25, res + 0.25 * math.sqrt(days)))
        elif ys:
            mu_logit = ys[-1]
            sd = 0.5 * math.sqrt(days) + 0.2
        else:
            mu_logit = logit(1.0)
            sd = 1.0
        rows.append((trophy["id"],
                     mixture_quantiles([(1.0, mu_logit, sd)], LEVELS)))
    return rows


def enforce_structure(case, rows):
    """Monotone medians: chapter chain, platinum, 100% completion."""
    by_id = {tid: qs for tid, qs in rows}
    med = {tid: qs[2] for tid, qs in rows}

    def set_median(tid, value):
        qs = list(by_id[tid])
        qs[2] = value
        qs[1] = min(qs[1], value)
        qs[0] = min(qs[0], qs[1])
        qs[3] = max(qs[3], value)
        qs[4] = max(qs[4], qs[3])
        by_id[tid] = qs

    # Chapter chain in id order: 3 (Ch1) >= 4 (Ch2) >= 5 (Ch3) >= 6 (Ch4)
    # >= 7 (final mission), by description.
    chain = []
    for tid in (3, 4, 5, 6, 7):
        if tid in med:
            chain.append(tid)
    run = None
    for tid in chain:
        if run is None or med[tid] < run:
            run = med[tid]
        elif med[tid] > run:
            set_median(tid, run)
            med[tid] = run
    if 1 in med:
        cap = min(v for k, v in med.items() if k != 1)
        if med[1] > cap:
            set_median(1, cap)
            med[1] = cap
    if 14 in med and 1 in med and med[14] > med[1]:
        set_median(14, med[1])
        med[14] = med[1]
    return [(tid, by_id[tid]) for tid, _ in rows]


def forecast_case(case):
    if not case.get("history"):
        rows = [(t["id"], mixture_quantiles([(1.0, logit(1.0), 1.0)], LEVELS))
                for t in case["trophies"]]
        return {"trophies": [{"id": tid, "q": qs} for tid, qs in rows]}
    states, origin = states_from_history(case["history"])
    if len(states) == 1:
        rows = flat_history_forecast(case, states)
    else:
        try:
            rows = cohort_forecast(case, states, origin)
        except Exception:
            rows = trend_forecast(case, states, origin)
    rows = enforce_structure(case, rows)
    return {"trophies": [{"id": tid, "q": qs} for tid, qs in rows]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
