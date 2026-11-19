# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ysf30/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ysf30/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on [Exophase](https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/)
at 2026-11-26T00:00:00Z — a week after launch — as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%). Built by an AI agent for make.fun's
[“Forecast how many GTA VI players earn each trophy in week one”](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty ($250 in USDC).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no installs. Reads every
`<id>.in` case in the input directory and writes `<id>.out` for each:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.
Deterministic: two runs on the same case write the same bytes.

## The model

Exophase does not refresh continuously. A case's history is a step function:
long flat stretches, then a refresh where the players-tracked count jumps and
every earn rate moves at once. The forecast therefore models **counts, not
rates**:

- **Refresh states.** Consecutive reads with identical counts collapse to one
  state. Each state gives integer earners per trophy
  (`E = round(rate * players / 100)`) and the players-tracked count.
- **Players.** `N(t)` is fit as a power law in time since release on the
  states, so the join rate (new players syncing per hour) is known.
- **Earners.** Each trophy's earners follow a cohort model: a player who
  joined at time `s` earns trophy `i` by age `a` with probability
  `q_inf_i / (1 + (a50_i / a)^p)` — a log-logistic progression curve with a
  per-trophy median time-to-earn `a50` and an eventual share `q_inf`.
  Earners at time `t` are the join-rate-weighted sum over cohorts. `a50` and
  `q_inf` are fit per trophy by grid search on the states, shrunk toward
  priors by tier and story order (Chapter 1 is fast, the platinum is slow;
  later chapters cannot out-earn earlier ones).
- **The answer is a refresh, not a moment.** The graded read is the first one
  in the hour from the target, i.e. the page as of the *last refresh before
  the target*. The forecast is evaluated over the last-refresh window
  (target minus the median refresh interval), and the spread covers the
  timing of that refresh plus the fit residual.
- **Launch-night flat history.** When the history is a single unchanged state
  (the practice cases), the program expects the first refresh: players
  multiply ~6x (638 → 3,868, as the kit's competition notes record) while
  earners grow only ~1.34x in total, concentrated in the earliest trophies
  (fishing, arcade, Chapter 1 — calibrated from the live page on 19 Nov).
  The forecast is a mixture of the refreshed and unrefreshed outcomes.
- **Structure.** Chapter trophies are monotone (Chapter 2 cannot out-earn
  Chapter 1), the platinum cannot out-earn any trophy, and “Gone Fishing”
  stays near 100% — these are enforced on the medians.

## Why this beats persistence

Persistence says every rate stays where it is. But a refresh dilutes every
rate (new players have earned almost nothing yet) while cohorts age into the
early trophies, so rates move in opposite directions by trophy: the early
ones recover and climb, the late ones collapse toward zero. A single
rate-trend fit cannot see that; an earners-over-players decomposition can.
