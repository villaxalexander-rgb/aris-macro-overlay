# A.R.I.S Research Artifacts — April 2026 Alpha Engine Review

This directory contains the research that drove the Phase A decision to swap from pure BSV to a 50/50 TSMOM + BSV hybrid.

## Files

FilePurpose`aris_backtest_v2.py`Scratch copy of the backtest engine used for scenario exploration. Mirrors `../aris_backtest.py` but unused in production.`price_cache.pkl`Frozen yfinance snapshot pulled 2026-04-24. 22 commodity futures × 4,753 trading days (2007-05-22 → 2026-04-10) + regime series derived from SPY/TIP proxies. **This is the deterministic data substrate** — all scenario comparisons in this folder were run against this single cache, so they are apples-to-apples.`scenario_runner.py`v1 sweep — unfrozen, drifting yfinance data. Superseded by v2.`scenario_runner_v2.py`v2 sweep — 13 BSV parameterizations on frozen data. Established that no BSV parameterization produces positive Sharpe.`scenario_results_v2.json`v2 sweep results. Best: scenario 12 (CAGR +1.12%, Vol 11.6%, Sharpe −0.11 arith / +0.10 raw, DD −27.4%) with ideal NAV eligibility + fractional contracts + weekly rebalance + 100% gross cap + baseline composite.`tsmom_runner.py`TSMOM sweep — 9 parameterizations of Moskowitz/Ooi/Pedersen time-series momentum, plus 2 hybrid BSV+TSMOM blends.`tsmom_results.json`TSMOM sweep results. Best: **T6 hybrid** (CAGR +1.12%, Vol 3.6%, Sharpe +0.31 raw, DD −8.3%) with 50% TSMOM continuous (3/6/12m) + 50% BSV, vol-targeted 10%, weekly rebalance, 100% gross cap.

## How to reproduce

```bash
cd backtest/research
pip install yfinance pandas numpy --break-system-packages
python3 scenario_runner_v2.py    # 13 BSV scenarios — ~5 min
python3 tsmom_runner.py           # 9 TSMOM scenarios — ~5 min
```

Both runners read `price_cache.pkl` and produce the JSON output in-place. If the cache is missing, `scenario_runner.py` will re-fetch from yfinance (and drift slightly from these results).

## Key findings

1. **The April 12 metrics.json in `../results/` is not reproducible.** Re-running the unchanged backtest 12 days later produced CAGR −2.6% vs saved +3.7%. Root causes: yfinance continuous-futures retroactive adjustment + NAV-dependent lot-size feedback loop.

2. **No BSV parameterization produces positive risk-adjusted return.** Across 13 parameter variations (Stagflation energy weights 0.8/1.2/1.4, weekly/daily rebal, gross caps, composite reweights, pure momentum), best Sharpe was −0.11 (arith) / +0.10 (raw), CAGR +1.12%.

3. **The per-contract 3% cap creates path dependence.** At sub-$1M NAV, large-notional legs (Cocoa, Sugar, Feeder Cattle) drop out, cutting avg positions from 10 to 4 and locking in diversification loss. Lifting this to an "ideal NAV" reference adds +1.6pp CAGR.

4. **Cost drag was 1.39%/yr on daily rebalance.** Weekly rebalance cuts cost drag by ~77% ($68k → $16k across 15yr) with negligible signal fidelity loss.

5. **Pure momentum loses to the composite.** Value + reversal components are helping, not hurting — the opposite of the initial hypothesis.

6. **Stagflation weight flip (energy 0.8 → 1.4) is a dead hypothesis.** Moves CAGR by < 0.2pp in every variant.

7. **TSMOM beats BSV on risk-adjusted return at matched CAGR.** T6 hybrid delivers BSV's best CAGR at one-third the vol and one-third the drawdown.

8. **T6 is consistent with real CTAs over the same period.** BTOP50, AQR Managed Futures, Winton Global all produced Sharpe 0.2-0.7 on commodity-only TSMOM in the post-QE era. The 2011-2026 window was structurally bad for commodity trend-following.

## What gets shipped

`../../PHASE_A_PATCH_SPEC.md` has the concrete patch to swap BSV → T6 hybrid in the live signal pipeline. Ship target: end May 2026.
