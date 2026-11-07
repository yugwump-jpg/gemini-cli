# Desk Rules

Every agent reads this file before acting. Changes need the owner's approval.

## Owner answers (intake, 2026-11-07)

| Rule | Value |
|---|---|
| Markets | Crypto, spot only, long only, liquid majors listed on Alpaca (BTC, ETH, SOL and similar) |
| Paper capital | $200 |
| Risk per trade | 2% of equity (≈ $4 at start), sized from an ATR-based stop |
| Holding period | As short as survives fees. Hourly scheduling, so no scalping; test holds of hours to days |
| Never trade | Leverage, perps, shorting, memecoins, new listings, anything promoted in pump groups or without data |
| Time zone | Mountain Time (America/Denver); morning brief by 07:30 MT |
| Broker | Alpaca paper trading only |

## Hard rules

- Only real data with a named source. Never invent prices, backtest results or returns. Label every estimate as an estimate.
- No guaranteed profit claims. Account for multiple testing, survivorship and lookahead bias.
- Nothing that trades real money, connects an account or pays for anything without the owner's approval.
- Only the Trader touches order tools. The Risk Officer's veto is final.
- No new strategy launches into FOMC or CPI days, or major token unlocks for the coin traded.
- Every handoff is logged to `journal/` as TICKER / IDEA / EVIDENCE / NEXT STEP.

## Kill switch

Stop all new trades for the day on any of:
- 2% daily loss (note: with 2% risk per trade, one full stop-out reaches this)
- 3 losses in a row
- Stale data (latest bar older than 2 bars)

## Strategy bar

Score 1–5 each on out-of-sample Sharpe, deflated Sharpe, max drawdown, trade count and edge after costs.
Keep only strategies scoring 18+ of 25, positive after costs on unseen data across at least 30 trades.

## Promotion ladder

2 weeks paper → paper results match backtest → live at 10% size with owner approval → full rules.
