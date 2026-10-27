# Investment Copilot

A polished personal ETF portfolio workspace built with Python, Streamlit, Pandas and Plotly. Investment Copilot supports local SQLite development and hosted PostgreSQL deployments with administrator-provisioned accounts and per-user data schemas. It includes holdings, portfolio snapshots, benchmark comparisons, risk analytics and ETF research.

**Current version: 1.3.3** — semantic version recorded in `VERSION`; see `CHANGELOG.md` for release notes.

> This is personal-finance software, not a licensed Bloomberg terminal or a brokerage platform. Market data can be delayed, incomplete or incorrect. Analytics are educational and are not tax, investment or legal advice. No trades are executed.

## What's included in this release

### Portfolio workflow
- **Investments-first workflow** with a polished, light portfolio home inspired by modern consumer investment dashboards, value and daily-change cards, valuation history, allocation, holdings grid, dynamic multi-row add grid, spreadsheet paste, CSV import, persistent custom display names, and flexible add/buy/sell/remove actions.
- Transaction ledger for opening positions, buys, sells, dividends, fees and cash movements.
- SQLite-backed instrument master with display name, ISIN, exchange, asset class and notes.
- SQLite-backed multi-asset watchlist with quick-add presets for FX pairs (EURPLN, EURUSD), crypto (Bitcoin, Ethereum), commodities (gold, silver, oil, copper, natural gas), US Treasury yields/futures, bond ETFs, major indices and volatility; custom provider symbols remain supported.
- Multi-currency valuations and historical transaction-date FX where supported.
- Manual valuation snapshots and CSV ledger exports.

### Terminal and analytics
- Light, wide-layout investor dashboard with allocation, daily move estimates, holdings table and data-status warnings.
- Performance workspace with adjusted-close ETF total-return index, reporting-currency conversion, selectable benchmarks, relative return, annualized return/volatility, drawdown and linked ticker contributions.
- Separate approximate money-weighted XIRR using recorded transaction flows.
- Holdings risk view with annualized volatility, maximum drawdown, worst daily move and correlations. Price-series gaps are not silently filled indefinitely.
- ETF Research workspace for instrument metadata, exposure by held ETF, sourced constituent CSV imports, weight-coverage checks and pairwise shared-weight overlap.
- Market proxy dashboard, AI read-only analyst (optional OpenRouter key), and Monte Carlo Forecast Lab whose inflation-adjusted labels follow the selected reporting currency.

### Data reliability and operations
- Persistent SQLite cache for latest quotes, adjusted daily prices and historical FX fixings; rows are keyed by ticker/date to avoid duplicate daily history.
- TTL-cached quote results, provider/source status, price-date warnings, an explicit FX/price gap policy, and an ingestion run audit trail.
- Separate scheduled worker (`scripts/market_data_worker.py`) refreshes portfolio/watchlist/benchmark symbols and records complete PLN valuations when possible.
- ETF constituent files retain source name, optional source URL, as-of date and import time; incomplete weight coverage is flagged and not normalized to 100%.
- Automated tests for ledger accounting, cache persistence, return calculations, FX conversion, benchmark alignment, linked attribution and ETF overlap.
- Docker Compose starter runs the terminal and a separate persistent market-data worker against the same named data volume.

## Run in VS Code + WSL2

```bash
cd adatch-investor-portfolio
chmod +x start.sh
./start.sh
```

Open `http://localhost:8501`. The launcher creates `.venv`, installs dependencies and copies `.env.example` to `.env` if needed.

Run tests:

```bash
.venv/bin/python -m compileall -q app.py portfolio.py advanced_analytics.py performance_engine.py research_analytics.py scripts tests
ETF_APP_LOCAL_DEV_MODE=1 .venv/bin/python -m unittest discover -s tests -v
```

## Historical performance, benchmark and attribution

The **Performance** workspace builds a holdings-weighted, chain-linked return series from daily adjusted-close histories and the transaction ledger. The code uses adjusted series as a dividend/split-adjusted total-return proxy, converts each listing into the selected reporting currency using historical Frankfurter reference FX data, and compares the result with the chosen benchmark over common valid dates. The ticker contribution table uses a geometric linking method so linked contributions reconcile closely to the portfolio series.

Important boundaries: this is an **ETF-holdings return proxy**, not a full-account return. It excludes brokerage cash, taxes/withholding, non-ETF assets and exact intraday transaction timing. The provider's adjusted-close series assumes fund distributions are reinvested as reflected by the provider. Historical price alignment permits only a short three-observation forward fill to handle different exchange/FX holidays; longer missing-data gaps break the comparison segment rather than being treated as a zero return. A custom benchmark uses a Yahoo Finance symbol and is converted to the selected reporting currency where listing-currency and FX data exist.

## Sourced ETF holdings and overlap

Under **ETF Research → Sourced ETF constituent data & overlap**, import CSVs with the following columns:

```csv
fund_ticker,constituent_ticker,weight_pct,constituent_name
```

`weight_pct` uses percentage points: `4.25` means 4.25%, not 0.0425. Provide the issuer/factsheet or provider name and the holdings as-of date; a URL is optional for a local factsheet file. The overlap score is the sum of the smaller reported weights across shared constituents. It is a useful overlap proxy only when both files represent sufficiently complete fund holdings with comparable snapshot dates. The application displays total weight coverage and flags partial snapshots; it never fabricates holdings or scales partial files to look complete.

## Scheduled market-data ingestion

The worker refreshes the currently held symbols, watchlist symbols and default benchmark symbols (`^GSPC,^STOXX50E,^IXIC`), stores quotes and adjusted historical prices in SQLite, and records a run log. It also saves a daily PLN valuation snapshot if the portfolio valuation is complete.

For local VS Code + WSL2, start the app with `./start.sh`, then open a **second terminal** and run:

```bash
./scripts/run_market_data_worker.sh
```

Use `./scripts/run_market_data_worker.sh --once` for a one-off job. The persistent worker runs at startup and then every 24 hours by default. Configure these values in `.env`:

```dotenv
ETF_MARKET_DATA_REFRESH_SECONDS=86400
ETF_MARKET_DATA_HISTORY_PERIOD=5y
ETF_MARKET_DATA_EXTRA_TICKERS=^GSPC,^STOXX50E,^IXIC
```

Create the local environment file if needed (`cp .env.example .env`), then run `docker compose up -d --build`. This starts both the terminal and the background worker and shares the same persistent volume. Do not run two workers against the same database unless that is intentional; SQLite supports concurrent access, but duplicate refresh jobs waste provider requests.

## Optional AI analyst

Add a key to `.env`:

```dotenv
OPENROUTER_API_KEY=""
OPENROUTER_MODEL=openrouter/free
```

The AI request happens only after you click Analyze. The selected prompt and current portfolio summary/quote information are sent to OpenRouter. Do not include secrets or sensitive personal details in prompts. Model availability and pricing depend on the provider.

## Recommended portfolio setup

1. Start in **Investments** to enter current units and average acquisition price. You can add multiple ETFs, edit opening-only positions in the management grid, set quantity to zero to remove a simple position, or use the explicit remove action to erase all records for a ticker.
2. Duplicate tickers are blocked in add/import flows to prevent accidental double-counting. Positions with BUY/SELL history are protected from simple editing and must be changed through **Transactions**. The explicit full-remove action deletes every transaction for that ticker and should be used only when you intend to erase its portfolio history.
3. Use **Transactions** to record future buys, sells, dividends and fees. Use the transaction date that matches the source record.
4. Add a preferred display name and optional ISIN/exchange in **ETF Research**. Custom labels are stored separately from quote-provider names.
5. Add symbols to **Watchlist** to track them without creating a holding. Choose a quick-add preset or enter a provider symbol such as `EURPLN=X`, `BTC-USD`, `GC=F`, `^TNX` or `ZN=F`. Quote availability depends on the market-data provider. Note that `^TNX` tracks the US 10-year Treasury yield index, while `ZN=F` represents 10-year Treasury note futures; neither is the same as a specific cash Treasury bond price.
6. Refresh market data and review quote status, price date and error fields before relying on valuations.
7. Back up the database and periodically export the ledger CSV.

## Accounting and model limitations

- **Cost basis:** weighted-average book cost; this is not tax-lot accounting and does not calculate local tax obligations.
- **XIRR:** approximate money-weighted return from OPENING/BUY/SELL/dividend cash flows and current ETF value as terminal value, with historical FX conversion where available. It excludes cash balances, taxes and other assets; missing FX prevents a result. It is not a broker-verified account return.
- **Snapshots:** point-in-time ETF market value only. They do not themselves reconstruct historical holdings or external cash flows; the Performance workspace separately builds a cash-excluded ETF-holdings return proxy from adjusted prices and the ledger.
- **Historical risk:** per-instrument price history, not a holdings-weighted portfolio backtest. Return and dividend treatment depends on provider data and adjustment settings.
- **FX:** Frankfurter/ECB reference rates are not executable exchange rates. Some currency/date pairs may be unavailable.
- **ETF research:** instrument metadata is user-managed. Pairwise constituent overlap is available when you import reliable, dated holdings snapshots; no holdings are bundled or fabricated. Sector/geographic look-through visuals still require a suitable constituent dataset with sector/country attributes.
- **Forecast Lab:** modelled outcomes under fixed assumptions, not predictions or confidence guarantees. Current model does not fully model taxes, TER, trading costs, fat tails or changing market regimes.
- **Market data:** Yahoo Finance via `yfinance` is a convenient research source, not a guaranteed real-time or licensed institutional feed. Review provider terms and use a licensed provider for commercial/production requirements.

## Deployment and security

Hosted deployments use managed PostgreSQL with a shared authentication table and a separate PostgreSQL schema per user. Accounts are created by an administrator; there is no public self-registration. Passwords are salted PBKDF2-HMAC-SHA256 hashes, and generated temporary passwords must be changed on first login. Hosted mode refuses to access portfolio tables without a validated tenant schema. Each new hosted account starts with an empty ledger; SQLite records are not imported automatically.

For the full GitHub + Streamlit Community Cloud setup, required secrets, clean-start behavior, per-user isolation, and database backup/recovery instructions, follow [`docs/STREAMLIT_DEPLOYMENT.md`](docs/STREAMLIT_DEPLOYMENT.md). Never commit `.env`, database URLs, secrets, SQLite files, PostgreSQL dumps, or investment exports. Keep the original local database/archive separate from the new hosted database.

Local development is still supported through `./start.sh`, which explicitly opts into local single-user SQLite mode when `DATABASE_URL` is not set. This bypasses authentication and is **not suitable for a public deployment**.

## Backups and automated checks

Create a consistent live SQLite backup from the project root when using local development:

```bash
./scripts/backup_db.sh
```

Set `ETF_APP_DB_PATH` or `ETF_APP_DATA_DIR` to back up a non-default database, and `ETF_BACKUP_DIR` to change the output directory. Keep backup files outside version control and test restore procedures periodically. A GitHub Actions workflow compiles the Python modules and runs unit tests on pushes and pull requests.

For hosted PostgreSQL backups and recovery drills, use `scripts/backup_postgres.sh` and `scripts/restore_postgres.sh` as described in the deployment guide.
