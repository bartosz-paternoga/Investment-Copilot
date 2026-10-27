# Changelog

## 1.3.3 — 2026-10-27
- Fixed the local-development banner by installing CSS before rendering the banner and forcing high-contrast foreground colors with explicit opacity/text-fill overrides.
- Reworked button styling: primary actions now use a brighter emerald CTA with white text; secondary actions use a white surface, dark readable text, and a subtle green hover state.
- Strengthened selectors for Streamlit primary and form-submit buttons to reduce style drift across Streamlit component versions.

## 1.3.0 — 2026-10-27
- Expanded Instrument Watchlist beyond equities with quick-add presets for FX pairs, cryptocurrencies, commodities, US Treasury yields/futures, bond ETFs, major indices and volatility.
- Added clearer market-symbol guidance and descriptions, while preserving free-form provider symbols and the existing quote/history cache.
- Clarified that the US 10-year Treasury yield index (^TNX) is a yield indicator, not the Treasury bond price; included ZN=F for 10-year Treasury note futures.

## 1.2.0 — 2026-10-27
- Renamed the user-facing product to Investment Pilot.
- Made Overview the first and default workspace page.
- Fixed responsive hero layout so the Refresh prices action no longer overlaps or clips on narrow hosted screens.

# Changelog

## 1.3.2 — 2026-10-27
- Improved contrast and readability for the local-development notice and primary action buttons across Streamlit versions.
- Fixed responsive hero text sizing and wrapping to prevent the portfolio subtitle from clipping on narrow screens.
- Updated visible product branding from Investment Pilot to Investment Copilot.
- Refined primary button hover/focus states and green accent styling for a cleaner, modern portfolio-dashboard experience.


## 1.3.1 — 2026-10-27
- Added hosted PostgreSQL support while preserving the local SQLite adapter for explicitly enabled single-user development.
- Added administrator-provisioned accounts, salted PBKDF2 password hashes, temporary passwords with first-login reset, account deactivation, and per-account PostgreSQL schemas.
- Made hosted startup fail closed if `DATABASE_URL` or an authenticated tenant context is missing.
- Added Streamlit Community Cloud deployment documentation and PostgreSQL backup/restore scripts; hosted startup intentionally creates a clean database and never imports legacy SQLite holdings.
- Extended CI compile checks to include authentication and database modules.


## 1.1.0 — 2026-10-27

### Changed
- Rebranded the application as **Adatch Investor Portfolio** while keeping the existing ledger and analytics modules.
- Replaced the dark terminal skin with a light, modern portfolio interface: lighter canvas, white surfaces, rounded metric cards, calmer typography, and a restrained green accent system.
- Updated Plotly charts to use a light theme to match the rest of the UI.
- Renamed the main portfolio workspace to **Investments** and refined sidebar branding.

### Added
- Search and performance-based filters for holdings (today's gainers/losers and profitable/loss-making positions).
- CSV export for the currently filtered holdings grid.
- KPI deltas for estimated daily return and unrealized return on cost basis when the underlying data is complete.
- Native Streamlit light-theme configuration and disabled usage-stat collection by default.

### Preserved
- Existing transaction accounting, market-data cache, price/FX warnings, benchmark analytics, research tools, risk analysis, watchlist, forecast lab, and data-operations workflow.

This project follows Semantic Versioning (`MAJOR.MINOR.PATCH`). The version in `VERSION` is the single runtime version source.

## 1.0.1 — 2026-10-27

### Fixed
- Load cached/latest quotes on the default **Quick holdings** portfolio home, so its headline valuation and allocation do not remain blank until the user opens another workspace or presses refresh.
- Resolve the currency for manually priced buys/sales from valid quote metadata, then the latest transaction currency, then the selected reporting currency. This avoids silently assuming EUR when a quote is unavailable and avoids passing an empty currency that the ledger would normalize to PLN.

### Added
- Display the current app version in the sidebar.
- Add regression tests for trade-currency resolution.

## 1.0.0 — baseline

Initial versioned baseline for the original portfolio analytics feature set. See `README.md` for capabilities, setup instructions, accounting boundaries, data provenance, and deployment guidance.
