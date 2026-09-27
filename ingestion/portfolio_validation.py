"""Portfolio transaction CSV ingestion + validation.

Reads a raw transaction CSV and splits every row into valid_rows or
rejected_rows. Rows are never silently dropped: each rejection carries
a specific, actionable reason string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from ingestion.providers.base import SplitEvent

REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "transaction_type",
    "quantity",
    "price",
    "fees",
    "tax",
    "currency",
]

ALLOWED_TRANSACTION_TYPES = {"BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAWAL"}

# Plausible exchange-symbol shape: ticker body + a known exchange suffix.
TICKER_PATTERN = re.compile(r"^[A-Z0-9&\-]+\.(NS|BO)$")

# Genuinely ambiguous date formats we refuse to silently coerce, e.g. 01/02/2024
AMBIGUOUS_DATE_PATTERN = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")


@dataclass
class ValidationResult:
    valid_rows: pd.DataFrame
    rejected_rows: pd.DataFrame
    # Audit trail of quantity/price rewrites applied by
    # apply_split_adjustments below, one entry per adjusted row --
    # informational, distinct from rejected_rows (nothing here was
    # rejected) and from clean_and_standardize's anomaly notes.
    split_adjustment_notes: list[str] = field(default_factory=list)


def _parse_date_strict(raw: str) -> tuple[date | None, str | None]:
    """Returns (parsed_date, error). Rejects ambiguous slash-separated
    dates like 01/02/2024 rather than guessing month/day order.
    """
    raw = str(raw).strip()
    if AMBIGUOUS_DATE_PATTERN.match(raw):
        return None, f"ambiguous date format '{raw}' (use ISO YYYY-MM-DD)"
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date(), None
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw).date(), None
    except ValueError:
        return None, f"unparseable date '{raw}'"


def _validate_row(row: pd.Series, today: date) -> str | None:
    """Returns a rejection reason string, or None if the row is valid."""

    txn_type = str(row["transaction_type"]).strip().upper()
    if txn_type not in ALLOWED_TRANSACTION_TYPES:
        return (
            f"transaction_type '{row['transaction_type']}' not in "
            f"allowed set {sorted(ALLOWED_TRANSACTION_TYPES)}"
        )

    parsed_date, date_err = _parse_date_strict(row["date"])
    if date_err:
        return date_err
    if parsed_date > today:
        return f"future-dated transaction: {parsed_date} > today ({today})"

    ticker = str(row["ticker"]).strip().upper()
    if not TICKER_PATTERN.match(ticker):
        return (
            f"ticker '{row['ticker']}' does not match plausible exchange "
            "symbol shape (expected suffix .NS or .BO)"
        )

    if txn_type in ("BUY", "SELL"):
        try:
            quantity = float(row["quantity"])
        except (ValueError, TypeError):
            return f"quantity '{row['quantity']}' is not numeric"
        if quantity <= 0:
            return f"quantity must be > 0 for {txn_type}, got {quantity}"

        try:
            price = float(row["price"])
        except (ValueError, TypeError):
            return f"price '{row['price']}' is not numeric"
        if price <= 0:
            return f"price must be > 0 for {txn_type}, got {price}"

    return None


def _base_symbol(ticker: str) -> str:
    """Same exchange-suffix-stripped grouping dbt's int_daily_holdings
    uses to net a company's NSE/BSE trades as one position (they're
    fungible in demat) -- kept in sync deliberately, since this
    function exists specifically to reject the one thing that grouping
    can't fix: a genuine oversell (see _reject_oversold_sells below).
    """
    return ticker.strip().upper().split(".")[0]


def apply_split_adjustments(
    df: pd.DataFrame, splits_by_ticker: dict[str, list[SplitEvent]]
) -> tuple[pd.DataFrame, list[str]]:
    """Rewrites BUY/SELL quantity/price for transactions dated BEFORE a
    real stock split, so every transaction ends up expressed in TODAY's
    share-count terms -- comparable across time the same way a
    post-split transaction already is.

    Real incident this fixes: PCJEWELLER.NS underwent a real 1:10 split
    on 2024-12-16, between a 5-share BUY (2024-12-11/12) and what was
    actually that SAME position's full exit, recorded post-split as a
    50-share SELL (2025-12-29). Without adjustment, the pre-split BUY
    reads as only 5 shares against a 50-share SELL -- an impossible
    45-share oversell that _reject_oversold_sells below would (and did,
    before this fix) reject as bad data, when the trade was entirely
    legitimate: 5 shares x a 1:10 split = 50 shares, exactly matching
    the SELL.

    For a transaction dated before split_date, quantity is multiplied
    and price divided by the split's ratio (the yfinance convention:
    10.0 = 1 old share -> 10 new shares), keeping total transaction
    value (quantity * price) invariant. Multiple splits after a
    transaction's date compound (product of all applicable ratios).
    fees/tax are absolute currency amounts and are never adjusted.

    Splits are looked up per BASE symbol (exchange suffix stripped,
    same grouping _reject_oversold_sells/int_daily_holdings.sql use),
    since a split is a real corporate event affecting the whole company
    regardless of which exchange a given trade executed on -- the union
    of whatever split events any of that company's exchange-listed
    tickers reported (e.g. if only "PCJEWELLER.NS" has split data but a
    trade was recorded as "PCJEWELLER.BO", the same real split still
    applies to it).

    Returns the adjusted DataFrame (a copy; unaffected rows are
    untouched) plus one human-readable audit note per adjusted row --
    informational, never a rejection, and distinct from
    clean_and_standardize's anomaly/duplicate notes.
    """
    if not splits_by_ticker:
        return df, []

    splits_by_base: dict[str, list[tuple[date, float]]] = {}
    for ticker, events in splits_by_ticker.items():
        bucket = splits_by_base.setdefault(_base_symbol(ticker), [])
        bucket.extend((ev.split_date, ev.ratio) for ev in events)

    if not any(splits_by_base.values()):
        return df, []

    df = df.copy()
    notes: list[str] = []
    for i in df.index:
        txn_type = str(df.at[i, "transaction_type"]).strip().upper()
        if txn_type not in ("BUY", "SELL"):
            continue

        base = _base_symbol(str(df.at[i, "ticker"]))
        events = splits_by_base.get(base)
        if not events:
            continue

        parsed_date, date_err = _parse_date_strict(df.at[i, "date"])
        if date_err:
            continue  # unparseable -- the structural pass will reject it anyway

        try:
            quantity = float(df.at[i, "quantity"])
            price = float(df.at[i, "price"])
        except (ValueError, TypeError):
            continue  # non-numeric -- the structural pass will reject it anyway

        applicable = sorted(d_r for d_r in events if d_r[0] > parsed_date)
        if not applicable:
            continue

        multiplier = 1.0
        for _split_date, ratio in applicable:
            multiplier *= ratio
        if multiplier == 1.0:
            continue

        new_quantity = quantity * multiplier
        new_price = price / multiplier
        df.at[i, "quantity"] = str(new_quantity)
        df.at[i, "price"] = str(new_price)

        split_desc = ", ".join(f"1:{ratio:g} on {d}" for d, ratio in applicable)
        notes.append(
            f"row {i} ({df.at[i, 'ticker']}, {df.at[i, 'date']}): quantity adjusted for a real "
            f"split ({split_desc}) -- {quantity:g} -> {new_quantity:g} shares "
            f"(price {price:g} -> {new_price:g})"
        )

    return df, notes


def _reject_oversold_sells(df: pd.DataFrame, valid_mask: list[bool], reasons: list[str | None]) -> None:
    """Second validation pass, after the structural per-row checks
    above: walks the currently-valid BUY/SELL rows in chronological
    order per underlying company (netted across .NS/.BO, same grouping
    int_daily_holdings.sql uses) and rejects any SELL whose quantity
    exceeds what's actually available from prior BUYs.

    Real incident this closes: a real 30-ticker upload had a genuine
    bad row (PCJEWELLER.NS: 5 shares ever bought, a single SELL of 50)
    that passed every per-row structural check (positive quantity,
    valid ticker/date/type) and reached the warehouse, where it read as
    a permanently impossible -45 share holding and failed dbt's hard
    quantity_held range test for the WHOLE build -- taking every
    downstream mart down with it (mart_asset_performance, mart_
    allocation, etc. all SKIPPED). Per this project's own Phase 2
    design (one bad row is rejected individually; it never fails the
    whole batch), that row belongs here, caught and excluded before it
    ever reaches the warehouse -- not discovered later as a build-
    breaking hard-test failure. Mutates valid_mask/reasons in place.
    """
    candidates: list[tuple[date, int, str, str, float]] = []
    for i, is_valid in enumerate(valid_mask):
        if not is_valid:
            continue
        row = df.iloc[i]
        txn_type = str(row["transaction_type"]).strip().upper()
        if txn_type not in ("BUY", "SELL"):
            continue
        parsed_date, date_err = _parse_date_strict(row["date"])
        if date_err:
            continue  # already rejected by the structural pass
        try:
            quantity = float(row["quantity"])
        except (ValueError, TypeError):
            continue  # already rejected by the structural pass
        candidates.append((parsed_date, i, _base_symbol(str(row["ticker"])), txn_type, quantity))

    candidates.sort(key=lambda c: (c[0], c[1]))

    running: dict[str, float] = {}
    for parsed_date, i, base_symbol, txn_type, quantity in candidates:
        current = running.get(base_symbol, 0.0)
        if txn_type == "BUY":
            running[base_symbol] = current + quantity
            continue
        if quantity > current + 1e-9:
            valid_mask[i] = False
            reasons[i] = (
                f"SELL quantity {quantity:g} for '{base_symbol}' exceeds the {current:g} "
                f"share(s) actually available from prior BUYs as of {parsed_date} "
                "(checked net of .NS/.BO for the same company) -- oversold/implausible position"
            )
        else:
            running[base_symbol] = current - quantity


def validate_csv(
    path: str,
    today: date | None = None,
    splits: dict[str, list[SplitEvent]] | None = None,
) -> ValidationResult:
    """`splits` (optional): real stock-split events per ticker, e.g. from
    YFinanceProvider.fetch_corporate_actions -- applied BEFORE the
    oversold check below, so a pre-split BUY is compared against a
    post-split SELL on the same, correct share-count basis instead of
    being flagged as an impossible oversell. Omitted (the default) means
    no adjustment is applied -- callers that can't reach a market-data
    provider (e.g. pure offline tests) get the original, unadjusted
    behavior, never a silent network dependency inside this function.
    """
    today = today or date.today()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    split_notes: list[str] = []
    if splits:
        df, split_notes = apply_split_adjustments(df, splits)

    valid_mask = []
    reasons = []
    for _, row in df.iterrows():
        reason = _validate_row(row, today)
        valid_mask.append(reason is None)
        reasons.append(reason)

    _reject_oversold_sells(df, valid_mask, reasons)

    df = df.copy()
    df["rejection_reason"] = reasons
    valid_rows = df[[m for m in valid_mask]].drop(columns=["rejection_reason"]).reset_index(drop=True)
    rejected_rows = df[[not m for m in valid_mask]].reset_index(drop=True)

    return ValidationResult(valid_rows=valid_rows, rejected_rows=rejected_rows, split_adjustment_notes=split_notes)
