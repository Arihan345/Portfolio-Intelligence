"""Portfolio transaction CSV ingestion + validation.

Reads a raw transaction CSV and splits every row into valid_rows or
rejected_rows. Rows are never silently dropped: each rejection carries
a specific, actionable reason string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

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


def validate_csv(path: str, today: date | None = None) -> ValidationResult:
    today = today or date.today()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")

    valid_mask = []
    reasons = []
    for _, row in df.iterrows():
        reason = _validate_row(row, today)
        valid_mask.append(reason is None)
        reasons.append(reason)

    df = df.copy()
    df["rejection_reason"] = reasons
    valid_rows = df[[m for m in valid_mask]].drop(columns=["rejection_reason"]).reset_index(drop=True)
    rejected_rows = df[[not m for m in valid_mask]].reset_index(drop=True)

    return ValidationResult(valid_rows=valid_rows, rejected_rows=rejected_rows)
