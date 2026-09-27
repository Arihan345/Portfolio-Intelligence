"""Cleaning + standardization layer.

Sits between validated rows and warehouse-ready data. Distinct from
validation: validation rejects rows that are structurally wrong;
cleaning normalizes, deduplicates, imputes, and flags rows that are
structurally valid but need business-rule handling before landing in
the warehouse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# Explicit alias mapping for known ticker suffix variants -> canonical form.
TICKER_ALIASES = {
    "TCS.NSE": "TCS.NS",
    "RELIANCE.NSE": "RELIANCE.NS",
    "INFY.NSE": "INFY.NS",
}

PRICE_ANOMALY_THRESHOLD = 0.15  # ±15% day-over-day move


@dataclass
class DataQualityReport:
    rows_in: int = 0
    rows_cleaned: int = 0
    duplicates_dropped: int = 0
    near_duplicates_flagged: int = 0
    values_imputed: int = 0
    anomalies_flagged: int = 0
    rows_unresolved: int = 0
    # Rows whose quantity/price were rewritten by
    # ingestion.portfolio_validation.apply_split_adjustments for a real
    # stock split -- informational, never a rejection or an anomaly (a
    # split-caused price drop is not a data problem). Kept as its own
    # counter, distinct from anomalies_flagged, so a real corporate
    # action is never presented as "concerning volatility" on the Data
    # Quality page.
    split_adjustments_applied: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_cleaned": self.rows_cleaned,
            "duplicates_dropped": self.duplicates_dropped,
            "near_duplicates_flagged": self.near_duplicates_flagged,
            "values_imputed": self.values_imputed,
            "anomalies_flagged": self.anomalies_flagged,
            "rows_unresolved": self.rows_unresolved,
            "split_adjustments_applied": self.split_adjustments_applied,
            "notes": self.notes,
        }


def _normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    return TICKER_ALIASES.get(ticker, ticker)


def _infer_currency(ticker: str) -> str | None:
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "INR"
    return None


def clean_and_standardize(
    df: pd.DataFrame,
    price_anomaly_threshold: float = PRICE_ANOMALY_THRESHOLD,
    split_adjustment_notes: list[str] | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """split_adjustment_notes (optional): audit notes produced by
    ingestion.portfolio_validation.apply_split_adjustments, surfaced
    here as their own counter/notes -- distinct from anomalies_flagged,
    since a real stock split's price change is not the kind of
    "concerning" price-anomaly this function's own detection below
    flags (df is already split-adjusted by the time this runs, so that
    detection naturally no longer misfires on it either)."""
    report = DataQualityReport(rows_in=len(df))
    if split_adjustment_notes:
        report.split_adjustments_applied = len(split_adjustment_notes)
        report.notes.extend(f"[split-adjusted] {n}" for n in split_adjustment_notes)
    df = df.copy()

    # --- Ticker normalization ---
    df["ticker"] = df["ticker"].apply(_normalize_ticker)

    # --- Date normalization (already ISO-parseable post-validation) ---
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # --- Numeric coercion ---
    for col in ("quantity", "price", "fees", "tax"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Missing value handling: fees/tax -> 0 (documented business rule) ---
    for col in ("fees", "tax"):
        missing = df[col].isna().sum()
        if missing:
            report.values_imputed += int(missing)
            report.notes.append(f"imputed {missing} missing '{col}' as 0")
        df[col] = df[col].fillna(0)

    # --- Missing currency: infer from ticker suffix, else flag unresolved ---
    df["currency_original"] = df["currency"]
    missing_currency_mask = df["currency"].isna() | (df["currency"].str.strip() == "")
    for idx in df[missing_currency_mask].index:
        inferred = _infer_currency(df.at[idx, "ticker"])
        if inferred:
            df.at[idx, "currency"] = inferred
            report.values_imputed += 1
            report.notes.append(
                f"row {idx}: inferred missing currency as {inferred} from ticker suffix"
            )
        else:
            report.rows_unresolved += 1
            report.notes.append(
                f"row {idx}: currency missing and unresolvable (ticker '{df.at[idx, 'ticker']}' has no known suffix)"
            )

    # --- Currency conversion: convert non-INR to INR, keep original alongside ---
    # Documented FX source: static rate table for now (placeholder until a
    # live FX provider is wired in); original currency/amount is preserved.
    FX_TO_INR = {"INR": 1.0, "USD": 83.0, "EUR": 90.0}
    df["price_inr"] = df.apply(
        lambda r: r["price"] * FX_TO_INR.get(r["currency"], float("nan")), axis=1
    )
    unresolved_fx = df["price_inr"].isna() & df["price"].notna()
    if unresolved_fx.any():
        n = int(unresolved_fx.sum())
        report.rows_unresolved += n
        report.notes.append(f"{n} rows have currency with no known FX rate to INR")

    # --- Duplicate detection ---
    exact_dup_mask = df.duplicated(
        subset=["date", "ticker", "transaction_type", "quantity", "price"], keep="first"
    )
    report.duplicates_dropped = int(exact_dup_mask.sum())
    if report.duplicates_dropped:
        report.notes.append(f"dropped {report.duplicates_dropped} exact duplicate rows")
    df = df[~exact_dup_mask].copy()

    near_dup_mask = df.duplicated(subset=["date", "ticker"], keep=False) & ~df.duplicated(
        subset=["date", "ticker", "transaction_type", "quantity", "price"], keep=False
    )
    df["near_duplicate_flag"] = near_dup_mask
    report.near_duplicates_flagged = int(near_dup_mask.sum())
    if report.near_duplicates_flagged:
        report.notes.append(
            f"flagged {report.near_duplicates_flagged} near-duplicate rows for manual review "
            "(same ticker+date, different quantity/price)"
        )

    # --- Price anomaly detection (day-over-day move beyond threshold) ---
    # Flag only, never reject; keep the row in the pipeline. Only BUY/SELL
    # rows carry an actual traded share price - DIVIDEND/DEPOSIT/WITHDRAWAL
    # rows reuse the 'price' column for unrelated amounts and must not be
    # chained into the same price series or every dividend looks like a
    # market crash.
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["price_anomaly_flag"] = False
    tradeable = df[df["transaction_type"].isin(["BUY", "SELL"])]
    for ticker, group in tradeable.groupby("ticker"):
        prev_price = None
        for idx in group.index:
            price = df.at[idx, "price"]
            if prev_price is not None and prev_price != 0 and pd.notna(price):
                pct_move = abs(price - prev_price) / prev_price
                if pct_move > price_anomaly_threshold:
                    df.at[idx, "price_anomaly_flag"] = True
                    report.anomalies_flagged += 1
                    report.notes.append(
                        f"row {idx} ({ticker}, {df.at[idx, 'date']}): price moved "
                        f"{pct_move:.1%} from {prev_price} to {price} (threshold "
                        f"{price_anomaly_threshold:.0%})"
                    )
            if pd.notna(price):
                prev_price = price

    report.rows_cleaned = len(df)
    return df, report
