from datetime import date, datetime

from pydantic import BaseModel


class DataQualityReportResponse(BaseModel):
    """Mirrors ingestion.cleaning.DataQualityReport.as_dict() exactly --
    same field names, same shape, no invented summary format."""

    rows_in: int
    rows_cleaned: int
    duplicates_dropped: int
    near_duplicates_flagged: int
    values_imputed: int
    anomalies_flagged: int
    rows_unresolved: int
    split_adjustments_applied: int
    notes: list[str]


class RejectedRow(BaseModel):
    row_index: int
    reason: str


class UploadResponse(BaseModel):
    portfolio_id: int
    pipeline_run_id: str
    status: str
    valid_row_count: int
    rejected_row_count: int
    rejected_rows: list[RejectedRow]
    data_quality_report: DataQualityReportResponse


class HoldingWeight(BaseModel):
    ticker: str
    weight: float


class PortfolioOverviewResponse(BaseModel):
    portfolio_id: int
    as_of_date: date
    invested_capital: float
    market_value: float
    absolute_return: float
    pct_return: float
    holdings: list[HoldingWeight]


class PortfolioSummary(BaseModel):
    """One row in the portfolio switcher/picker -- enough to identify
    and distinguish saved portfolios without fetching each one's full
    overview. market_value/holdings_count are null for a portfolio
    that uploaded but has no analytics yet (e.g. every row rejected)."""

    portfolio_id: int
    name: str
    created_at: datetime
    market_value: float | None
    holdings_count: int | None


class PortfolioListResponse(BaseModel):
    portfolios: list[PortfolioSummary]


class DeletePortfolioResponse(BaseModel):
    portfolio_id: int
    deleted: bool
