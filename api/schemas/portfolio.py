from datetime import date

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
