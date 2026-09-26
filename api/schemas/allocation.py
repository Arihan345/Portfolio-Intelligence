from pydantic import BaseModel


class AllocationResponse(BaseModel):
    portfolio_id: int
    asset_allocation: dict[str, float]
    sector_allocation: dict[str, float]
    currency_allocation: dict[str, float]
    cash_allocation: dict[str, float]
    largest_holding_pct: float
    top_2_pct: float
    herfindahl_hirschman_index: float
