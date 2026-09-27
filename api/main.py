"""Phase 9: FastAPI app. A thin layer over analytics/, monte_carlo/,
ml/, and mlops/ -- see each router module for which real function every
endpoint delegates to.

Run with: uvicorn api.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI

from api.routers import forecast, model, monte_carlo, pipeline, portfolio

app = FastAPI(title="Portfolio Intelligence API")

app.include_router(portfolio.router)
app.include_router(monte_carlo.router)
app.include_router(forecast.router)
app.include_router(model.router)
app.include_router(pipeline.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
