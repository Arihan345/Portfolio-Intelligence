-- Q1: total portfolio value on 2024-06-01
SELECT pv.date_key, d.full_date, pv.market_value_inr, pv.invested_capital_inr
FROM fact_portfolio_value pv
JOIN dim_date d ON d.date_key = pv.date_key
WHERE pv.portfolio_id = 1 AND d.full_date = '2024-06-01';

-- Q2: current (latest) holding quantity per asset
SELECT a.ticker, h.quantity_held, h.cost_basis_inr, d.full_date AS as_of
FROM fact_holdings h
JOIN dim_asset a ON a.asset_key = h.asset_key
JOIN dim_date d ON d.date_key = h.date_key
WHERE h.portfolio_id = 1
  AND h.date_key = (SELECT MAX(date_key) FROM fact_holdings WHERE portfolio_id = 1)
ORDER BY a.ticker;

-- Q3: portfolio value time series (sanity check the curve)
SELECT d.full_date, pv.market_value_inr, pv.invested_capital_inr, r.daily_return, r.cumulative_return
FROM fact_portfolio_value pv
JOIN dim_date d ON d.date_key = pv.date_key
LEFT JOIN fact_portfolio_returns r ON r.portfolio_id = pv.portfolio_id AND r.date_key = pv.date_key
WHERE pv.portfolio_id = 1
ORDER BY d.full_date
LIMIT 5;

-- Q4: transaction audit trail with lineage
SELECT a.ticker, t.transaction_type, t.quantity, t.price, t.fees, t.tax,
       t.source_system, t.ingested_at, t.pipeline_run_id
FROM fact_transactions t
JOIN dim_asset a ON a.asset_key = t.asset_key
WHERE t.portfolio_id = 1
ORDER BY t.date_key;

-- Q5: dividends received (via the fact_dividends view)
SELECT a.ticker, fd.dividend_amount_inr, d.full_date
FROM fact_dividends fd
JOIN dim_asset a ON a.asset_key = fd.asset_key
JOIN dim_date d ON d.date_key = fd.date_key
WHERE fd.portfolio_id = 1;

-- Q6: pipeline monitoring surface (Streamlit Monitoring page source)
SELECT run_id, dag_id, status, rows_processed, started_at, completed_at, duration_seconds
FROM pipeline_runs
ORDER BY started_at DESC;
