-- Seed reference dimensions: dim_currency, dim_date, dim_user, dim_portfolio.

INSERT INTO dim_currency (currency_code, currency_name, symbol, decimal_places) VALUES
    ('INR', 'Indian Rupee', '₹', 2),
    ('USD', 'US Dollar', '$', 2),
    ('EUR', 'Euro', '€', 2)
ON CONFLICT DO NOTHING;

-- dim_date: generate every calendar day for 2023-01-01 .. 2026-12-31.
-- is_trading_day approximated as weekday (Mon-Fri); a real NSE/BSE
-- holiday calendar is a documented follow-up (see 01_dimensions.sql).
INSERT INTO dim_date (date_key, full_date, day_of_month, day_of_week, day_name,
                       week_of_year, month_num, month_name, quarter, year, is_trading_day)
SELECT
    (to_char(d, 'YYYYMMDD'))::INT,
    d,
    extract(day FROM d)::SMALLINT,
    extract(isodow FROM d)::SMALLINT,
    to_char(d, 'Day'),
    extract(week FROM d)::SMALLINT,
    extract(month FROM d)::SMALLINT,
    to_char(d, 'Month'),
    extract(quarter FROM d)::SMALLINT,
    extract(year FROM d)::SMALLINT,
    extract(isodow FROM d) < 6
FROM generate_series('2023-01-01'::DATE, '2026-12-31'::DATE, '1 day'::INTERVAL) AS d
ON CONFLICT DO NOTHING;

INSERT INTO dim_user (user_id, email, display_name)
VALUES (1, 'arihanandotra03@gmail.com', 'Arihan Andotra')
ON CONFLICT DO NOTHING;

INSERT INTO dim_portfolio (portfolio_id, user_id, portfolio_name, base_currency_code)
VALUES (1, 1, 'Primary Portfolio', 'INR')
ON CONFLICT DO NOTHING;
