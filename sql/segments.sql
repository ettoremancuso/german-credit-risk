-- Default rates by borrower segment.
-- Runs on SQLite; the notebook loads data/raw/germancredit.csv into a table called `loans`
-- and adds a 0/1 column `bad` (1 = default).

-- name: by_checking_account
SELECT status_of_existing_checking_account AS segment,
       COUNT(*)                            AS n,
       SUM(bad)                            AS defaults,
       ROUND(100.0 * AVG(bad), 1)          AS default_rate_pct
FROM loans
GROUP BY segment
ORDER BY default_rate_pct DESC;

-- name: by_duration_band
SELECT CASE
         WHEN duration_in_month <= 12 THEN '01-12 months'
         WHEN duration_in_month <= 24 THEN '13-24 months'
         WHEN duration_in_month <= 36 THEN '25-36 months'
         ELSE '37+ months'
       END                                 AS segment,
       COUNT(*)                            AS n,
       SUM(bad)                            AS defaults,
       ROUND(100.0 * AVG(bad), 1)          AS default_rate_pct,
       ROUND(AVG(credit_amount), 0)        AS avg_amount_dm
FROM loans
GROUP BY segment
ORDER BY segment;

-- name: by_purpose
SELECT purpose                             AS segment,
       COUNT(*)                            AS n,
       ROUND(100.0 * AVG(bad), 1)          AS default_rate_pct,
       ROUND(SUM(credit_amount) / 1000.0, 0) AS exposure_k_dm
FROM loans
GROUP BY segment
HAVING COUNT(*) >= 20
ORDER BY default_rate_pct DESC;
