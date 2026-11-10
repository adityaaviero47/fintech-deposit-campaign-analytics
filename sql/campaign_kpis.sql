-- Campaign KPIs on the table `calls` (one row per client, where campaign = calls made to that client, y = 1 if subscribed)

-- 1. headline KPIs
SELECT COUNT(*) AS clients, SUM(y) AS conversions, ROUND(AVG(y), 4) AS conversion_rate,
       SUM(campaign) AS total_calls, ROUND(100.0 * SUM(y) / SUM(campaign), 3) AS conversions_per_100_calls
FROM calls;

-- 2. conversion by number of calls made to the client
WITH b AS (
  SELECT y,
         CASE WHEN campaign <= 3 THEN CAST(campaign AS TEXT) WHEN campaign <= 5 THEN '4-5'
              WHEN campaign <= 10 THEN '6-10' ELSE '11+' END AS bucket,
         CASE WHEN campaign <= 3 THEN campaign WHEN campaign <= 5 THEN 4 WHEN campaign <= 10 THEN 6 ELSE 11 END AS sort_key
  FROM calls)
SELECT bucket, COUNT(*) AS clients, ROUND(AVG(y), 4) AS conversion FROM b GROUP BY bucket, sort_key ORDER BY sort_key;

-- 3. conversion by channel and by the outcome of the previous campaign, with lift vs the overall rate
WITH overall AS (SELECT AVG(y) AS r FROM calls)
SELECT 'channel' AS cut, contact AS segment, COUNT(*) AS clients, ROUND(AVG(y), 4) AS conversion,
       ROUND(AVG(y) / (SELECT r FROM overall), 2) AS lift FROM calls GROUP BY contact
UNION ALL
SELECT 'previous outcome', poutcome, COUNT(*), ROUND(AVG(y), 4), ROUND(AVG(y) / (SELECT r FROM overall), 2) FROM calls GROUP BY poutcome;

-- 4. a cap of K calls per client: calls saved and conversions at risk (K = 3 and 5)
WITH caps(k) AS (VALUES (3), (5))
SELECT k AS cap, SUM(MAX(campaign - k, 0)) AS calls_saved,
       SUM(CASE WHEN campaign > k THEN y ELSE 0 END) AS conversions_at_risk,
       ROUND(100.0 * (SUM(y) - SUM(CASE WHEN campaign > k THEN y ELSE 0 END)) / (SUM(campaign) - SUM(MAX(campaign - k, 0))), 3) AS conv_per_100_calls_after_cap
FROM calls CROSS JOIN caps GROUP BY k;
