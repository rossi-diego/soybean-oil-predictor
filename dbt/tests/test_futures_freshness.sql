-- Test: silver futures data must have records within the last 7 days.
-- Fails if the most recent trade date is older than 7 days.
select 1
from {{ ref('silver_futures_daily') }}
having max(trade_date) < current_date - interval '7 days'
