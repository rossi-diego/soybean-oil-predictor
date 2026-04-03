-- Test: no duplicate trade dates in the gold training features.
select trade_date, count(*) as cnt
from {{ ref('gold_training_features') }}
group by trade_date
having count(*) > 1
