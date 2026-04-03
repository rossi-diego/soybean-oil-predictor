-- Test: BOC1 target column must never be null in gold layer.
select trade_date
from {{ ref('gold_training_features') }}
where boc1 is null
