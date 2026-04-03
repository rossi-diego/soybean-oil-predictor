-- Gold layer: latest feature vector for real-time predictions.
-- Returns the most recent row from the training features table.

{{ config(materialized='view') }}

select *
from {{ ref('gold_training_features') }}
order by trade_date desc
limit 1
