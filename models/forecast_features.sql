{{ config(materialized='table') }}

WITH base AS (
    SELECT *
    FROM {{ ref('cleaned_data') }}
),

features AS (
    SELECT 
        "DATETIME",
        "TOTAL_WATT_HOURS",
        EXTRACT(HOUR FROM "DATETIME") AS hour_of_day,
        EXTRACT(DOW FROM "DATETIME") AS day_of_week,

        CASE 
            WHEN EXTRACT(DOW FROM "DATETIME") IN (0,6) THEN 1
            ELSE 0
        END AS is_weekend

    FROM base
    WHERE "TOTAL_WATT_HOURS" > 0
)


SELECT *
FROM features
