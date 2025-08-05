{{ config(materialized='table') }}

WITH raw_data AS (
    SELECT *
    FROM {{ source('energy', 'raw_data') }}  -- Replace with your actual source table name
),

valid_data AS (
    SELECT *
    FROM raw_data
    WHERE "CHARGE_WATT_HOUR" >= 0
      AND "CHARGE_START_TIME_AT" <= "CHARGE_STOP_TIME_AT"
),

with_filled_end_time AS (
    SELECT
        *,
        CASE
            WHEN "CHARGE_STOP_TIME_AT" IS NULL
            THEN "CHARGE_START_TIME_AT" + ("CHARGE_DURATION_MINS"::int || ' minutes')::interval
            ELSE "CHARGE_STOP_TIME_AT"
        END AS computed_stop_time,
        -- Calculate energy per minute for distribution across hours
        "CHARGE_WATT_HOUR" / NULLIF("CHARGE_DURATION_MINS", 0) AS energy_per_minute
    FROM valid_data
),

-- Generate a series of hours covering our time range
hour_series AS (
    SELECT DISTINCT
        DATE_TRUNC('hour', hour_time) AS hour_start
    FROM (
        SELECT generate_series(
            (SELECT MIN(DATE_TRUNC('hour', "CHARGE_START_TIME_AT")) FROM with_filled_end_time),
            (SELECT MAX(DATE_TRUNC('hour', computed_stop_time)) FROM with_filled_end_time),
            '1 hour'::interval
        ) AS hour_time
    ) series
),

-- Calculate overlapping time and energy for each hour-session combination
hour_session_overlaps AS (
    SELECT
        h.hour_start AS "DATETIME",
        s."CHARGE_WATT_HOUR",
        
        -- For sessions contained entirely within an hour
        CASE WHEN 
            DATE_TRUNC('hour', s."CHARGE_START_TIME_AT") = DATE_TRUNC('hour', s.computed_stop_time)
            AND DATE_TRUNC('hour', s."CHARGE_START_TIME_AT") = h.hour_start
        THEN 
            s."CHARGE_WATT_HOUR"
        
        -- For sessions spanning multiple hours
        ELSE
            -- Calculate minutes of overlap between session and this hour
            EXTRACT(EPOCH FROM (
                LEAST(
                    h.hour_start + '1 hour'::interval, 
                    s.computed_stop_time
                ) - 
                GREATEST(
                    h.hour_start, 
                    s."CHARGE_START_TIME_AT"
                )
            )) / 60 * s.energy_per_minute
        END AS "TOTAL_WATT_HOURS"
    FROM hour_series h
    JOIN with_filled_end_time s
        ON h.hour_start <= DATE_TRUNC('hour', s.computed_stop_time)
        AND h.hour_start + '1 hour'::interval > s."CHARGE_START_TIME_AT"
    WHERE 
        -- Ensure there is actual overlap between the hour and session
        s.computed_stop_time > h.hour_start
        AND s."CHARGE_START_TIME_AT" < h.hour_start + '1 hour'::interval
),

aggregated AS (
    SELECT
        "DATETIME",
        SUM("TOTAL_WATT_HOURS") AS "TOTAL_WATT_HOURS"
    FROM hour_session_overlaps
    GROUP BY "DATETIME"
)

SELECT *
FROM aggregated
ORDER BY "DATETIME"