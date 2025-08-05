import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

import matplotlib.pyplot as plt

engine = create_engine("postgresql://dbtuser:dbtpass@postgres:5432/dbtdb")

st.title("Energy Forecast Dashboard")

df_cleaned = pd.read_sql("SELECT * FROM cleaned_data", engine)
df_forecast = pd.read_sql("SELECT * FROM forecast", engine)
df_test_predictions = pd.read_sql("SELECT * FROM test_predictions", engine)
df_model_metrics = pd.read_sql("SELECT * FROM model_metrics", engine)

selected_model = df_model_metrics.loc[df_model_metrics['selected_for_forecast'] == True, 'model'].values[0]

# Filter forecast data for the next 7 days
df_forecast.rename(columns={"ds": "t", "yhat": "Total_watt_hours"}, inplace=True)
next_7_days = df_forecast.tail(24*7)


# Plotting next 7 days forecast
st.subheader(f"Next 7 Days Forecast ({selected_model})")
st.write("This chart shows the forecast of the energy consumption for the next 7 days. The forecast tells, how much energy needs to be reserved for the upcoming sessions during the next 7 days.")
st.line_chart(next_7_days.set_index("t")["Total_watt_hours"])
# Add a forecast table
st.subheader("Forecast Table")
st.write(next_7_days[["t", "Total_watt_hours"]].set_index("t"))

# Plotting with the cleaned data

st.subheader("Actual Combined With The Forecast")
st.write("The model is trained on the cleaned data and the forecast is generated for the next 7 days. "\
         f"The model used for the forecast is {selected_model}.")
# Rename columns
df_cleaned.rename(columns={"DATETIME": "t", "TOTAL_WATT_HOURS": "Total_watt_hours"}, inplace=True)


# Add type column
df_cleaned["type"] = "Actual"
df_forecast["type"] = "Forecast"

# Combine dataframes
combined_df = pd.concat([df_cleaned, df_forecast], ignore_index=True)

st.line_chart(combined_df, x="t", y="Total_watt_hours", color="type")



# Bonus: Plotting the test predictions
st.subheader("Bonus: Model Monitoring")
st.write("This chart shows the predictions made on the test set.")

## Prepare actual values
df_actual = df_test_predictions[['ds','y']].drop_duplicates().copy()
df_actual.rename(columns={"ds": "t", "y": "Total_watt_hours"}, inplace=True)
df_actual["type"] = "Actual"
df_model_preds = df_test_predictions[['ds','y_pred','model']].copy()
df_model_preds.rename(columns={"ds": "t", "y_pred": "Total_watt_hours", "model":"type"}, inplace=True)

df_actual = df_test_predictions[['ds', 'y']].drop_duplicates()
df_actual.rename(columns={"ds": "t", "y": "Total_watt_hours"}, inplace=True)
df_actual["type"] = "Actual"

## Prepare model predictions
df_model_preds = df_test_predictions[['ds', 'y_pred', 'model']].copy()
df_model_preds.rename(columns={"ds": "t", "y_pred": "Total_watt_hours", "model": "type"}, inplace=True)

## List of available models
available_models = df_model_preds["type"].unique().tolist()

# User selects models to display, default is all
selected_models = st.multiselect(
    "Select models to compare with actuals:",
    options=available_models,
    default=available_models  # All models selected by default
)

## Filter selected models
filtered_model_preds = df_model_preds[df_model_preds["type"].isin(selected_models)]

## Combine actual and selected model predictions
combined_df = pd.concat([df_actual, filtered_model_preds])

## Plot
st.line_chart(
    combined_df.pivot(index="t", columns="type", values="Total_watt_hours")
)


# Model metrics table
st.subheader("Model Metrics")
st.write("This table shows the model metrics for the models used in the training phase. The forecast model is selected based on the rank score. " \
         "It's calculated by dividing RMSE by R^2 score. The lower the rank score, the better the model.")
st.write(df_model_metrics[["selected_for_forecast","model", "r2_score", "rmse", "rank_score", "date_updated"]].set_index("model"))