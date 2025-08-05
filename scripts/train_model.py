import pandas as pd
from sqlalchemy import create_engine
from prophet import Prophet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from datetime import timedelta, datetime
import warnings
import logging
from sklearn.exceptions import ConvergenceWarning
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.info("Starting the model training process...")
# Connect to DB
engine = create_engine("postgresql://dbtuser:dbtpass@postgres:5432/dbtdb")

# Load data from dbt model
df = pd.read_sql("SELECT * FROM forecast_features", engine)
df = df.rename(columns={"DATETIME": "ds", "TOTAL_WATT_HOURS": "y"})
df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

# Split the data
train = df[:int(len(df)*0.8)]
test = df[int(len(df)*0.8):]

X_train = train[['hour_of_day', 'day_of_week', 'is_weekend']]
y_train = train[['y']]
X_test = test[['hour_of_day', 'day_of_week', 'is_weekend']]
y_test = test[['y']]

# Define models
model_dt = DecisionTreeRegressor(random_state=101)
model_rf = RandomForestRegressor(random_state=101)
model_gb = GradientBoostingRegressor(random_state=101)

mlp = MLPRegressor(max_iter=200, random_state=101)
params = {
    'hidden_layer_sizes': [(10,30,10), (X_train.shape[1],)],
    'activation': ["identity", "logistic", "tanh", "relu"],
    'solver': ["lbfgs", "sgd", "adam"],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
model_nn = GridSearchCV(mlp, params, n_jobs=-1, cv=2)

# Fit all models
model_dt.fit(X_train, y_train)
model_rf.fit(X_train, y_train)
model_gb.fit(X_train, y_train)
model_nn.fit(X_train, y_train.values.ravel())

# Make predictions
preds = {
    'DecisionTree': model_dt.predict(X_test),
    'RandomForest': model_rf.predict(X_test),
    'GradientBoosting': model_gb.predict(X_test),
    'NeuralNetwork': model_nn.predict(X_test)
}

# Store metrics
metrics_data = []
predictions_data = []
now = datetime.utcnow()

for model_name, y_pred in preds.items():
    r2 = r2_score(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    
    metrics_data.append({
        "model": model_name,
        "r2_score": r2,
        "rmse": rmse,
        "rank_score": rmse / r2 if r2 != 0 else float("inf"),
        "date_updated": now
    })
    
    # Save predictions for monitoring
    temp = test.copy()
    temp['model'] = model_name
    temp['y_pred'] = y_pred
    predictions_data.append(temp[['ds', 'y', 'y_pred', 'model']])

# Concatenate all predictions
predictions_df = pd.concat(predictions_data)
predictions_df.to_sql("test_predictions", engine, if_exists="replace", index=False)


metrics_df = pd.DataFrame(metrics_data)

# Choose best model
best_model_name = metrics_df.sort_values("rank_score").iloc[0]['model']
metrics_df["selected_for_forecast"] = metrics_df["model"].apply(lambda x: 1 if x == best_model_name else 0)

metrics_df.to_sql("model_metrics", engine, if_exists="replace", index=False)
logging.info("Model metrics saved to the database.")
# === Future Forecasting (7 days ahead) ===

# Get the best model instance
model_lookup = {
    'DecisionTree': model_dt,
    'RandomForest': model_rf,
    'GradientBoosting': model_gb,
    'NeuralNetwork': model_nn
}
best_model = model_lookup[best_model_name]

# Generate 7 days hourly future timestamps
future_dates = pd.date_range(start=df['ds'].max() + timedelta(hours=1), periods=7*24, freq='h')

# Prepare features
future_df = pd.DataFrame({'ds': future_dates})
future_df['hour_of_day'] = future_df['ds'].dt.hour
future_df['day_of_week'] = future_df['ds'].dt.dayofweek.apply(lambda x: (x + 1) % 7)  # Adjust to Sunday=0
future_df['is_weekend'] = future_df['day_of_week'].apply(lambda x: 1 if x in [0, 6] else 0)

# Predict
future_df['yhat'] = best_model.predict(future_df[['hour_of_day', 'day_of_week', 'is_weekend']])
future_df['yhat'] = future_df['yhat'].clip(lower=0)

# Store forecast
future_df[['ds', 'yhat']].to_sql("forecast", engine, if_exists="replace", index=False)
logging.info("Forecast data saved to the database.")

logging.info("Model training and forecasting completed successfully.")
logging.info("Best ML model: %s", best_model_name)
logging.info("Pipeline completed successfully")

