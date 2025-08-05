import pandas as pd
import boto3
from io import BytesIO
from io import StringIO
from sqlalchemy import create_engine
import requests

# Read example data directly into pandas (replace with your actual data source)
df = pd.read_csv("ec_example_data.csv", sep = ";")
df['CHARGE_START_TIME_AT'] = pd.to_datetime(df['CHARGE_START_TIME_AT'], errors='coerce')
df['CHARGE_STOP_TIME_AT'] = pd.to_datetime(df['CHARGE_STOP_TIME_AT'], errors='coerce')

engine = create_engine("postgresql://dbtuser:dbtpass@postgres:5432/dbtdb")
df.to_sql("raw_data", engine, if_exists="replace", index=False)
