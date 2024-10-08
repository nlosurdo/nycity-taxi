# Databricks notebook source
# MAGIC %md
# MAGIC # External Data Ingestion - Delta Tables Pipeline

# COMMAND ----------

import dlt
from pyspark.sql.functions import *

source_folder = '/FileStore/sources/nycity-taxi/external_data'

# Helper function to clean column names (remove invalid characters)
def clean_column_names(df):
    for column in df.columns:
        cleaned_column = column.strip().replace(' ', '_')
        df = df.withColumnRenamed(column, cleaned_column)
    return df

@dlt.table(
    name="taxi_ext_weather_data",
    comment="Delta table for external weather data"
)
@dlt.incremental()
def trip_fare():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{source_folder}/weather_nyc_201304.csv")

    return clean_column_names(df)

