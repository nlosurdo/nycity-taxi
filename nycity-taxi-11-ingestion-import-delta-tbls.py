# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion - Delta Tables Pipeline

# COMMAND ----------

import dlt
from pyspark.sql.functions import *

source_folder = '/FileStore/sources/nycity-taxi'

# Helper function to clean column names
def clean_column_names(df):
    for column in df.columns:
        cleaned_column = column.strip().replace(' ', '_')
        df = df.withColumnRenamed(column, cleaned_column)
    return df

@dlt.table(
    name="taxi_trip_fare",
    comment="Delta table for trip fare data"
)
@dlt.incremental()
def trip_fare():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{source_folder}/trip_fare_4.csv")

    return clean_column_names(df)

@dlt.table(
    name="taxi_trip_data",
    comment="Delta table for trip data"
)
@dlt.incremental()
def trip_data():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{source_folder}/trip_data_4.csv")
    return clean_column_names(df) 
