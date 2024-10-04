# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion - Delta Tables Pipeline

# COMMAND ----------

import dlt
from pyspark.sql.functions import *

# TODO: Enable custom library install for dlt pipeline 
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()

# # Define source paths for CSVs
# source_folder = os.environ.get("SOURCE_FOLDER")
source_folder = '/FileStore/sources/nycity-taxi'

# Helper function to clean column names (remove invalid characters)
def clean_column_names(df):
    for column in df.columns:
        cleaned_column = column.strip().replace(' ', '_')
        df = df.withColumnRenamed(column, cleaned_column)
    return df

@dlt.table(
    name="taxi_trip_fare",
    comment="Delta table for trip fare data"
)
def trip_fare():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{source_folder}/trip_fare_4.csv")

    return clean_column_names(df)

@dlt.table(
    name="taxi_trip_data",
    comment="Delta table for trip data"
)
def trip_data():
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{source_folder}/trip_data_4.csv")
    return clean_column_names(df) 

# TODO: Need to handle incremental loading
# dbutils.fs.rm(f"{source_folder}/trip_fare_4.csv", True)
# dbutils.fs.rm(f"{source_folder}/trip_data_4.csv", True)

# print("Files deleted successfully.")
