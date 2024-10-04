# Databricks notebook source
# MAGIC %md
# MAGIC # Data Ingestion

# COMMAND ----------

# Read the Delta table into a Spark DataFrame
spark_df = spark.read.table('hive_metastore.nycity_taxi.taxi_trip_data')

# COMMAND ----------

import pandas as pd

# Convert the Spark DataFrame to a Pandas DataFrame
pandas_df = spark_df.toPandas()

# Show or manipulate the Pandas DataFrame
print(pandas_df.head())

# COMMAND ----------

from pandas_profiling import ProfileReport

# Create a pandas profiling report
profile = ProfileReport(pandas_df, title="Pandas Profiling Report", explorative=True)

# Save the report to an HTML file
profile.to_file("pandas_profiling_report.html")

# Alternatively, you can display it directly in a Jupyter notebook
# profile.to_notebook_iframe()

