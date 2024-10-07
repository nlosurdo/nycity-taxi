# Databricks notebook source
# MAGIC %md
# MAGIC # Data Exploration

# COMMAND ----------

import pandas as pd
from pandas_profiling import ProfileReport

# COMMAND ----------

# taxi_trip_data spark dataframe
spark_df = spark.read.table('hive_metastore.nycity_taxi.taxi_trip_data')

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame

pandas_df = spark_df.toPandas()
print(pandas_df.head())

# COMMAND ----------

# Create a pandas profiling report
profile = ProfileReport(pandas_df, title="Pandas Profiling Report", explorative=True)
profile.to_file("./results/taxi_trip_data_profile.html")


# COMMAND ----------

# taxi_trip_fare spark dataframe
spark_df = spark.read.table('hive_metastore.nycity_taxi.taxi_trip_fare')

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame

pandas_df = spark_df.toPandas()
print(pandas_df.head())

# COMMAND ----------

# Create a pandas profiling report
profile = ProfileReport(pandas_df, title="Pandas Profiling Report", explorative=True)
profile.to_file("./results/taxi_trip_fare_profile.html")

# COMMAND ----------

# Convert the Spark DataFrame to a Pandas DataFrame
taxi_common_data = spark.read.table('hive_metastore.nycity_taxi.taxi_common_data')

# COMMAND ----------

# Create a pandas profiling report
pandas_df = taxi_common_data.toPandas()

profile = ProfileReport(pandas_df, title="Pandas Profiling Report", explorative=True)
profile.to_file("./results/taxi_common_data_profile.html")
