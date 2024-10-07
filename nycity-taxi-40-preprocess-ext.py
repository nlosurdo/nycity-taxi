# Databricks notebook source
# MAGIC %md
# MAGIC # PreProcess Weather Data and write them in Features in Store

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# COMMAND ----------

# Step 1: Read the input tables
fs_taxi_data = spark.read.table("autoguidovie.nycity_taxi.fs_taxi_data")
taxi_weather_data = spark.read.table("hive_metastore.nycity_taxi.taxi_ext_weather_data")
display(fs_taxi_data.count())

# COMMAND ----------

# Extract pickup_datetimes to join
fs_taxi_data = fs_taxi_data.withColumn(
    "pickup_datetime",
    F.to_timestamp(
        F.split(F.col("pr_key"), "_").getItem(2), "yyyy-MM-dd HH:mm:ss"
    ),
)

taxi_weather_data = taxi_weather_data.withColumn(
    "datetime_normalized",
    F.to_timestamp(
        F.col("datetime"), "yyyy-MM-dd'T'HH:mm:ss.SSSXXX"
    ),
)

# Left join with a condition to consider pickup_datetime within an hour of datetime
fs_taxi_enriched_data = fs_taxi_data.join(
    taxi_weather_data.select(
        "datetime_normalized",
        "humidity",
        "precip",
        "preciptype",
        "windspeed",
        "cloudcover",
        "visibility",
        "conditions",
        "temp"
    ),
    (fs_taxi_data.pickup_datetime >= taxi_weather_data.datetime_normalized) &
    (fs_taxi_data.pickup_datetime < F.expr("datetime_normalized + interval 1 hour")),
    how="left",
).drop(
    *["datetime_normalized", "pickup_datetime"]
)
display(fs_taxi_enriched_data.count())
# Write to Databricks Feature Store
try:
    fs.get_table("nycity_taxi.fs_taxi_enriched_data")
    fs.drop_table("nycity_taxi.fs_taxi_enriched_data")
except Exception as e:
    print("Table does not exist. Proceeding to create it.")

fs.create_table(
    name="nycity_taxi.fs_taxi_enriched_data",
    primary_keys=["pr_key"],
    df=fs_taxi_enriched_data,
    description="NYC Taxi data enriched with weather conditions",
)
