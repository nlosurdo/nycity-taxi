# Databricks notebook source
# MAGIC %md
# MAGIC # Pre Processing - Preparing Base DataFrame

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

# Table 1 - taxi_trip_data
taxi_trip_data = spark.read.table('hive_metastore.nycity_taxi.taxi_trip_data')
display(taxi_trip_data.count())

# COMMAND ----------

# Rimozione record con valori 0 sui campi di interesse
taxi_trip_data = taxi_trip_data.filter(
    (taxi_trip_data['trip_time_in_secs'] != 0) &
    (taxi_trip_data['trip_distance'] != 0) &
    (taxi_trip_data['pickup_latitude'] != 0) &
    (taxi_trip_data['pickup_longitude'] != 0)
)
display(taxi_trip_data.count())

# COMMAND ----------

# Rimozione  dei duplicati per chiave
key_columns = ['medallion', 'hack_license', 'pickup_datetime']

duplicati = taxi_trip_data.groupBy(key_columns).count().filter("count > 1").select(key_columns)

taxi_trip_data = taxi_trip_data.join(duplicati, on=key_columns, how='left_anti')
display(taxi_trip_data.count())

# COMMAND ----------

# Table 2 - taxi_trip_fare
taxi_trip_fare = spark.read.table('hive_metastore.nycity_taxi.taxi_trip_fare')
display(taxi_trip_fare.count())

# COMMAND ----------

# Rimozione  dei duplicati per chiave
duplicati = taxi_trip_fare.groupBy(key_columns).count().filter("count > 1").select(key_columns)

taxi_trip_fare = taxi_trip_fare.join(duplicati, on=key_columns, how='left_anti')
display(taxi_trip_fare.count())

# COMMAND ----------

# Join for final taxi_common_data table, deduplicate colnames, create pr_key
fare_columns = [col for col in taxi_trip_fare.columns if col not in taxi_trip_data.columns]

taxi_common_data = taxi_trip_data.join(
    taxi_trip_fare.select(key_columns+fare_columns),
    on=key_columns,
    how='inner')

taxi_common_data = taxi_common_data.withColumn(
    "pr_key",
    F.concat(
        F.col("medallion"),
        F.lit("_"),
        F.col("hack_license"),
        F.lit("_"),
        F.col("pickup_datetime")
    )
)

taxi_common_data = taxi_common_data.drop("medallion", "hack_license")

final_columns = ["pr_key"] + [col for col in taxi_common_data.columns if col != 'pr_key']
taxi_common_data = taxi_common_data.select(final_columns)

display(taxi_common_data.count())

# COMMAND ----------

# Scrivere la tabella unita in formato Delta
output_deltatable = "hive_metastore.nycity_taxi.taxi_common_data"

taxi_common_data.write.format("delta").mode("overwrite").saveAsTable(output_deltatable)
