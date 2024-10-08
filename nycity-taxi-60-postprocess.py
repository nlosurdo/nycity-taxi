# Databricks notebook source
# MAGIC %md
# MAGIC # PreProcess and write Features in Store

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import rand

import geopandas as gpd
import folium


# COMMAND ----------

# GeoMap
taxi_processed_data = spark.read.table("hive_metastore.nycity_taxi.taxi_processed_data")

# Convert Spark DataFrame to Pandas DataFrame
taxi_processed_data_pandas = (
    taxi_processed_data.select(
        "pickup_latitude", "pickup_longitude", "pickup_clusters_m"
    )
    .orderBy(rand())
    .limit(20000)
    .toPandas()
)

# Create a Folium map centered on the average pickup latitude and longitude
mean_latitude = taxi_processed_data_pandas["pickup_latitude"].mean()
mean_longitude = taxi_processed_data_pandas["pickup_longitude"].mean()

map_clusters = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=12)

# Define a color palette for the clusters
colors = [
    "#FF6666",  # bright red
    "#FF9966",  # bright orange
    "#FFFF66",  # bright yellow
    "#66FF66",  # bright green
    "#66FFFF",  # bright cyan
    "#6699FF",  # bright blue
    "#9966FF",  # bright purple
    "#FF66FF",  # bright magenta
    "#FF33CC",  # vibrant pink
    "#FFCC33",  # vibrant yellow-orange
    "#FF9933",  # vibrant orange
    "#66FFCC",  # bright aqua
    "#33CC33",  # bright lime
    "#66CCFF",  # bright sky blue
    "#CC99FF",  # bright lavender
    "#FF6699",  # bright coral pink
    "#33CCCC",  # vibrant teal
    "#99FF66",  # bright light green
    "#FFCCCC",  # light salmon pink
    "#CC66FF",  # vibrant purple
]


# Add points to the map for each cluster
for idx, row in taxi_processed_data_pandas.iterrows():
    cluster_index = int(row["pickup_clusters_m"]) % len(colors)
    folium.CircleMarker(
        location=(row["pickup_latitude"], row["pickup_longitude"]),
        radius=1,
        color=colors[cluster_index],
        fill=True,
        fill_color=colors[cluster_index],
        fill_opacity=0.1,
        tooltip=f"Cluster: {row['pickup_clusters_m']}",
    ).add_to(map_clusters)

# Show the map
map_clusters

# COMMAND ----------

# Pick up top 20 medium clusters and write DeltaTable
top_clusters = taxi_processed_data.groupBy('pickup_clusters_m') \
    .agg(F.count('*').alias('count')) \
    .orderBy(F.desc('count')) \
    .limit(20)

taxi_top20_clusters = taxi_processed_data.join(
    top_clusters, 
    on='pickup_clusters_m', 
    how='inner'
)

taxi_top20_clusters.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("hive_metastore.nycity_taxi.taxi_top20_clusters")

# COMMAND ----------

# Best PickUp, Drop Off Clusters for drivers to pick

# Find the highest average total_amount
taxi_avg_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m'
).agg(
    F.avg('total_amount').alias('avg_total_amount')
)

# Calculate the sum total_amount for each group and normalize it
total_sum = taxi_processed_data.agg(F.sum('total_amount').alias('total_sum_total_amount')).collect()[0]['total_sum_total_amount']

taxi_sum_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m'
).agg(
    F.sum('total_amount').alias('sum_total_amount')
)

taxi_sum_normalized = taxi_sum_amount.withColumn(
    'normalized_sum_total_amount', 
    F.col('sum_total_amount') / total_sum
)

# Step 3: Combine results from Step 1 and Step 2
taxi_combined = taxi_avg_amount.join(
    taxi_sum_normalized, 
    on=['pickup_clusters_m', 'dropoff_clusters_m']
)

# Calculate the weighted value using normalized sum_total_amount as the weight
taxi_combined = taxi_combined.withColumn(
    'weighted_value', 
    F.col('avg_total_amount') * F.col('normalized_sum_total_amount')
)

# Get the top 20 combinations based on the weighted value
taxi_driver_top20_zones = taxi_combined.orderBy(F.desc('weighted_value')).limit(20)

# Show the result
display(taxi_driver_top20_zones)

# COMMAND ----------

# Best PickUp, Drop Off Clusters, Week/Hour for taxi drivers to pick

# Find the highest average total_amount
taxi_avg_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour'
).agg(
    F.avg('total_amount').alias('avg_total_amount')
)

# Calculate the sum total_amount for each group and normalize it
total_sum = taxi_processed_data.agg(F.sum('total_amount').alias('total_sum_total_amount')).collect()[0]['total_sum_total_amount']

taxi_sum_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour'
).agg(
    F.sum('total_amount').alias('sum_total_amount')
)

taxi_sum_normalized = taxi_sum_amount.withColumn(
    'normalized_sum_total_amount', 
    F.col('sum_total_amount') / total_sum
)

# Step 3: Combine results from Step 1 and Step 2
taxi_combined = taxi_avg_amount.join(
    taxi_sum_normalized, 
    on=['pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour']
)

# Calculate the weighted value using normalized sum_total_amount as the weight
taxi_combined = taxi_combined.withColumn(
    'weighted_value', 
    F.col('avg_total_amount') * F.col('normalized_sum_total_amount')
)

# Get the top 20 combinations based on the weighted value
taxi_driver_top100_zonetime = taxi_combined.orderBy(F.desc('weighted_value')).limit(100)

# Show the result
display(taxi_driver_top100_zonetime)

# COMMAND ----------

# Best PickUp, Drop Off Clusters for drivers to maximize profit

# Find the highest average total_amount
taxi_avg_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m'
).agg(
    F.sum('total_amount').alias('total_amount'),
    F.sum('trip_time_in_secs').alias('total_trip_time_in_secs')
).withColumn(
    'avg_amount_per_sec', F.col('total_amount') / F.col('total_trip_time_in_secs')
)

# Calculate the sum total_amount for each group and normalize it
total_sum = taxi_processed_data.agg(F.sum('total_amount').alias('total_sum_total_amount')).collect()[0]['total_sum_total_amount']

taxi_sum_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m'
).agg(
    F.sum('total_amount').alias('sum_total_amount')
)

taxi_sum_normalized = taxi_sum_amount.withColumn(
    'normalized_sum_total_amount', 
    F.col('sum_total_amount') / total_sum
)

# Step 3: Combine results from Step 1 and Step 2
taxi_combined = taxi_avg_amount.join(
    taxi_sum_normalized, 
    on=['pickup_clusters_m', 'dropoff_clusters_m']
)

# Calculate the weighted value using normalized sum_total_amount as the weight
taxi_combined = taxi_combined.withColumn(
    'weighted_value', 
    F.col('avg_amount_per_sec') * F.col('normalized_sum_total_amount')
)

# Get the top 20 combinations based on the weighted value
taxi_driver_top20_profit_zones = taxi_combined.orderBy(F.desc('weighted_value')).limit(20)

# Show the result
display(taxi_driver_top20_profit_zones)

# COMMAND ----------

# Best PickUp, Drop Off Clusters, Week/Hour for taxi drivers to maximize profit

# Find the highest average total_amount
taxi_avg_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour'
).agg(
    F.sum('total_amount').alias('total_amount'),
    F.sum('trip_time_in_secs').alias('total_trip_time_in_secs')
).withColumn(
    'avg_amount_per_sec', F.col('total_amount') / F.col('total_trip_time_in_secs')
)

# Calculate the sum total_amount for each group and normalize it
total_sum = taxi_processed_data.agg(F.sum('total_amount').alias('total_sum_total_amount')).collect()[0]['total_sum_total_amount']

taxi_sum_amount = taxi_processed_data.groupBy(
    'pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour'
).agg(
    F.sum('total_amount').alias('sum_total_amount')
)

taxi_sum_normalized = taxi_sum_amount.withColumn(
    'normalized_sum_total_amount', 
    F.col('sum_total_amount') / total_sum
)

# Step 3: Combine results from Step 1 and Step 2
taxi_combined = taxi_avg_amount.join(
    taxi_sum_normalized, 
    on=['pickup_clusters_m', 'dropoff_clusters_m', 'pickup_day_of_week', 'pickup_hour']
)

# Calculate the weighted value using normalized sum_total_amount as the weight
taxi_combined = taxi_combined.withColumn(
    'weighted_value', 
    F.col('avg_amount_per_sec') * F.col('normalized_sum_total_amount')
)

# Get the top 100 combinations based on the weighted value
taxi_driver_top100_profit_zonetime = taxi_combined.orderBy(F.desc('weighted_value')).limit(100)

# Show the result
display(taxi_driver_top100_profit_zonetime)
