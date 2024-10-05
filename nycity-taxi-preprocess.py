# Databricks notebook source
# MAGIC %md
# MAGIC # PreProcess and write Features in Store

# COMMAND ----------

from pyspark.sql import functions as F
from databricks.feature_store import FeatureStoreClient
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import rand

import geopandas as gpd
import folium

fs = FeatureStoreClient()

# COMMAND ----------

# Step 1: Read the input table
taxi_common_data = spark.read.table("hive_metastore.nycity_taxi.taxi_common_data")

# Step 2: Convert store_and_fwd_flag
taxi_common_data = taxi_common_data.withColumn(
    "store_and_fwd_flag",
    F.when(F.col("store_and_fwd_flag").isNull(), "Unknown")
    .when(F.col("store_and_fwd_flag") == True, "True")
    .otherwise("False"),
)

# Step 3: Create pickup_day_of_week and pickup_hour columns
taxi_common_data = taxi_common_data.withColumn(
    "pickup_day_of_week", F.dayofweek("pickup_datetime")
)
taxi_common_data = taxi_common_data.withColumn("pickup_hour", F.hour("pickup_datetime"))
taxi_common_data = taxi_common_data.withColumn(
    "pickup_minute", F.minute("pickup_datetime")
)

# COMMAND ----------

display(taxi_common_data.limit(10))

# COMMAND ----------

# Step 4: Clustering for pickup and dropoff coordinates with 100 and 200 clusters
def cluster_coordinates(df, lat_col, lon_col, k, new_col_name):
    assembler = VectorAssembler(inputCols=[lat_col, lon_col], outputCol="features")
    assembled_data = assembler.transform(df)
    kmeans = KMeans(k=k, seed=42)
    model = kmeans.fit(assembled_data)
    clusters = model.transform(assembled_data)
    return clusters.withColumnRenamed("prediction", new_col_name).drop("features")


# Apply clustering for pickup and dropoff with 100 clusters (small granularity)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data, "pickup_latitude", "pickup_longitude", 50, "pickup_clusters_s"
)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data_cl,
    "dropoff_latitude",
    "dropoff_longitude",
    50,
    "dropoff_clusters_s",
)

# Apply clustering for pickup and dropoff with 100 clusters (medium granularity)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data_cl, "pickup_latitude", "pickup_longitude", 100, "pickup_clusters_m"
)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data_cl,
    "dropoff_latitude",
    "dropoff_longitude",
    100,
    "dropoff_clusters_m",
)

# Apply clustering for pickup and dropoff with 200 clusters (large granularity)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data_cl, "pickup_latitude", "pickup_longitude", 200, "pickup_clusters_l"
)
taxi_common_data_cl = cluster_coordinates(
    taxi_common_data_cl,
    "dropoff_latitude",
    "dropoff_longitude",
    200,
    "dropoff_clusters_l",
)

# COMMAND ----------

display(taxi_common_data_cl.limit(10))
display(taxi_common_data_cl.groupBy("pickup_clusters_m").count().orderBy("count"))

# COMMAND ----------

# Step 5: Write intermediate table to Hive for verification
taxi_common_data_cl.write.mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable("hive_metastore.nycity_taxi.taxi_processed_data")

# COMMAND ----------

# Step 6: Remove unnecessary columns
columns_to_remove = [
    "trip_time_in_secs",
    "fare_amount",
    "tip_amount",
    "tolls_amount",
    "pickup_datetime",
    "dropoff_datetime",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
taxi_common_data_fs = taxi_common_data_cl.drop(*columns_to_remove)

# Step 7: Define target variable and primary key
taxi_common_data_fs = taxi_common_data_fs.withColumnRenamed("total_amount", "Y")

# Step 8: Write to Databricks Feature Store
try:
    fs.get_table("nycity_taxi.fs_taxi_data")
    fs.drop_table("nycity_taxi.fs_taxi_data")
except Exception as e:
    print("Table does not exist. Proceeding to create it.")

fs.create_table(
    name="nycity_taxi.fs_taxi_data",
    primary_keys=["pr_key"],
    df=taxi_common_data_fs,
    description="NYC Taxi data with clustered pickup and dropoff locations",
)

# COMMAND ----------

# Step 9: Write to Databricks GeoMap
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

# Step 10: Pick up top 20 medium clusters and write DeltaTable
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
