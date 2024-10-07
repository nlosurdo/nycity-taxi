# Databricks notebook source
# MAGIC %md
# MAGIC # Register and Deploy Models

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# COMMAND ----------

# Get The Model Artifact from Experiment Run
run_id = '57316c9e729442b7a4a80bad4660380c'
model_uri = f"runs:/{run_id}/model"

# COMMAND ----------

# Register the model in the registry
model_name = "NyTaxi-XGBoostReduced"

mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# Deploy the model to Databricks Serve
client = MlflowClient()

latest_version_info = client.get_latest_versions(name=model_name, stages=["None"])
model_version = latest_version_info[0].version

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage="Production",
    archive_existing_versions=True
)
