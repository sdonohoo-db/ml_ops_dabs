# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Model Transition and Alias Assignment
# MAGIC
# MAGIC Promote a model to the current environment and assign an alias to the destination model.

# COMMAND ----------

# MAGIC %pip install mlflow==2.12.2
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from config import DeployModelConfig

# COMMAND ----------

dbutils.widgets.text("config_path", "./workflow_configs/model_deployment.yaml")
dbutils.widgets.text("environment", "dev")
config_path = dbutils.widgets.get("config_path")
environment = dbutils.widgets.get("environment")

cfg = DeployModelConfig.from_yaml(config_path)
cfg

# COMMAND ----------

def copy_and_alias_model(src_uri: str, dest_path: str, dest_alias: str):
    """Copy source model to the destination and give it an alias"""
    print(f"Starting copy:\n\tSource: {src_uri}\n\tDestination: {dest_path} with alias {dest_alias}")
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.tracking.MlflowClient()
    # Copy model version from source to destination
    copied_model_version = client.copy_model_version(src_uri, dest_path)
    print(f"Model version copied to destination successfully. New model version: {copied_model_version.version}")
    # Assign alias to the destination model
    client.set_registered_model_alias(name=dest.path, alias=dest_alias, version=copied_model_version.version)
    print(f"Alias '{dest_alias}' assigned to model version {copied_model_version.version}")

# COMMAND ----------

predecessor_envs = {"prod": "dev"}
if environment != "dev":
    src = getattr(cfg, f"{predecessor_envs[environment]}_model")
    dest = getattr(cfg, f"{environment}_model")
    copy_and_alias_model(src.uri, dest.path, dest.model_alias)
else:
    # no-op in dev
    print("Running in dev, so nothing to do")

# COMMAND ----------


