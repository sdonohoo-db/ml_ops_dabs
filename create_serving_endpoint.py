# Databricks notebook source
from pprint import pprint

from config import DeployModelConfig, ServingEndpointPermissions
from serving.create_endpoint import create_endpoint, ModelServingConfig

# COMMAND ----------

dbutils.widgets.text("config_path", "workflow_configs/model_deployment.yaml")
dbutils.widgets.text("perms_config_path", "workflow_configs/endpoint_perms.yaml")
dbutils.widgets.text("environment", "dev")
config_path = dbutils.widgets.get("config_path")
perms_config_path = dbutils.widgets.get("perms_config_path")
environment = dbutils.widgets.get("environment")

# COMMAND ----------

cfg = DeployModelConfig.from_yaml(config_path)
cfg

# COMMAND ----------

perms_cfg = ServingEndpointPermissions.from_yaml(perms_config_path)
perms_cfg

# COMMAND ----------

model = getattr(cfg, f"{environment}_model")
scale_to_zero_enabled = False if environment == "prod" else True
serving_cfg = ModelServingConfig(
    endpoint_name=cfg.endpoint_name,
    registered_model_name=model.path,
    model_version=model.model_version,
    catalog=model.catalog,
    schema=model.schema,
    scale_to_zero_enabled=scale_to_zero_enabled,
)
serving_cfg

# COMMAND ----------

endpoint_info = create_endpoint(serving_cfg, perms_cfg)
pprint(endpoint_info)

# COMMAND ----------


