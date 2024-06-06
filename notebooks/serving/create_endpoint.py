"""Functionality for creating and updating Databricks Model Serving endpoints."""

import requests
from dataclasses import dataclass, field

from mlflow.deployments import get_deploy_client
from mlflow.exceptions import MlflowException

from serving.utils import get_api_credentials
from config import ServingEndpointPermissions

@dataclass
class ModelServingConfig:
    endpoint_name: str
    registered_model_name: str
    model_version: str
    catalog: str
    schema: str
    # fields below will be filled in automatically upon init
    scale_to_zero_enabled: bool = True
    inference_table_prefix: str = field(init=False)
    api_root: str = field(init=False)
    api_token: str = field(init=False)
    workload_size: str = 'Small'
    workload_type: str = 'CPU'

    def __post_init__(self):
        self.api_root, self.api_token = get_api_credentials()
        self.inference_table_prefix = (
            self.endpoint_name.replace("-", "_") + "_request_response"
        )

    def get_entity_config(self):
        """Return the json config for this entity to be used with the Databricks
        REST API for creating/updating endpoints."""
        return {
            "entity_name": self.registered_model_name,
            "entity_version": self.model_version,
            "workload_size": self.workload_size,
            "worksload_type": self.workload_type,
            "scale_to_zero_enabled": self.scale_to_zero_enabled,
        }


def get_endpoint_info(endpoint_name: str):
    """Get info from existing endpoint

    Returns None if the specified endpoint does not exist.
    """
    api_root, api_token = get_api_credentials()
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {api_token}"}
    resp = requests.get(
        f"{api_root}/api/2.0/serving-endpoints/{endpoint_name}", headers=headers
    )
    if resp.status_code == 404:
        # endpoint doesn't exist
        return None
    resp.raise_for_status()
    return resp.json()


def create_endpoint(
    cfg: ModelServingConfig, perms_cfg: ServingEndpointPermissions = None
):
    """Create a new endpoint with the specified configuration."""
    data = {
        "name": cfg.endpoint_name,
        "config": {
            "served_entities": [cfg.get_entity_config()]
            # ,"auto_capture_config": {
            #     "catalog_name": cfg.catalog,
            #     "schema_name": cfg.schema,
            #     "table_name_prefix": cfg.inference_table_prefix,
            # },
        },
    }
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {cfg.api_token}"}
    url = f"{cfg.api_root}/api/2.0/serving-endpoints"
    response = requests.post(url=url, json=data, headers=headers)
    response.raise_for_status()
    response_json = response.json()
    serving_endpoint_id = response_json["id"]
    if perms_cfg:
        perms_url = f"{cfg.api_root}/api/2.0/permissions/serving-endpoints/{serving_endpoint_id}"
        perms_response = requests.patch(
            url=perms_url, json=perms_cfg.acl, headers=headers
        )
        if not perms_response.ok:
            print(
                f"Updating endpoint perms returned with status {perms_response.status_code}"
            )
    return response_json


def make_update_request(endpoint_name: str, data: dict):
    """Make serving endpoint update request to Databricks REST API"""
    api_root, api_token = get_api_credentials()
    headers = {"Context-Type": "text/json", "Authorization": f"Bearer {api_token}"}
    url = f"{api_root}/api/2.0/serving-endpoints/{endpoint_name}/config"
    response = requests.put(url=url, json=data, headers=headers)
    response.raise_for_status()
    return response


def update_endpoint(
    cfg: ModelServingConfig, endpoint_info: str
):
    """Update existing endpoint from the config and existing endpoint info

    NOTE: This currently just overwrites the existing entities in the endpoint with the new
    entity. You can instead specify multiple entities (e.g. the new one and the existing one),
    as well as traffic percentages between the two. This enables different rollout strategies such
    as A/B testing or gradual rollout.

    TODO: We may want to inherit the value of `scale_to_zero_enabled` from the current entity
    when replacing it completely.
    """
    data = {
        "served_entities": [cfg.get_entity_config()],
    }
    response = make_update_request(cfg.endpoint_name, data)
    return response.json()


def create_or_update_endpoint(
    cfg: ModelServingConfig, perms_cfg: ServingEndpointPermissions = None
):
    """Create or update a Provisioned Throughput Databricks Foundation Model endpoint with the
    specified configuration."""
    endpoint_info = get_endpoint_info(cfg.endpoint_name)
    if endpoint_info is None:
        # does not exist yet
        print(f"Creating {cfg.endpoint_name}")
        return create_endpoint(cfg, perms_cfg)
    else:
        # update existing
        print(f"Updating {cfg.endpoint_name}")
        return update_endpoint(cfg, endpoint_info)


