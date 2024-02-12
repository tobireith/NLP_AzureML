import sys
import os
import timeit
from datetime import datetime
import numpy as np
import pandas as pd
from random import randrange
import urllib
from urllib.parse import urlencode

import azure.ai.ml
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Workspace, AmlCompute, Component
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml.dsl import pipeline
from azure.ai.ml import load_component
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.sweep import (
    Choice,
    Uniform
)

# NOTE:  set your workspace name here!
workspace_name="treithmaier-amls-01"
# NOTE:  if you do not have a compute instance already, we will create one
# Alternatively, change the name to a CPU-based compute cluster
compute_name="treithmaier-vm01"

# NOTE:  for local runs, I'm using the Azure CLI credential
# For production runs as part of an MLOps configuration using
# Azure DevOps or GitHub Actions, I recommend using the DefaultAzureCredential
#ml_client=MLClient.from_config(DefaultAzureCredential())
ml_client=MLClient.from_config(AzureCliCredential())
ws=ml_client.workspaces.get(workspace_name)

# Make sure the compute ressource exists already
try:
    compute_instance=ml_client.compute.get(compute_name)
    print(
        f"You already have a compute instance named {compute_name}, we'll reuse it as is."
    )

except Exception:
    print("Creating a new cpu compute target...")

    # Let's create the Azure Machine Learning compute object with the intended parameters
    # if you run into an out of quota error, change the size to a comparable VM that is available.\
    # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

    compute_instance=AmlCompute(
        name=compute_name,
        # Azure Machine Learning Compute is the on-demand VM service
        type="amlcompute",
        # VM Family
        size="Standard_E2a_v4",
        # How many seconds will the node running after the job termination
        idle_time_before_scale_down=180,
        # Dedicated or LowPriority. The latter is cheaper but there is a chance of job termination
        tier="Dedicated",
    )
    print(
        f"AMLCompute with name {compute_instance.name} will be created, with compute size {compute_instance.size}"
    )
    # Now, we pass the object to MLClient's create_or_update method
    cpu_compute_instancecluster=ml_client.compute.begin_create_or_update(compute_instance)

parent_dir="./config"

def get_or_register_component(component_name, component_file_path):
    """
    Checks if a component exists in Azure ML and uploads it if not.
    Returns the component.
    """
    try:
        # Attempt to get the component. If it exists, no upload is needed.
        component = ml_client.components.get(name=component_name, version="1")
        print(f"Component {component_name} found in the registry.")
    except Exception as e:
        print(f"Component {component_name} not found. Uploading from {component_file_path}.")
        # The component does not exist, so create and upload it from the corresponding .yml file.
        component = ml_client.components.create_or_update(load_component(component_file_path))
    return component

# Define the names and paths of your components
component_names_and_paths = {
    "feature_engineering": "feature-engineering.yml",
    "feature_text_preprocessing": "feature-text-preprocessing.yml",
    "split_data": "split-data.yml",
    "feature_encoding": "feature-encoding.yml",
    "train_model": "train-model.yml",
    "register_model": "register-model.yml"
}

# Iterate through your components and register or load them as needed
components = {}
for name, file_path in component_names_and_paths.items():
    full_path = os.path.join(parent_dir, file_path)
    components[name] = get_or_register_component(name, full_path)

# Now you can use `components` to access the registered components,
# e.g., `components['feature_engineering']` instead of `load_component(...)`.

# Build the Pipeline with all the components
@pipeline(name="training_pipeline", description="Build a training pipeline")
def build_pipeline(raw_data):
    step_feature_engineering = components['feature_engineering'](input_data=raw_data)
    step_feature_text_preprocessing = components['feature_text_preprocessing'](input_data=step_feature_engineering.outputs.output_data)
    step_split_data = components['split_data'](input_data=step_feature_text_preprocessing.outputs.output_data)
    step_feature_encoding = components['feature_encoding'](input_data_train=step_split_data.outputs.output_data_train,
                                   input_data_test=step_split_data.outputs.output_data_test)

    train_model_data = components['train_model'](train_data=step_feature_encoding.outputs.output_data_train,
                                   test_data=step_feature_encoding.outputs.output_data_test,
                                   max_leaf_nodes=128,
                                   min_samples_leaf=32,
                                   max_depth=12,
                                   learning_rate=0.1,
                                   n_estimators=100)
    # If GPU-Compute Targets are needed, use train_model.compute = "gpu-cluster"

    components['register_model'](model=train_model_data.outputs.model_output, test_report=train_model_data.outputs.test_report)
    return { "model": train_model_data.outputs.model_output,
             "report": train_model_data.outputs.test_report }

def prepare_pipeline_job(compute_name):
    # must have a dataset already in place
    cpt_asset = ml_client.data.get("amazon_fine_food_reviews_05", version="1")
    raw_data=Input(type='uri_folder', path=cpt_asset.path)
    pipeline_job=build_pipeline(raw_data)
    # set pipeline level compute
    pipeline_job.settings.default_compute=compute_name
    # set pipeline level datastore
    pipeline_job.settings.default_datastore="workspaceblobstore"
    pipeline_job.settings.force_rerun=False
    pipeline_job.display_name="train_pipeline"
    return pipeline_job

prepped_job=prepare_pipeline_job(compute_name)
# Register the components to the workspace
ml_client.jobs.create_or_update(prepped_job, experiment_name="NLP_Sentiment_Analysis_Coded_Amazon_Fine_Food")

print("Now look in the Azure ML Jobs UI to see the status of the pipeline job.  This will be in the 'NLP_Sentiment_Analysis_Coded_Amazon_Fine_Food' experiment.")
