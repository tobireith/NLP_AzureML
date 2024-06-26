import argparse
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
from azure.ai.ml.parallel import parallel_run_function, RunFunction

from azure.ai.ml.sweep import (
    Choice,
    Uniform
)

# Constants and global variables
WORKSPACE_NAME = "treithmaier-amls-01"
COMPUTE_NAME = "treithmaier-vm-gpu01"
PARENT_DIR = "./components/"
COMPONENT_NAMES_AND_PATHS = {
    "feature_engineering": "feature_engineering/feature_engineering.yml",
    "split_data": "split_data/split_data.yml",
    "train_model_roberta_twitter": "train_model_roberta_twitter/train_model_roberta_twitter.yml",
    "register_model": "register_model/register_model.yml"
}

# Initialize ML client
# NOTE:  for local runs, I'm using the Azure CLI credential
# For production runs as part of an MLOps configuration using
# Azure DevOps or GitHub Actions, I recommend using the DefaultAzureCredential
#ml_client=MLClient.from_config(DefaultAzureCredential())
ml_client = MLClient.from_config(AzureCliCredential())
ws = ml_client.workspaces.get(WORKSPACE_NAME)

def get_or_register_component(component_name, component_file_path):
    """
    Checks if a component exists in Azure ML and uploads it if not.
    Returns the component.
    """
    try:
        # Attempt to get the component. If it exists, no upload is needed.
        component = ml_client.components.get(name=component_name)
        print(f"Component {component_name} (version {component.version}) found in the registry.")
    except Exception as e:
        print(f"Component {component_name} not found. Uploading from {component_file_path}.")
        # The component does not exist, so create and upload it from the corresponding .yml file.
        component = ml_client.components.create_or_update(load_component(component_file_path))
    return component

def get_components():
    """
    Get the components from Azure or the .yml files in the `parent_dir` directory.
    """
    components = {}
    # Iterate through your components and register or load them as needed
    # So that you can use `components` to access the registered components,
    # e.g., `components['feature_engineering']` instead of `load_component(...)`.
    for name, file_path in COMPONENT_NAMES_AND_PATHS.items():
        print(f"Loading Component {name}...")
        full_path = os.path.join(PARENT_DIR, file_path)
        # components[name] = get_or_register_component(name, full_path)
        components[name] = load_component(full_path)
    return components


# Build the Pipeline with all the components
@pipeline(
        name="train_pipeline_for_multiple_models_balanced", 
        description="Build a training pipeline to train multiple models"
)
def build_pipeline(raw_data):
    train_model_components = {}
    # Load the components
    components = get_components()

    # Step 1: Feature Engineering
    step_feature_engineering = components['feature_engineering'](input_data=raw_data)
    # Step 2: Split Data
    step_split_data = components['split_data'](
        input_data=step_feature_engineering.outputs.output_data
    )
    # Step 3: Train Transformer Model
    # If GPU-Compute Targets are needed, use train_model.compute = "gpu-cluster"
    step_train_model_roberta_twitter = components['train_model_roberta_twitter'](
        train_data=step_split_data.outputs.output_data_train,
        test_data=step_split_data.outputs.output_data_test
    )
    results = {}
    results['train_model_roberta_twitter'] = {
        "model": step_train_model_roberta_twitter.outputs.model_output,
        "results_df": step_train_model_roberta_twitter.outputs.test_results,
        "report": step_train_model_roberta_twitter.outputs.test_report
    }
    print(results)
    # Step 4: Register Model
    step_register_model = components['register_model'](
        model=step_train_model_roberta_twitter.outputs.model_output,
        model_name="train_model_roberta_twitter",
        test_report=step_train_model_roberta_twitter.outputs.test_report
    )


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
    pipeline_job.display_name="train_pipeline_roberta_twitter"
    return pipeline_job

def main():
    # Make sure the compute ressource exists already
    try:
        compute_instance=ml_client.compute.get(COMPUTE_NAME)
        print(
            f"You already have a compute instance named {COMPUTE_NAME}, we'll reuse it as is."
        )

    except Exception:
        print("Creating a new cpu compute target...")

        # Let's create the Azure Machine Learning compute object with the intended parameters
        # if you run into an out of quota error, change the size to a comparable VM that is available.\
        # Learn more on https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

        compute_instance=AmlCompute(
            name=COMPUTE_NAME,
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

    prepped_job=prepare_pipeline_job(COMPUTE_NAME)
    # Register the components to the workspace
    ml_client.jobs.create_or_update(prepped_job, experiment_name="NLP_Sentiment_Analysis_Coded_Amazon_Fine_Food")

    print("Now look in the Azure ML Jobs UI to see the status of the pipeline job.  This will be in the 'NLP_Sentiment_Analysis_Coded_Amazon_Fine_Food' experiment.")

if __name__ == "__main__":
    main()
