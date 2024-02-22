from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential
from azure.ai.ml.entities import ComputeInstance, AmlCompute

WORKSPACE_NAME = "treithmaier-amls-01"

# Load the workspace from the saved config file
ml_client = MLClient.from_config(AzureCliCredential())
ws = ml_client.workspaces.get(WORKSPACE_NAME)

# Define the name and configuration of the GPU compute instance
COMPUTE_NAME = "treithmaier-vm-gpu01"
COMPUTE_VM_SIZE = "Standard_NC4as_T4_v3"

# Check if the compute ressource exists already
try:
    compute_instance=ml_client.compute.get(COMPUTE_NAME)
    print(f"You already have a compute instance named {COMPUTE_NAME}, not creating a new one.")

except Exception:
    print("Creating a new GPU compute target...")

    compute_instance = ComputeInstance(
        # Name of the compute instance
        name = COMPUTE_NAME,
        # VM Family
        size = COMPUTE_VM_SIZE,
        # How many minutes will the instance be running after the job termination
        idle_time_before_shutdown_minutes = 45,
    )
    print(f"ComputeInstance with name {compute_instance.name} will be created, with compute size {compute_instance.size}")
    # Now, we pass the object to MLClient's create_or_update method
    gpu_compute_instance = ml_client.compute.begin_create_or_update(compute_instance)
    print(f"ComputeInstance with name {compute_instance.name} has been created.")
