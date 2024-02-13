import argparse
from azure.ai.ml import MLClient, Input, Output
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential, AzureCliCredential
from azure.ai.ml import load_component

def update_component(component_name, component_version=None):
    # Load the Azure ML client
    ml_client=MLClient.from_config(AzureCliCredential())

    try:
        # Attempt to get the component.
        component = ml_client.components.get(name=component_name)
        print(f"Component {component_name} (version {component.version}) found in the registry.")
    except Exception as e:
        print(f"Component {component_name} not found.")
        
    # Define the component file path
    component_file_path = f'./config/{component_name}.yml'

    # Create or update the component
    component = ml_client.components.create_or_update(load_component(component_file_path), version=component_version)

    print(f"Component {component_name} (version {component.version}) has been uploaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create or Update a component to Azure ML.')
    parser.add_argument('component_name', type=str, help='The name of the component to upload.')
    parser.add_argument('component_version', type=str, help='The version of the component to upload.')

    args = parser.parse_args()

    update_component(args.component_name, args.component_version)