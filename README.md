# Natural Language Processing with Azure Machine Learning

This repository is based on the code for the presentation entitled [Beyond the Basics with Azure ML](https://www.catallaxyservices.com/presentations/beyond-the-basics-with-azureml/) and is modified to meet my requirements.

## Generating Data

This data comes from the [Amazon Fine Food Reviews dataset from kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).  It includes ~500.000 food reviews from Amazon.

Import this into Azure ML using the Dataset name `amazon_fine_food_reviews_05`.  Be sure to upload this as a `uri_folder` instead of an `MLtable` or `uri_file`!

## Running the Code

### Basic Notebook

Import the notebook in the `Notebook` folder into Azure Machine Learning.  You will need to create a compute instance to run this.

### ML Pipeline

In order to run the ML pipeline notebooks and jobs locally, you will need to have the following installed on your machine:

* Python (preferably the [Anaconda distribution](https://www.anaconda.com/download#downloads)), with `pip` installed:  `conda install -c anaconda pip`
* [The Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
* The Azure ML Azure CLI extension:  `az extension add -n ml`
* Pip packages:  `pip install azure-ai-ml`, `pip install azure-identity`
* [Visual Studio Code](https://code.visualstudio.com/download)
* The [Azure ML Visual Studio Code extension](https://code.visualstudio.com/docs/datascience/azure-machine-learning)

Before you run the code, make sure your console has you logged into Azure via CLI:

```cmd
az login
```

Then, create a folder called `.azureml` and a file named `config.json`.  The file should look like the following structure:

```json
{
    "subscription_id": "YOUR SUBSCRIPTION ID",
    "resource_group": "YOUR RESOURCE GROUP",
    "workspace_name": "YOUR WORKSPACE NAME"
}
```

Note that you must be logged into `az cli` with an account which has access to the subscription, resource group, and workspace.

From there, run the training code:

```python
python deploy-train.py
```

You can see the job in action by going to [Azure ML Studio](https://ml.azure.com) and viewing the "NLP_Sentiment_Analysis_Coded_Amazon_Fine_Food" experiment.  There will be a new "train_pipeline" job.

For scoring, run the following code:

```python
python deploy-score.py
```

This will create a batch endpoint and deployment, upload data to a Datastore in Azure ML, create a job to generate predictions, and downloads the resulting predictions to a local file called `predictions.csv`.

**IMPORTANT NOTE** -- You must *explicitly* grant rights to the account running `deploy-score.py` against the Azure ML workspace.  I granted Owner because I was running this personally, but it must be explicitly granted and not just have ownership as a side effect of subscription-level or resource group-level rights.

If you do not do this, you will likely get a strange `BY_POLICY` error message when running this script.