import argparse
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

mlflow.sklearn.autolog()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from sklearn.metrics import classification_report

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

MODEL_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "booster": "gbtree",
    "random_state": 11,
    "use_label_encoder": False
}

def main(args):
    print(os.listdir(args.train_data))

    train_file_list=[]
    for filename in os.listdir(args.train_data):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.train_data, filename), "r") as f:
            input_df=pd.read_csv((Path(args.train_data) / filename))
            train_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    train_df=pd.concat(train_file_list)

    print(os.listdir(args.test_data))

    test_file_list=[]
    for filename in os.listdir(args.test_data):
        print("Reading file: %s ..." % filename)
        with open(os.path.join(args.test_data, filename), "r") as f:
            input_df=pd.read_csv((Path(args.test_data) / filename))
            test_file_list.append(input_df)

    # Concatenate the list of Python DataFrames
    test_df=pd.concat(test_file_list)
    
    X_train, y_train = extract_target_feature(train_df)
    X_test, y_test = extract_target_feature(test_df)
    
    model, results = train_model(X_train, X_test, y_train, y_test)
    
    print(f'Saving model XBoost...')
    mlflow.sklearn.save_model(model, args.model_output)
    
    print('Saving evauation results...')
    with open(Path(args.test_report) / 'results.json', 'w') as fp:
        json.dump(results, fp)
    
def extract_target_feature(df):
    label_column="Sentiment"
        
    X=df.drop(label_column, axis=1)
    y=df[label_column]

    # return split data
    return X, y

def train_model(X_train, X_test, y_train, y_test):
    print(f'Starting training of model XBoost...')
    print("Model parameters: ", MODEL_PARAMS)
    # train model
    model = XGBClassifier(**MODEL_PARAMS)

    model=model.fit(X_train, y_train)
    
    print(f'Training of model XBoost finished, starting evaluation...')
    y_preds=model.predict(X_test)

    results = classification_report(y_test, y_preds, output_dict=True)

    # Log metrics in mlflow
    mlflow.log_param("model", "SVC")
    for label, metrics in results.items():
        if label == "accuracy":
            mlflow.log_metric(label, metrics)
        else:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)
    
    # Log model parameters in mlflow
    # Log parameters
    for param, value in MODEL_PARAMS.items():
        mlflow.log_param(param, value)

    # Confusion Matrix
    sns.set()
    print("Creating Confusion Matrix...")
    plt.figure(figsize=(5,3))
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')

    # Save Confusion Matrix as image
    confusion_matrix_image_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_image_path)
    plt.close()

    print("Done")
    
    print(results)

    # return model
    return model, results

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path of prepped train data")
    parser.add_argument("--test_data", type=str, help="Path of prepped test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_report", type=str, help="Path of test_report")

    args=parser.parse_args()
    return args

# run script
if __name__ == "__main__":
    mlflow.start_run()
    
    args=parse_args()
    main(args)
    
    mlflow.end_run()
    print('Done!')
