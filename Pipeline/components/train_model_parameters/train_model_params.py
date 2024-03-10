import argparse
from pathlib import Path
from uuid import uuid4
from datetime import datetime
import os
import time
import json
import pickle
import sys
import numpy as np
import pandas as pd
import urllib
from math import sqrt

import sklearn.ensemble
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import clone
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import mlflow
import mlflow.sklearn

MODEL_PARAMS = {
    "naive_bayes": {
        "alpha": 1.0,
        "fit_prior": True,
    },
    "svc": {
        "C": 1.0,
        "kernel": "rbf",
        "random_state": 11
    },
    "logistic_regression": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "max_iter": 100,
        "random_state": 11
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 12,
        "random_state": 11
    },
    "gradient_boosting": {
        "max_leaf_nodes": 128,
        "min_samples_leaf": 32,
        "max_depth": 12,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "validation_fraction": 0.1,
        "random_state": 11
    }
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
    
    X_train, y_train=process_data(train_df)
    X_test, y_test=process_data(test_df)

    # train model: Gradient Boosted Decision Trees
    
    model, results = train_model(args.model_name, X_train, X_test, y_train, y_test)
    
    print(f'Saving model {args.model_name}...')
    mlflow.sklearn.save_model(model, args.model_output)
    
    print('Saving evauation results...')
    with open(Path(args.test_report) / 'results.json', 'w') as fp:
        json.dump(results, fp)
    
def process_data(df):
    label_column="Score"
        
    X=df.drop(label_column, axis=1)
    y=df[label_column]

    # return split data
    return X, y

def load_model_params(model_name):
    if model_name not in MODEL_PARAMS:
        raise ValueError(f"Unknown model name: {model_name}. Valid model names are: {', '.join(MODEL_PARAMS.keys())}")

    return MODEL_PARAMS[model_name]

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def create_model(model_name):
    params = load_model_params(model_name)
    if model_name == 'naive_bayes':
        model = MultinomialNB(**params)
    elif model_name == 'logistic_regression':
        model = LogisticRegression(**params)
    elif model_name == 'gradient_boosting':
        model = GradientBoostingClassifier(**params)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(**params)
    elif model_name == 'svc':
        model = SVC(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

def train_model(model_name, X_train, X_test, y_train, y_test):
    # train model
    model = create_model(model_name)
    model=model.fit(X_train, y_train)
    
    y_preds=model.predict(X_test)

    results = classification_report(y_test, y_preds, output_dict=True)
    
    print(results)

    # return model
    return model, results

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path of prepped train data")
    parser.add_argument("--test_data", type=str, help="Path of prepped test data")
    parser.add_argument('--model_name', type=str, help="Name of model to train")
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
