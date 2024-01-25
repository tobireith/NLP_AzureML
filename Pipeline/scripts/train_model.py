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
    params={
        'max_leaf_nodes': args.max_leaf_nodes,
        'min_samples_leaf': args.min_samples_leaf,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'validation_fraction': 0.1,
        'random_state': 11
        }
    
    model, results=train_model(params, X_train, X_test, y_train, y_test)
    
    print('Saving model...')
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


def train_model(params, X_train, X_test, y_train, y_test):
    # train model
    model=GradientBoostingClassifier(**params)
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
    parser.add_argument('--max_leaf_nodes', type=int)
    parser.add_argument('--min_samples_leaf', type=int)
    parser.add_argument('--max_depth', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--n_estimators', type=int)
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
