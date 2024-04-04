import argparse
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report

from transformers import AutoTokenizer, AutoConfig
from transformers import TFAutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

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

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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

    print(f'TODO: Remove this later... Cutting down the size of the dataset for testing purposes...')
    train_df = train_df.head(1000)
    test_df = test_df.head(1000)
    
    X_train, y_train = extract_target_feature(train_df)
    X_test, y_test = extract_target_feature(test_df)
    
    model, metrics, results_df = train_model(X_train, X_test, y_train, y_test)
    
    print(f'Saving model...')
    mlflow.sklearn.save_model(model, args.model_output)
    
    print('Saving evauation metrics...')
    with open(Path(args.test_report) / 'metrics.json', 'w') as fp:
        json.dump(metrics, fp)

    print('Saving model results...')
    results_df.to_csv(Path(args.test_results) / 'results_df.csv', index=False)
    
def extract_target_feature(df):
    label_column="Sentiment"
        
    X=df.drop(label_column, axis=1)
    y=df[label_column]

    # return split data
    return X, y


def predict_sentiment(text, tokenizer, model, config):
    encoded_input = tokenizer(text, return_tensors='tf', max_length=512, truncation=True)
    output = model(encoded_input)
    scores = output[0][0].numpy()
    scores = softmax(scores)
    return {config.id2label[i]: scores[i] for i in range(len(scores))}


def train_model(X_train, X_test, y_train, y_test):
    print(f'Starting training of model twitter-roberta-base-sentiment-latest...')

    # Load model
    print(f'Loading model cardiffnlp/twitter-roberta-base-sentiment-latest...')
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    # Evaluate Model
    print(f'Predicting sentiment for test data...')
    results = []
    for text in tqdm(X_test):  # Fortschrittsbalken anzeigen
        sentiment_scores = predict_sentiment(text, tokenizer, model, config)
        results.append(sentiment_scores)

    print(f'Creating DataFrame with results...')
    results_df = pd.DataFrame()
    results_df['Predicted Sentiment'] = results
    results_df['Text'] = X_test
    results_df['True Sentiment'] = y_test
    
    print(f'Evaluation finished')

    print(f'TODO: Calculating metrics...')
    metrics = {}
    # metrics = classification_report(results_df['True Sentiment'], results_df['Predicted Sentiment'], output_dict=True)

    # Log metrics in mlflow
    mlflow.log_param("model", MODEL)
    for label, metric in metrics.items():
        if label == "accuracy":
            mlflow.log_metric(label, metric)
        else:
            for metric_name, metric_value in metric.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"{label}_{metric_name}", metric_value)
    
    print(metrics)

    
    # Confusion Matrix
    #sns.set()
    print("TODO: Creating Confusion Matrix:")
    #plt.figure(figsize=(5,3))
    #sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, fmt='d', cmap='Blues')
    #plt.title('Confusion Matrix')

    # Save Confusion Matrix as image
    #confusion_matrix_image_path = "confusion_matrix.png"
    #plt.savefig(confusion_matrix_image_path)
    #plt.close()

    print("Done")

    # return model
    return model, metrics, results_df

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path of prepped train data")
    parser.add_argument("--test_data", type=str, help="Path of prepped test data")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument("--test_report", type=str, help="Path of test_report")
    parser.add_argument("--test_results", type=str, help="Path of test_results")

    args=parser.parse_args()
    return args

# run script
if __name__ == "__main__":
    mlflow.start_run()
    
    args=parse_args()
    main(args)
    
    mlflow.end_run()
    print('Done!')
