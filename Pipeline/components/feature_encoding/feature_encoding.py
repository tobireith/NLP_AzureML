import argparse
from pathlib import Path
import os
import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import sys
import timeit
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data_train", type=str, help="Name of the folder containing input train data for this operation")
parser.add_argument("--input_data_test", type=str, help="Name of the folder containing input test data for this operation")
parser.add_argument("--output_data_train", type=str, help="Name of folder we will write modified training results out to")
parser.add_argument("--output_data_test", type=str, help="Name of folder we will write modified test results out to")

args=parser.parse_args()

print("Performing feature encoding...")

print(os.listdir(args.input_data_train))

train_file_list=[]
for filename in os.listdir(args.input_data_train):
    print("Reading file: %s ..." % filename)
    with open(os.path.join(args.input_data_train, filename), "r") as f:
        input_df=pd.read_csv((Path(args.input_data_train) / filename))
        train_file_list.append(input_df)

# Concatenate the list of Python DataFrames
train_df=pd.concat(train_file_list)

print(os.listdir(args.input_data_test))

test_file_list=[]
for filename in os.listdir(args.input_data_test):
    print("Reading file: %s ..." % filename)
    with open(os.path.join(args.input_data_test, filename), "r") as f:
        input_df=pd.read_csv((Path(args.input_data_test) / filename))
        test_file_list.append(input_df)

# Concatenate the list of Python DataFrames
test_df=pd.concat(test_file_list)


label_column='Score'
def X_y_split(df):
    X=df.drop(label_column, axis=1)
    y=df[label_column]

    # return split data
    return X, y

X_train, y_train = X_y_split(train_df)
X_test, y_test = X_y_split(test_df)

# Feature Encoding steps

# Set the parameters for TF-IDF Vectorization
MAX_FEATURES = 1000
MIN_DF = 0.01
MAX_DF = 0.99
NGRAM_RANGE = (1,3)

# Create the TF-IDF-Vectorizer
# Limit the features by using min and max df.
vectorizer_tfidf = TfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF, max_df=MAX_DF)

# Use fit_transform on the training data
X_train_transformed = vectorizer_tfidf.fit_transform(X_train['Text'])
# transform the test data by using the Vectorizer
X_test_transformed = vectorizer_tfidf.transform(X_test['Text'])

# Converting the transformed Training- and Testdata into DataFrames
train_df_transformed = pd.DataFrame(X_train_transformed.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
test_df_transformed = pd.DataFrame(X_test_transformed.toarray(), columns=vectorizer_tfidf.get_feature_names_out())

# Adding the Labels to the DataFrames
train_df_transformed[label_column] = y_train.reset_index(drop=True)
test_df_transformed[label_column] = y_test.reset_index(drop=True)

# Write the results out for the next step.
print("Writing results out...")
train_df_transformed.to_csv((Path(args.output_data_train) / "TrainDataTransformed.csv"), index=False)
test_df_transformed.to_csv((Path(args.output_data_test) / "TestDataTransformed.csv"), index=False)

print("Done!")
