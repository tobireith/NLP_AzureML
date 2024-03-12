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
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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


label_column='Sentiment'
def X_y_split(df):
    X=df.drop(label_column, axis=1)
    y=df[label_column]
    return X, y

X_train, y_train = X_y_split(train_df)
X_test, y_test = X_y_split(test_df)

# Feature Encoding for Text-Data using Doc2Vec

# The Doc2Vec-Model needs tagged data and the tagger data needs to be a list of words.
X_train_tokens = X_train['Text'].apply(lambda x: str(x).split())
X_test_tokens = X_test['Text'].apply(lambda x: str(x).split())

# Creation of TaggedDocument-Objects in order to tag the training-data for training the Doc2Vec Model
# Test-Data does not need to be tagged!
tagged_train_data = [TaggedDocument(words=words, tags=[f'train_{i}']) for i, words in enumerate(X_train_tokens)]

# Define and train the model
doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=10)
doc2vec_model.build_vocab(tagged_train_data)
doc2vec_model.train(tagged_train_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

vocab_size = len(doc2vec_model.wv.key_to_index)
print(f"Size of vocabulary: {vocab_size}")

# Transform the training data by using the Doc2Vec-Model
X_train_doc2vec = np.array([doc2vec_model.dv[f'train_{i}'] for i in range(len(X_train_tokens))])
# Transform the test data by using the Doc2Vec-Model
X_test_doc2vec = np.array([doc2vec_model.infer_vector(doc) for doc in X_test_tokens])

# Each Review is now represented by a 100-dimensional vector

# Converting the transformed Training- and Testdata into DataFrames
train_df_transformed = pd.DataFrame(X_train_doc2vec, columns=[f"Doc2Vec_{i}" for i in range(100)])
test_df_transformed = pd.DataFrame(X_test_doc2vec, columns=[f"Doc2Vec_{i}" for i in range(100)])

# Adding y-Column to the DataFrames
train_df_transformed[label_column] = y_train.reset_index(drop=True)
test_df_transformed[label_column] = y_test.reset_index(drop=True)

# Write the results out for the next step.
print("Writing results out...")
train_df_transformed.to_csv((Path(args.output_data_train) / "TrainDataTransformed.csv"), index=False)
test_df_transformed.to_csv((Path(args.output_data_test) / "TestDataTransformed.csv"), index=False)

print("Done!")
