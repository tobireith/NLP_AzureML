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
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer


parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data", type=str, help="Name of folder we will write results out to")

args=parser.parse_args()

print("Performing text preprocessing...")

lines=[
    f"Input data path: {args.input_data}",
    f"Output data path: {args.output_data}",
]

for line in lines:
    print(line)

print(os.listdir(args.input_data))

file_list=[]
for filename in os.listdir(args.input_data):
    print("Reading file: %s ..." % filename)
    with open(os.path.join(args.input_data, filename), "r") as f:
        input_df=pd.read_csv((Path(args.input_data) / filename))
        file_list.append(input_df)

# Concatenate the list of Python DataFrames
df=pd.concat(file_list)

# Text preprocessing steps

# Compile patterns for performance
TAG_RE = re.compile(r'<[^>]+>')
PUNCT_NUM_RE = re.compile('[^a-zA-Z]')
SINGLE_CHAR_RE = re.compile(r"\s+[a-zA-Z]\s+")
MULTI_SPACE_RE = re.compile(r'\s+')

# Initialize objects outside of function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
important_tags = ['FW', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', \
                      'RB', 'RBR', 'RBS', 'RP', 'UH',  \
                      'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

# Text preprocessing
def preprocess_text(sen):
    # Lowercasing
    sentence = sen.lower()
    # Remove html tags
    sentence = TAG_RE.sub('', sentence)
    # Remove punctuations and numbers
    sentence = PUNCT_NUM_RE.sub(' ', sentence)
    # Single character removal
    sentence = SINGLE_CHAR_RE.sub(' ', sentence)
    # Remove multiple spaces
    sentence = MULTI_SPACE_RE.sub(' ', sentence).strip()

    # Tokenize
    tokens = nltk.word_tokenize(sentence)
    # POS Tagging for all tokens in one go
    tagged_tokens = nltk.pos_tag(tokens)

    # Lemmatize and remove stopwords and non-important POS tags
    filtered_tokens = [
        lemmatizer.lemmatize(word) 
        for word, tag in tagged_tokens
        if word not in stop_words
        and tag in important_tags
    ]
    return filtered_tokens

# Now apply preprocess_text to all rows in the 'Text' column
# and save the processed text back into the 'Text' column
df['Text'] = df['Text'].apply(preprocess_text)

# Write the results out for the next step.
print("Writing results out...")
df.to_csv((Path(args.output_data) / "TextPreprocessed.csv"), index=False)

print("Done!")
