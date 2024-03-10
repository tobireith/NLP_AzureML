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
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import 	WordNetLemmatizer
from nltk.corpus import wordnet


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

# Map NLTK POS-TAG to WordNet POS-TAG Format
def map_pos_tag_nltk_to_wordnet(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        # Return noun by default because this is the default value for the lemmatize call in the WordNet Lemmatizer
        return wordnet.NOUN 

# Compile patterns for data cleaning
TAG_RE = re.compile(r'<[^>]+>')
PUNCT_NUM_RE = re.compile('[^a-zA-Z]')
SINGLE_CHAR_RE = re.compile(r"(?:\s|^)[a-zA-Z](?=\s|$)")
MULTI_SPACE_RE = re.compile(r'\s+')

# Initialize objects outside of function
stop_words = set(stopwords.words('english'))

important_tags = [
    'FW', # foreign words
    'JJ', 'JJR', 'JJS', # adjectives
    'NN', 'NNP', 'NNS', 'NNPS', # nouns
    'RB', 'RBR', 'RBS', # adverbs
    'RP', # particles
    'UH', # interjections
    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ' # verbs
]

lemmatizer = WordNetLemmatizer()

# Text preprocessing
def preprocess_text(sentence):
    # Lowercasing
    sentence = sentence.lower()
    # Remove HTML tags
    sentence = TAG_RE.sub('', sentence)
    # Remove punctuations and numbers
    sentence = PUNCT_NUM_RE.sub(' ', sentence)
    # Single character removal
    sentence = SINGLE_CHAR_RE.sub('', sentence)
    # Remove multiple spaces
    sentence = MULTI_SPACE_RE.sub(' ', sentence).strip()

    # Tokenization
    tokens = nltk.word_tokenize(sentence)

    # Remove stopwords
    tokens_no_stopwords = [word for word in tokens if word not in stop_words]

    # POS Tagging for all remaining tokens in one go
    tokens_pos_tagged = nltk.pos_tag(tokens_no_stopwords)

    # Filter by POS tags
    tokens_pos_filtered = [
        (word, pos_tag)
        for word, pos_tag in tokens_pos_tagged
        if pos_tag in important_tags
    ]

    # Lemmatization
    tokens_lemmatized = [
        lemmatizer.lemmatize(
            word, pos=map_pos_tag_nltk_to_wordnet(pos_tag)
        )
        for word, pos_tag in tokens_pos_filtered
    ]

    processed_sentence = ' '.join(tokens_lemmatized)
    return processed_sentence

# Now apply preprocess_text to all rows in the 'Text' column
# and save the processed text back into the 'Text' column
df['Text'] = df['Text'].apply(preprocess_text)

# Write the results out for the next step.
print("Writing results out...")
df.to_csv((Path(args.output_data) / "TextPreprocessed.csv"), index=False)

print("Done!")
