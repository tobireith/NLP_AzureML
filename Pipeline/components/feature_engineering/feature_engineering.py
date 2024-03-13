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


parser=argparse.ArgumentParser("prep")
parser.add_argument("--input_data", type=str, help="Name of the folder containing input data for this operation")
parser.add_argument("--output_data", type=str, help="Name of folder we will write results out to")

args=parser.parse_args()

print("Performing feature engineering...")

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

# Feature engineering steps

# Remove missing values
df.dropna(inplace=True)

# Map Score to Sentiment
def map_score_to_sentiment(score):
  match score:
    case 1 | 2: # Negaitve
      return 0
    case 3: # Neutral
      return 1
    case 4 | 5: # Positive
      return 2
     
df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)

# Feature Selection
# Drop all other features than 'Sentiment' and 'Text'
df.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Score', 'Summary'], axis=1, inplace=True)

# Write the results out for the next step.
print("Writing results out...")
df.to_csv((Path(args.output_data) / "FeatureEngineering.csv"), index=False)

print("Done!")
