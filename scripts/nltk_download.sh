#!/bin/bash

# Download NLTK packages without interactive prompts
python -m nltk.downloader -q popular
python -m nltk.downloader -q stopwords
python -m nltk.downloader -q tagsets