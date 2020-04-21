# Script for preprocessing the New York Times dataset and saving it as pickle files
# Run with <py nyt_to_numpy.py>

import math
import numpy as np
import pickle
import random
from sklearn.feature_extraction.text import CountVectorizer


# New York Times dataset directory
nyt_dir = "data/nytimes/"


# Convert corpus to sklearn compatible format
display = 1000
doc_count = 0
with open(nyt_dir+"nytimes_news_articles.txt", "r", encoding="latin-1") as src, open(nyt_dir+"nytimes_data.txt", "w", encoding="latin-1") as dest:
    doc = ""
    for line in src:
        if line.startswith("URL"):
            dest.write(doc + "\n")
            doc = ""
            doc_count += 1
            if doc_count % display == 0:
                print("Documents processed:", doc_count)
        else:
            doc = doc + " " + line.rstrip("\n\r")
    dest.write(doc)
    print("Total documents:", doc_count)


# Vectorize corpus to bag of words representation and split into train/test sets
with open(nyt_dir+"nytimes_data.txt", "r", encoding="latin-1") as src:
    lines = src.readlines()
    
    random.seed(503)
    test_ind = random.sample(range(doc_count), math.floor(doc_count*0.2))
    train = [line for i, line in enumerate(lines) if i not in test_ind]
    test = [lines[i] for i in test_ind]
    print("Training set count:", len(train))
    print("Test set count:", len(test))
    
    # Keep only alpha characters, remove stop words, remove words with document frequencies less than 1% of dataset
    vectorizer = CountVectorizer(strip_accents="ascii", stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=0.01)
    data_tr = vectorizer.fit_transform(train)
    print("Word count:", data_tr.shape[1])
    
    data_te = vectorizer.transform(test)


# Save vocabulary as dictionary
words = vectorizer.get_feature_names()
vocab = {words[i]: i for i in range(0,len(words))}
with open(nyt_dir+"vocab.pkl", "wb") as dest:
    pickle.dump(vocab, dest)


# Save train and test sets
np.save(nyt_dir+"train.txt.npy", data_tr.toarray(), allow_pickle=True)
np.save(nyt_dir+"test.txt.npy", data_te.toarray(), allow_pickle=True)