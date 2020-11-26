import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer

r = 200 # 基底

def main():
    news = fetch_20newsgroups(data_home="newsgroups", subset="train", remove=("headers", "footers", "quotes"))

    countVectorizer = CountVectorizer(stop_words="english", min_df=2, preprocessor=remove_number, max_features=10000)
    bags = countVectorizer.fit_transform(news.data)

    # df = pd.DataFrame(bags.toarray(), columns=countVectorizer.get_feature_names())
    # print(df.head())
    print("bags shape:" + str(bags.shape))

    V = bags.T
    nmf = NMF(n_components=r, max_iter=400)
    W = nmf.fit_transform(V)
    print("W shape:" + str(W.shape))

def remove_number(tokens):
    r = re.sub("(\d)+|_", "", tokens.lower())
    return r

if __name__ == "__main__":
    main()