import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

n = 300 # 語彙数
m = 100 # 記事数
r = 200 # 基底

def main():
    news = fetch_20newsgroups(data_home="newsgroups", subset="test", remove=("headers", "footers", "quotes"))
    X, X_test, label, label_test = train_test_split(news.data, news.target, train_size=m, random_state=0, shuffle=True)

    countVectorizer = CountVectorizer(stop_words="english", min_df=2, preprocessor=remove_number, max_features=n)
    bags = countVectorizer.fit_transform(X)

    features = countVectorizer.get_feature_names()
    # df = pd.DataFrame(bags.toarray(), columns=features)
    # print(df.head())
    print("bags shape:" + str(bags.shape))

    V = bags.T
    nmf = NMF(n_components=r, max_iter=400, random_state=2)
    W = nmf.fit_transform(V)
    H = nmf.components_
    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))
    
    df = pd.DataFrame(W.T, columns=features)
    print(df.head())


    W_sorted = []
    for i in range(r):
        tmp_list = []
        for j in range(n):
            tmp_list.append((W[j][i], features[j]))
        tmp_list.sort(key=lambda x: x[0], reverse=True)
        W_sorted.append(tmp_list)

    H_sorted = []
    for i in range(m):
        tmp_list = []
        for j in range(r):
            tmp_list.append((H[j][i], j))
        tmp_list.sort(key=lambda x: x[0], reverse=True)
        H_sorted.append(tmp_list)
    
    for i in range(8):
        print("%.2f %s" % (W_sorted[0][i][0], W_sorted[0][i][1]))

    for i in range(20):
        print(H_sorted[0][i])

    for i in range(8):
        print("%.2f %s %.2f %s %.2f %s %.2f %s " % (W_sorted[167][i][0], W_sorted[167][i][1], W_sorted[181][i][0], W_sorted[181][i][1], W_sorted[50][i][0], W_sorted[50][i][1], W_sorted[18][i][0], W_sorted[18][i][1]))

    print(X[0])
    

def remove_number(tokens):
    r = re.sub("(\d)+|_", "", tokens.lower())
    return r

if __name__ == "__main__":
    main()