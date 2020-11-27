import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

n = 15000 # 語彙数
m = 10000 # 記事数
r = 200 # 基底

article_number = 1 # 記事番号
w_num = 10 # 頻度の高いwを，いくつ表示したいか
h_num = 5 # hをいくつ表示したいか

def main():
    # データを準備
    news = fetch_20newsgroups(data_home="newsgroups", subset="all", remove=("headers", "footers", "quotes"))
    X, X_test, label, label_test = train_test_split(news.data, news.target, train_size=m, random_state=0, shuffle=True)

    # bag of words を作成
    countVectorizer = CountVectorizer(stop_words="english", min_df=2, preprocessor=remove_number, max_features=n)
    bags = countVectorizer.fit_transform(X)

    # 語彙を取得
    features = countVectorizer.get_feature_names()

    print("bags shape:" + str(bags.shape))
    
    # 論文と揃えるため，転置する
    V = bags.T

    # NMF
    nmf = NMF(n_components=r, max_iter=400, random_state=2)
    W = nmf.fit_transform(V)
    H = nmf.components_

    print("W shape:" + str(W.shape))
    print("H shape:" + str(H.shape))

    # 気合で行列を整形
    # タプルにしてソートしてる
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
    
    # ここで表を作成
    words_list = []
    for i in range(w_num):
        tmp_horizontal_list = []
        for j in range(h_num):
            tmp_horizontal_list.append(W_sorted[H_sorted[article_number][j][1]][i][1])
        words_list.append(tmp_horizontal_list)

    hidden_values = []
    for i in range(h_num):
        hidden_values.append(H_sorted[article_number][i][0])

    df = pd.DataFrame(words_list, columns=hidden_values)
    print(df)

    print(X[article_number])
    

def remove_number(tokens):
    r = re.sub("(\d)+|_", "", tokens.lower())
    return r

if __name__ == "__main__":
    main()