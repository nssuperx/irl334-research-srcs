import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

N = 150 # 語彙数
M = 100 # 記事数
r = 200 # 基底

article_number = 1 # 記事番号
w_num = 10 # 頻度の高いwを，いくつ表示したいか
h_num = 5 # hをいくつ表示したいか

def main():
    # データを準備
    news = fetch_20newsgroups(data_home="newsgroups", subset="all", remove=("headers", "footers", "quotes"))
    X, X_test, label, label_test = train_test_split(news.data, news.target, train_size=M, random_state=0, shuffle=True)
    # X, X_test, label, label_test = train_test_split(news.data, news.target, train_size=0.999, random_state=0, shuffle=True)

    # bag of words を作成
    # countVectorizer = CountVectorizer(stop_words="english", min_df=2, preprocessor=remove_number, max_features=N)
    countVectorizer = CountVectorizer(stop_words="english", preprocessor=remove_number, max_features=None)
    # countVectorizer = CountVectorizer(min_df=2, preprocessor=remove_number, max_features=None)
    bags = countVectorizer.fit_transform(X)

    # 語彙を取得
    features = countVectorizer.get_feature_names()

    n = len(features)
    m = len(X)

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

    # 1記事の語彙カウントのソート
    V_sorted = []
    for i in range(n):
        V_sorted.append((bags.toarray()[article_number][i], features[i]))
    V_sorted.sort(key=lambda x: x[0], reverse=True)

    for i in range(10):
        print(str(V_sorted[i]) + " ", end="")
    print()
    
    # ここで表を作成
    words_list = []
    for i in range(w_num):
        tmp_horizontal_list = []
        tmp_horizontal_list.append(V_sorted[i][1])
        for j in range(h_num):
            tmp_horizontal_list.append(W_sorted[H_sorted[article_number][j][1]][i][1])
        words_list.append(tmp_horizontal_list)

    hidden_values = []
    hidden_values.append("V")
    for i in range(h_num):
        hidden_values.append("h[" + str(i) + "]: " + str(format(H_sorted[article_number][i][0], ".3f")))

    df = pd.DataFrame(words_list, columns=hidden_values)
    print(df)

    fig, ax =plt.subplots()
    tb = ax.table(cellText=words_list, colLabels=hidden_values, loc='center')
    for i in range(len(hidden_values)):
        tb[0,i].set_facecolor('#252525')
        tb[0,i].set_text_props(color='w')
    tb.auto_set_font_size(False)
    tb.set_fontsize(14)
    ax.axis('off')
    ax.axis('tight')
    plt.show()

    print(X[article_number])
    

def remove_number(tokens):
    r = re.sub("(\d)+|_", "", tokens.lower())
    return r

if __name__ == "__main__":
    main()