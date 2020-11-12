#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_wordcloud(text):
    fontpath = 'NotoSansCJK-Regular.ttc'
    stop_words_en = [u'am', u'is', u'of', u'and', u'the', u'to', u'it', \
                  u'for', u'in', u'as', u'or', u'are', u'be', u'this', u'that', \
                  u'H', u'W', u'by', u'will', u'there', u'was', u'a', u'an', u'Fig']

    wordcloud = WordCloud(background_color="white",
                          font_path=fontpath,
                          width=900,
                          height=500,
                          contour_width=1,
                          contour_color="black",
                          colormap='winter',
                          stopwords=set(stop_words_en)).generate(text)

    #描画
    plt.figure(figsize=(15,20))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    #png出力
    wordcloud.to_file("lee1999_wordcloud.png")

#テキストの読み込み
with open('lee1999nature791seung.txt', 'r', encoding='utf-8') as fi:
    text = fi.read()

create_wordcloud(text)
