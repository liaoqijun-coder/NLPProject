"""
获取高频词云
"""

import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
import jieba.posseg as pseg
from wordcloud import WordCloud


def get_a_list(text):
    '''

    :param text:一个句子
    :return:
    '''
    r=[]
    for value in pseg.lcut(text):
        if value.flag=="a":
            r.append(value.word)
            # print(value)
            # print(value.word)

    return r

def get_cloud_word(word):
    wordcloud=WordCloud(font_path="./cn_data/SimHei.ttf",max_words=100,background_color="white")
    # 2. 获取展示词云的数据，字符串形式，空格隔开
    cloud_word=' '.join(word)
    # 3.生成词云
    wordcloud.generate(cloud_word)

    # 4.
    plt.figure()
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def main():
    # 1.读取数据集
    train_data=pd.read_csv("./cn_data/train.tsv",sep="\t")

    # print(train_data.head())
    # 2.获取训练数据的正样本
    p_train_sentence=train_data[train_data["label"]==1]["sentence"]
    # print(p_train_sentence)

    # 3. 获取正样本中所用的形容词
    p_train_adj=list(chain(*map(lambda x: get_a_list(x), p_train_sentence)))

    print(p_train_adj)
    get_cloud_word(p_train_adj)

if __name__ == '__main__':
    main()
