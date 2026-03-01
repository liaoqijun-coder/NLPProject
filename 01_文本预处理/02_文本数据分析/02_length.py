import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hist():
    # 1.获取数据

    train_data = pd.read_csv("./cn_data/train.tsv", sep='\t')
    dev_data = pd.read_csv("./cn_data/dev.tsv", sep='\t')

    # 2.新增一列，记录每个句子的长度
    train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))

    # 3.绘制长度分布图(柱状图)
    sns.countplot(data=train_data, x="sentence_length", hue="label")
    plt.show()

    # 4.绘制长度分布图(曲线图)
    sns.displot(data=train_data, x="sentence_length", hue="label", kde=True)
    plt.xticks([])
    plt.show()

def strip_plt():
    train_data = pd.read_csv("./cn_data/train.tsv", sep='\t')
    dev_data = pd.read_csv("./cn_data/dev.tsv", sep='\t')

    # 2.新增一列，记录每个句子的长度
    train_data["sentence_length"] = list(map(lambda x: len(x), train_data["sentence"]))

    # 3.绘制散点图
    sns.stripplot(data=train_data,y="sentence_length",x="label",hue="label")
    plt.show()

if __name__ == '__main__':
    strip_plt()

