"""
获取不同词汇词汇统计

"""

import jieba
import pandas as pd
from itertools import chain
def vocab_count():
    # 1.获取数据
    train_data=pd.read_csv("./cn_data/train.tsv",sep='\t')

    # 2.
    train_count=set(chain(*map(lambda x:jieba.lcut(x),train_data['sentence'])))
    print(len(train_count))

if __name__ == '__main__':
    vocab_count()