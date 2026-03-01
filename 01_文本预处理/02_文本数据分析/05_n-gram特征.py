"""
n-gram 就是把文本切成连续 n 个单元，用来表示局部顺序的特征
"""
import torch
# 实现
def  n_gram(train_data):
    n_gram=2
    alist=[train_data[i:] for i in range(n_gram)]

    return set(zip(*alist))

# 句子长度规范
from keras.preprocessing import sequence
def padding(inputs):
    max_len=10

    # padding="pre"：补齐的时候默认在前面补齐，如果想在后面补齐：padding="post"
    # truncating="pre"：截断的时候默认在前面补齐，如果想在后面截断：truncating="post"
    return sequence.pad_sequences(inputs,maxlen=max_len,padding="post",truncating="post")

def my_padding(inputs):
    max_len=10
    alist=[]
    for value in inputs:
        if len(value)>10:
            alist.append(value[:max_len])
        else:
            alist.append(value+[0]*(max_len-len(value)))
    return torch.tensor(alist)


if __name__ == '__main__':
    train_data= [1, 3, 2, 1, 5, 3]
    x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
               [2, 32, 1, 23, 1]]
    # print(n_gram(train_data))
    # print(padding(x_train))
    print(my_padding(x_train))
    print(my_padding(x_train).shape)
