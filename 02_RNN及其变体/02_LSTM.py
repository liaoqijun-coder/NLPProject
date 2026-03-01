import torch
import torch.nn as nn
def dm01_rnn_base():
    input_size=5
    sequence_len=1
    hidden_size=3
    batch_size=3
    num_layers=1
    # 参数解释
    # 参1：输入向量的维度，参2：隐藏层神经元的个数(输出维度)，隐藏层层数
    rnn=nn.LSTM(input_size,hidden_size,num_layers=1)

    # rnn需要2个参数：1.X:输入的数据 2.:上一时刻隐藏状态ht-1
    # 参1：每个句子的长度(每个样本的长度) 参2：句子的个数 参3：输入向量维度
    x=torch.randn(sequence_len,batch_size,input_size)
    # 参1： 隐藏层的层数 参2：一次输入句子的个数(一个批次送入几个样本) 参3：隐藏层神经元的个数
    h0=torch.randn(num_layers,batch_size,hidden_size)

    output,h1=rnn(x,h0)

    print(output)

    print("*"*100)
    print(h1)