# 导入torch工具
import json

import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from dask.array import shape
# 导入torch的数据源 数据迭代器工具包
from  torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
import  torch
import torch.nn as nn


# todo: 1.获取常用的字符数量：也就是one-hot编码的去重之后的词汇的总量n

all_letters = string.ascii_letters + " ,;.'"
# print(f'all_letters-->{all_letters}')

n_letters = len(all_letters)
print(f'当前字符的总量--》{n_letters}')

# todo:2. 获取国家名种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名个数
categorynum = len(categorys)
print('国家总数--->', categorynum)

# todo：3. 读取数据到内存
def read_data(filepath):
    # 3.1 定义两个空列表my_list_x（存储人名）, my_list_y（存储国家类型）
    my_list_x , my_list_y = [], []
    # 3.2 读取文件内容
    with open(filepath, encoding='utf-8') as fr:
        for line in fr.readlines():
            if len(line) <= 5:
                continue
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x, my_list_y


# todo：4.构建Dataset类
class NameDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        super().__init__()
        # 获取x
        self.my_list_x = my_list_x
        # 获取y
        self.my_list_y = my_list_y
        # 获取样本的数量
        self.sample_len = len(my_list_x)

    # 获取样本的数量
    def __len__(self):
        return self.sample_len

    # 根据索引取出元素item:代表索引
    def __getitem__(self, item):
        # 1.对异常索引进行修正
        item = min(max(item, 0), self.sample_len-1)
        # 2.根据索引取出样本
        x = self.my_list_x[item]
        # print(f'x-->{x}')
        y = self.my_list_y[item]
        # print(f'y--->{y}')
        # 3.将人名变成one-hot编码的张量形式
        # 3.1 初始化全零的张量 #(3, 57)
        tensor_x = torch.zeros(len(x), n_letters)
        # print(f'tensor_x-->{tensor_x}')
        # 3.2 遍历人名的每一个字母，进行one-hot编码的赋值
        for idx, letter in enumerate(x):
            # print(f'idx--》{idx}')
            # print(f'letter--》{letter}')
            tensor_x[idx][all_letters.find(letter)] = 1
            # print(f'tensor_x--》{tensor_x}')
        # print(f'tensor_x修改之后的：-->{tensor_x}')

        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long)
        return tensor_x, tensor_y


# todo: 5.实例化dataloader对象

def get_dataloader():
    # 读取文档数据
    my_list_x, my_list_y = read_data(filepath='./data/name_classfication.txt')
    # 获取dataset对象
    name_dataset = NameDataset(my_list_x, my_list_y)
    # 封装dataset得到dataloader对象: 会对数据进行增加维度
    train_dataloader = DataLoader(dataset=name_dataset,
                                  batch_size=1,
                                  shuffle=True)

    return train_dataloader


# todo: 6.搭建神经网络(rnn)
class NameRNN(nn.Module):
    #参数解释： num_layers(rnn层数),input_size(输入向量的维度),output_size(全连接层输入维度=分类的数量)
    # ,hidden_size(隐藏层神经元的个数(输出维度)，隐藏层层数)
    def __init__(self,num_layers,input_size,output_size,hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 1.定义rnn层
        self.rnn=nn.RNN(input_size,hidden_size,num_layers)

        # 2.定义全连接层
        self.out=nn.Linear(hidden_size,output_size)

        # 3.定义logSoftmax层
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,x0,h0):
        # print(f'x--->{x.shape}')
        # print(f'h0--->{h0.shape}')
        # x:代表输入的原始数据维度[seq_len, input_size]
        # h0:代表初始化的值，[num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x需要先升维：[seq_len, input_size]--》[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x0, dim=1)  # x.unsqueeze(dim=1)
        # print(f'x1-->{x1.shape}')
        # 将x1和h0送入RNN模型
        output, hn = self.rnn(x1, h0)

        # print(f'output--》{output.shape}')
        # print(f'hn--》{hn.shape}')
        # 获取最后一个单词的隐藏层张量来代表整个句子（人名）的语意
        # 这里可以直接用hn代替
        print(f"output.shape:{output.shape}")
        temp = output[-1]  # [1, 128]
        # 将temp送入输出层:result-->[1, 18]
        result = self.out(temp)
        return self.softmax(result), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

# todo 7:LSTM
class NameLSTM(nn.Module):
    #参数解释： num_layers(rnn层数),input_size(输入向量的维度),output_size(全连接层输入维度=分类的数量)
    # ,hidden_size(隐藏层神经元的个数(输出维度)，隐藏层层数)
    def __init__(self,num_layers,input_size,output_size,hidden_size):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 1.定义rnn层
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers)

        # 2.定义全连接层
        self.linear1=nn.Linear(hidden_size,output_size)

        # 3.定义logSoftmax层
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,x0,h0,c0):
        # print(f'x--->{x.shape}')
        # print(f'h0--->{h0.shape}')
        # x:代表输入的原始数据维度[seq_len, input_size]
        # h0:代表初始化的值，[num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x需要先升维：[seq_len, input_size]--》[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x0, dim=1)  # x.unsqueeze(dim=1)
        # print(f'x1-->{x1.shape}')
        # 将x1和h0送入RNN模型
        output, (hn,c1) = self.lstm(x1, (h0,c0))

        # print(f'output--》{output.shape}')
        # print(f'hn--》{hn.shape}')
        # 获取最后一个单词的隐藏层张量来代表整个句子（人名）的语意
        # 这里可以直接用hn代替
        temp = output[-1]  # [1, 128]
        # 将temp送入输出层:result-->[1, 18]
        result = self.out(temp)
        return self.softmax(result), hn,c1

    def inithidden(self):
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0, c0



my_lr = 1e-3
epochs = 1
def train_rnn():
    # 1.获取数据迭代器对象
    train_dataloader=get_dataloader()
    # 2.定义超参变量，并实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    model=NameRNN(1,input_size,output_size,hidden_size)

    # 3.定义损失函数对象
    criterion=nn.NLLLoss()

    # 4.定义优化器对象
    optimizer=optim.Adam(model.parameters())

    # 5.定义变量记录日志
    start_time = time.time()  # 开始的时间
    total_iter_num = 0  # 已经训练的样本的总个数
    total_loss = 0.0  # 已经训练的样本的损失之和
    total_loss_list = []  # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0  # 已经训练的样本中预测正确的样本的个数
    total_acc_list = []  # 每隔100个样本计算一下平均准确率，画图

    # 6.开始训练
    for epoch in range(epochs):
        for idx,(x,y) in enumerate(train_dataloader):
            h0=model.inithidden()
            output,hn=model(x[0],h0)


            # 计算损失
            loss=criterion(output,y)
            # 梯度清零+反向传播+梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# def my_test():
#     input_size = 57
#     hidden_size = 128
#     output_size = 18
#     data_set=get_dataloader()
#     model=NameRNN(1,input_size,output_size,hidden_size)
#     for x,y in data_set:
#         softmax,hn=model(x[0],model.inithidden())
#         print(f"softmax:{softmax}")
#         break
#     # train_rnn()
if __name__ == '__main__':
    # my_test()
    ...
