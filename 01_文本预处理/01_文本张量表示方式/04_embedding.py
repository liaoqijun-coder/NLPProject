# 实验：nn.Embedding层词向量可视化分析
# 1 对句子分词 word_list
# 2 对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3 创建nn.Embedding层，查看每个token的词向量数据
# 4 创建SummaryWriter对象, 可视化词向量
#   词向量矩阵embd.weight.data 和 词向量单词列表my_token_list添加到SummaryWriter对象中
#   summarywriter.add_embedding(embd.weight.data, my_token_list)
# 5 通过tensorboard观察词向量相似性
# 6 也可通过程序，从nn.Embedding层中根据idx拿词向量
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import jieba
def dm():
    # 1 对句子分词 word_list
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]
    word_list=[]
    for s in sentences:
        word_list.append(jieba.lcut(s))

    # print(word_list)

    #  2.对每个词进行词表映射
    my_tokenize=Tokenizer()
    my_tokenize.fit_on_texts(word_list)
    print(my_tokenize.word_index)

    # 3.获得所有词汇
    all_words=my_tokenize.index_word.values()
    # print(all_words)
    # print(len(all_words))

    # 4.初始化embedding层
    embd=nn.Embedding(num_embeddings=len(all_words),embedding_dim=8)

    # print(embd.weight.data)
    # print(embd.weight.shape)

    # 4 可视化展示
    # my_summary = SummaryWriter()

    # my_summary.add_embedding(embd.weight.data, all_words)

    # my_summary.close()
    # 5.取出每个单词的对应的向量
    print(embd.weight.data)
    for i in range(len(my_tokenize.index_word)):
        print(f"当前词汇：{my_tokenize.index_word[i+1]}")
        print(f"向量{embd(torch.tensor(i)).detach().numpy()}")
        print("*"*80)



if __name__ == '__main__':
    dm()