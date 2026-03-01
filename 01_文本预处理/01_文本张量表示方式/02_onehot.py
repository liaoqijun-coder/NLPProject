# 导入keras中的词汇映射器Tokenizer
# 构建词汇表（Vocabulary）
# 统计训练文本中所有出现的词（或字），并为每个词分配一个唯一的整数 ID。
from tensorflow.keras.preprocessing.text import Tokenizer

# 用于保存对象
import joblib

def dm01():
    # 准备语料库
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
    # 2.创建Tokenizer对象
    my_tokenizer=Tokenizer()
    my_tokenizer.fit_on_texts(vocabs)

    # 3.打印结果
    print(my_tokenizer.word_index)
    print(my_tokenizer.index_word)

    # 4.具体one-hot的动作
    for i in range(len(vocabs)):
        # 创建全0列表
        zero_list=[0]*len(vocabs)
        zero_list[i]=1
        print(f"{my_tokenizer.index_word[i+1]}的编码为{zero_list}")
    # 5. 保存tokenizer,方便下次使用
    joblib.dump(my_tokenizer, "model/tokenizer.pth")
    print("模型保存成功")

def dm02():
    # 加载模型
    my_tokenizer=joblib.load("model/tokenizer.pth")
    # 准备语料库
    vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}

    # 4.具体one-hot的动作
    for i in range(len(vocabs)):
        # 创建全0列表
        zero_list=[0]*len(vocabs)
        zero_list[i]=1
        print(f"{my_tokenizer.index_word[i+1]}的编码为{zero_list}")
if __name__ == '__main__':
    # dm01()
    dm02()