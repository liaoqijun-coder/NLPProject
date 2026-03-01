import fasttext

# 1.训练词向量
def dm01():
    # 模型训练
    model=fasttext.train_unsupervised("./data/fil9",epoch=1)
    # 模型保存
    model.save_model("./model/fil9.bin")

# 2.获取某个词的词向量和检验模型效果
def dm02():
    # 1.加载模型
    model=fasttext.load_model("model/fil9.bin")

    # 2.查看某个词的词向量
    # result=model.get_word_vector("the")
    # print(result)
    # print(type(result),result.shape)

    # 预测临近词
    result2=model.get_nearest_neighbors("dog")
    print(result2)

# 训练词向量模型:修改参数
def dm_fasttext_03():
    # 直接开始训练：以非监督的方式进行
    model = fasttext.train_unsupervised('./data/ai20aa',"cbow", dim=100, lr=0.1, epoch=1)
    # 保存模型
    model.save_model('./data/ai20_fil9_new.bin')
if __name__ == '__main__':
    # dm01()
    dm02()