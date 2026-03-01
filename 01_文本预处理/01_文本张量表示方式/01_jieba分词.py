import jieba
import fasttext
# todo 1. jieba精确模式分词：试图将句子最精确地切开，适合文本分析.

def dm01():
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # result1=jieba.cut(content,cut_all=False)
    # print(next(result1))
    result1=jieba.lcut(content)
    print(result1)


# todo 2. 全模式分词全模式分词:# 把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能消除歧义

def dm02():
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # result1=jieba.cut(content,cut_all=False)
    # print(next(result1))
    result1=jieba.lcut(content,cut_all=True)

    print(result1)

# todo 3:搜索引擎模式分词:# 在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词
def dm03():
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # result1=jieba.cut(content,cut_all=False)
    # print(next(result1))
    result1=jieba.lcut_for_search(content,cut_all=True)

    print(result1)
# todo 4:使用用户自定义词典:# 添加自定义词典后, jieba能够准确识别词典中出现的词汇，提升整体的识别准确率

def dm04():
    content = "传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能"
    # result1=jieba.cut(content,cut_all=False)
    # print(next(result1))
    result1=jieba.lcut(content,cut_all=False)  # 没有使用自定义字典
    print(result1)
    jieba.load_userdict("./data/user_dict")
    result2=jieba.lcut(content)
    print(result2)
if __name__ == '__main__':
    # dm01()
    dm04()
    # dm02()