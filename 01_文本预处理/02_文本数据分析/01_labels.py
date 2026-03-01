# 获取标签的数量分布

# 导入必备工具包
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# 1.获取数据

train_data=pd.read_csv("./cn_data/train.tsv",sep='\t')
dev_data=pd.read_csv("./cn_data/dev.tsv",sep='\t')
# print(train_data.head())
# print(dev_data.head())

# 2.使用sns.countplt画出标签数量分布
plt.figure()
sns.countplot(data=train_data,x="label",hue="label")
plt.show()

