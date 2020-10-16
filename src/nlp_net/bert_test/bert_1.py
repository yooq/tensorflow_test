# -*- coding: UTF-8 -*-
import pandas as pd

train=pd.read_csv("nCoV_100k_train.labled.csv")
test=pd.read_csv("nCov_10k_test.csv")
print(train.head())

columns=['id','date','user','content','image','vedio','target']
train.columns=columns
test.columns=columns[:-1]
train.drop(['id','date','user','image','vedio'],axis = 1,inplace = True)
print(train.head())

train[train.isnull().T.any().T]  #找出含有nan的所有行
train.dropna(axis=0, how='any', inplace=True)
train[train.isnull().T.any().T]  #找出含有nan的所有行

print('len:',len(test[test.isnull().T.any().T]))


import re
def clean(text):
    text = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", text)  # 去除正文中的@和回复/转发中的用户名
    text = re.sub(r"\[\S+\]", "", text)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text = re.sub(URL_REGEX, "", text)       # 去除网址
    text = text.replace("转发微博", "")       # 去除无意义的词语
    text = re.sub(r"\s+", " ", text) # 合并正文中过多的空格
    return text.strip()


train['content']=train['content'].apply(clean)
print(train['content'][15:25])

test.fillna("", inplace = True)#以0填充
test['content']=test['content'].apply(clean)
train['target'].value_counts()

train_1=train.loc[(train['target']=='-1')]
train0=train.loc[(train['target']=='0')]
train1=train.loc[(train['target']=='1')]
train = pd.concat([train_1,train_1,train_1,train0,train1,train1])

train['target'].value_counts()
train.reset_index(drop=True,inplace=True)
train.head()

train=train.sample(frac=1)
train.head()

# 划分训练集和验证集
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split( train["content"], train["target"], test_size=0.1, random_state=42)

