import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

train1 = pd.read_csv('train1_.csv', header=None)
train2 = pd.read_csv('train2_.csv', header=None)

pre1 = pd.read_csv('pre1.csv', header=None)
pre2 = pd.read_csv('pre2.csv', header=None)

y1 = pd.read_csv('f1_.csv', header=None)
y2 = pd.read_csv('f2_.csv', header=None)


x_prime = train2
y = pd.Categorical(y2[3]).codes
x_prime_train, x_prime_test, y_train, y_test = train_test_split(x_prime, y, test_size=0.3, random_state=0)

feature_pairs = [list(range(0, 51))]
for i, pair in enumerate(feature_pairs):
    # 准备数据
    x_train = x_prime_train[pair]
    x_test = x_prime_test[pair]

    # 决策树学习
    model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=7, oob_score=True)
    model.fit(x_train, y_train)
    y_pre = model.predict(pre2)
    for j in range(len(y_pre)):
        y_pre[j] = y_pre[j] + 1



    # 训练集上的预测结果
    y_train_pred = model.predict(x_train)
    acc_train = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict(x_test)
    acc_test = accuracy_score(y_test, y_test_pred)
    print('OOB Score:', model.oob_score_)
    print('\t训练集准确率: %.4f%%' % (100 * acc_train))
    print('\t测试集准确率: %.4f%%\n' % (100 * acc_test))
    data = pd.DataFrame(y_pre)
    data.to_csv('手机上网稳定性.csv', index=None)


