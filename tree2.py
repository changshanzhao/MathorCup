#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    feature = '手机上网整体满意度', '网络覆盖与信号强度', '手机上网速度', '手机上网稳定性', ' 居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他,请注明', '网络信号差 / 没有信号', '显示有信号上不了网','上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2', '爱奇艺', '优酷', '腾讯视频', '芒果TV','抖音', '快手', ' 火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', ' 穿越火线', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', ' 全部游戏都卡顿','微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博, 拼多多', '其他，请注明.5', '全部网页或APP都慢', '脱网次数'
    path = 'f2_.csv'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x_prime = data[list(range(1, 55))]
    y = pd.Categorical(data[0]).codes
    x_prime_train, x_prime_test, y_train, y_test = train_test_split(x_prime, y, test_size=0.3, random_state=0)
    pairs = [c for c in combinations(range(1,55), 54)]
    feature_pairs = []
    num = len(pairs)
    for i in range(num):
        feature_pairs.append(list(pairs[i]))

    for i, pair in enumerate(feature_pairs):
        # 准备数据
        x_train = x_prime_train[pair]
        x_test = x_prime_test[pair]

        # 决策树学习
        model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, oob_score=True)
        model.fit(x_train, y_train)
        importance = model.feature_importances_
        for j in range(0,54,3):
            print(feature[j], ':', importance[j], feature[j+1], ':', importance[j+1]
                  , feature[j+2], ':', importance[j+2])


        # 训练集上的预测结果
        y_train_pred = model.predict(x_train)
        acc_train = accuracy_score(y_train, y_train_pred)
        y_test_pred = model.predict(x_test)
        acc_test = accuracy_score(y_test, y_test_pred)
        print('OOB Score:', model.oob_score_)
        print('\t训练集准确率: %.4f%%' % (100*acc_train))
        print('\t测试集准确率: %.4f%%\n' % (100*acc_test))