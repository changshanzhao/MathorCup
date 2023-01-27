# 过拟合太严重了，弃用了

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split  # 引入训练集、测试集划分函数
import torch
import torch.nn.functional as Fun

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 0. 超参数设置
lr = 0.00002
epochs = 300
n_feature = 22
n_hidden = 300
n_output = 10

# 1. 数据准备
train1 = pd.read_csv('train1_.csv', header=None)
# train2 = pd.read_csv('train2_.csv', header=None)

pre1 = pd.read_csv('pre1.csv', header=None)
# pre2 = pd.read_csv('pre2.csv', header=None)

y1 = pd.read_csv('f1_.csv', header=None)
# y2 = pd.read_csv('f2_.csv', header=None)


x_prime = train1
y = pd.Categorical(y1[1]).codes
x_p_train, x_p_test, y_train, y_test = train_test_split(x_prime, y, test_size=0.2, random_state=22)
x_train = np.array(x_p_train)
x_pre = np.array(pre1)
x_test = np.array(x_p_test)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)
x_pre = torch.FloatTensor(x_pre)


# 2. 定义BP神经网络
class bpnnModel(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(bpnnModel, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 定义隐藏层网络
        self.out = torch.nn.Linear(n_hidden, n_output)  # 定义输出层网络


    def forward(self, x):
        x = Fun.relu(self.hidden(x))  # 隐藏层的激活函数,采用relu,也可以采用sigmod,tanh
        out = Fun.softmax(self.out(x), dim=1)  # 输出层softmax激活函数
        return out

# 3. 定义优化器损失函数
net = bpnnModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器选用随机梯度下降方式
loss_func = torch.nn.CrossEntropyLoss()  # 对于多分类一般采用的交叉熵损失函数

# 4. 训练数据
loss_steps = np.zeros(epochs)
accuracy_steps = np.zeros(epochs)
for epoch in range(epochs):
    y_pred = net(x_train)  # 前向过程
    loss = loss_func(y_pred, y_train)  # 输出与label对比
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播
    optimizer.step()  # 使用梯度优化器
    loss_steps[epoch] = loss.item()  # 保存loss
with torch.no_grad():
    y_pred = net(x_test)
    y0 = net(x_pre)
    y = torch.argmax(y0, dim=1)
    correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
    accuracy_steps[epoch] = correct.mean()
print("预测准确率", accuracy_steps[-1])

# 5 绘制损失函数和精度
fig_name = '网络覆盖与信号强度(语音)'
fontsize = 15
fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 12), sharex=True)
ax1.plot(accuracy_steps)
ax1.set_ylabel("test accuracy", fontsize=fontsize)
ax1.set_title(fig_name, fontsize='xx-large')
ax2.plot(loss_steps)
ax2.set_ylabel("train loss", fontsize=fontsize)
ax2.set_xlabel("epochs", fontsize=fontsize)
plt.tight_layout()
plt.savefig(fig_name + '.png')
plt.show()
for j in range(len(y)):
    y[j] = y[j] + 1
data = pd.DataFrame(y)
data.to_csv('BP网络覆盖与信号强度(语音).csv', index=None)