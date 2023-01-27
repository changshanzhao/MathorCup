import pandas as pd

# 读取数据
df1 = pd.read_csv('./data/附件1语音业务用户满意度数据.csv')
df2 = pd.read_csv('./data/附件2上网业务用户满意度数据.csv')
df3 = pd.read_csv('./data/附件3语音业务用户满意度预测数据.csv')
df4 = pd.read_csv('./data/附件4上网业务用户满意度预测数据.csv')

# 各个文件的指标名称
c1, c2, c3, c4 = list(df1.columns), list(df2.columns), list(df3.columns), list(df4.columns)
# [markdown]
# 整体满意度与三个方面（网络覆盖与信号强度、语音通话清晰度、语音通话稳定性）之间的关系

'''
可以看出，三个方面的评分与整体满意度是高度相关的（当然，这是显然的结果）
所以我们下面可以仅考虑整体满意度的评分与其他影响因素之间的关系即可。
'''

df1.iloc[:, 1:5].corr()

df2.iloc[:, 1:5].corr()

df3.iloc[:,1:5].corr()

df4.iloc[:,1:5].corr()
# [markdown]
# 数据编码

'''
由于原始数据中含有较多的非数值型数据（顺序编码）

为了便于理解，我们将改变数据的原始编码方式（原始表述方式是用1/2，或者-1/N来表示“是否”，我们习惯上用0/1表示）
手机型号在一定程度上反映的是用户个人的信息，不考虑
'''

# 附件一
temp1 = df1.copy()
del temp1['用户id'], temp1['用户描述'], temp1['用户描述.1'], temp1['终端品牌'], temp1['终端品牌类型']

d4 = {1: 1, 2: 0}  # 是否遇到过网络问题 0 1
d5_12 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 98: 1}
d13_19 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 98: 1}
d245g = {'2G': 0, '4G': 1, '5G': 2}
d_yuyin = {'GSM': 0, 'CSFB': 1, 'VOLTE': 2, 'VoLTE': 2, 'EPSFB': 3, 'VONR': 4}
d_yesno = {0: 0, '是': 1, '否': 0}
d_level = {'未评级': 1, '准星': 2, '一星': 3, '二星': 4, '三星': 5, '白金卡': 6, '钻石卡': 7, '银卡': 8, '金卡': 9}

temp1.iloc[:, 4] = temp1.iloc[:, 4].map(d4)

for i in range(5, 13):
    temp1.iloc[:, i] = temp1.iloc[:, i].map(d5_12)

for i in range(13, 20):
    temp1.iloc[:, i] = temp1.iloc[:, i].map(d13_19)

temp1['重定向次数'] = temp1['重定向次数'].fillna(0)
temp1['重定向驻留时长'] = temp1['重定向驻留时长'].fillna(0)
temp1['4\\5G用户'] = temp1['4\\5G用户'].map(d245g)
temp1['语音方式'] = temp1['语音方式'].map(d_yuyin)
temp1['外省流量占比'] = temp1['外省流量占比'].fillna(0)

for i in ['是否关怀用户', '是否去过营业厅', '是否4G网络客户（本地剔除物联网）', '是否5G网络客户', '是否实名登记用户']:
    temp1[i] = temp1[i].fillna(0).map(d_yesno)

temp1['客户星级标识'] = temp1['客户星级标识'].fillna(0).map(d_level)

temp1 = temp1.dropna()  # 只有五行有缺失，直接删除

# temp1.to_excel('f1.xlsx',index=None)

temp2 = df2.copy()
del temp2['用户'], temp2['场景备注数据'], temp2['现象备注数据'], temp2['APP大类备注'], temp2['APP小类视频备注'], temp2['APP小类游戏备注']
del temp2['APP小类上网备注'], temp2['终端类型'], temp2['操作系统'], temp2['终端制式'], temp2['终端品牌'], temp2['终端品牌类型']
del temp2['当月高频通信分公司'], temp2['码号资源-激活时间'], temp2['码号资源-发卡时间'], temp2['客户星级标识'], temp2['畅享套餐名称']

temp2 = temp2.fillna(0)
d4_55 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 98: 1, 99: 1}  # 第4-11列对应字典
d_yesno = {0: 0, '是': 1, '否': 0}
d_sex = {'女': 0, '性别不详': 1, '男': 1}
for i in range(4, 56):
    temp2.iloc[:, i] = temp2.iloc[:, i].map(d4_55)

for i in [62, 63, 66, 67, 97, 99, 100, 101]:
    temp2.iloc[:, i] = temp2.iloc[:, i].map(d_yesno)
temp2['性别'] = temp2['性别'].map(d_sex)

# temp2.to_excel('f2.xlsx',index=None)
# [markdown]
# 相关矩阵

x1 = temp1.corr().iloc[:, [0]].apply(abs).sort_values('语音通话整体满意度', ascending=0)
x2 = temp2.corr().iloc[:, [0]].apply(abs).sort_values('手机上网整体满意度', ascending=0)
# x1.to_excel('x1_top.xlsx')
# x2.to_excel('x2_top.xlsx')
# [markdown]
# 根据1相关矩阵筛选变量

# 根据各个影响因素对于整体满意度的相关系数，我们将删去下面这些变量，并将筛选后的文件保存到f1_.xlsx和f2_.xlsx
print(temp1.shape, temp2.shape)
dd1 = ['当月MOU', '语音通话-时长（分钟）', '前3月MOU', '省际漫游-时长（分钟）', '外省流量占比',
       '外省语音占比', 'GPRS-国内漫游-流量（KB）', 'GPRS总流量（KB）', '套外流量（MB）', '是否实名登记用户',
       '是否关怀用户', '客户星级标识', '是否去过营业厅', '语音方式', '是否5G网络客户', '资费投诉', '套外流量费（元）', '4\\5G用户', '家宽投诉']
for i in dd1:
    del temp1[i]
dd2 = ['年龄', '2G驻留时长', '重定向次数', '王者荣耀使用天数', '搜狐视频', '王者荣耀质差次数', '是否全月漫游用户',
       '梦幻西游', '王者荣耀APP使用流量', '游戏类APP使用天数', '当月GPRS资源使用量（GB）', '大众点评使用流量',
       '王者荣耀使用次数', '是否校园套餐用户', '校园卡校园合约捆绑用户', '抖音使用流量（MB）', '主套餐档位',
       '高单价超套客户（集团）', '视频类应用流量', '本年累计消费（元）', '是否不限量套餐到达用户', '饿了么使用流量',
       '近3个月平均消费（元）', '近3个月平均消费（剔除通信账户支付）', '套外流量费（元）', '网易系APP流量',
       '音乐类应用流量', '微信质差次数', '天猫使用流量', '阿里系APP流量', '小视频系APP流量', '上网质差次数',
       '通信类应用流量', '游戏类APP使用流量', '游戏类应用流量', '校园卡无校园合约用户', '畅享套餐档位',
       '快手使用流量', '优酷视频使用流量', '蜻蜓FMAPP使用流量', '高频高额超套客户（集团）',
       '游戏类APP使用次数', '今日头条使用流量', '美团外卖使用流量', '滴滴出行使用流量', '网页类应用流量',
       '腾讯系APP流量', '邮箱类应用流量', '当月MOU', '套外流量（MB）', '腾讯视频使用流量', '是否5G网络客户', '性别']
for i in dd2:
    del temp2[i]
print(temp1.shape, temp2.shape)

# 3
temp3 = df3.copy()
del temp3['用户id'], temp3['用户描述'], temp3['用户描述.1'], temp3['终端品牌'], temp3['终端品牌类型']

d4 = {1: 1, 2: 0}  # 是否遇到过网络问题 0 1
d5_12 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 98: 1}
d13_19 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 98: 1}
d245g = {'2G': 0, '4G': 1, '5G': 2}
d_yesno = {0: 0, '是': 1, '否': 0}
d_level = {'未评级': 1, '准星': 2, '一星': 3, '二星': 4, '三星': 5, '白金卡': 6, '钻石卡': 7, '银卡': 8, '金卡': 9}

temp3.iloc[:, 0] = temp3.iloc[:, 0].map(d4)

for i in range(1, 9):
    temp3.iloc[:, i] = temp3.iloc[:, i].map(d5_12)

for i in range(9, 16):
    temp3.iloc[:, i] = temp3.iloc[:, i].map(d13_19)

temp3['4\\5G用户'] = temp3['4\\5G用户'].map(d245g)
temp3['外省流量占比'] = temp3['外省流量占比'].fillna(0)
temp3['外省语音占比'] = temp3['外省语音占比'].fillna(0)

for i in ['是否关怀用户', '是否4G网络客户（本地剔除物联网）', '是否5G网络客户']:
    temp3[i] = temp3[i].fillna(0).map(d_yesno)

temp3['客户星级标识'] = temp3['客户星级标识'].fillna(0).map(d_level)

# 4
temp4 = df4.copy()
del temp4['用户id']
del temp4['终端品牌'], temp4['终端品牌类型']
del temp4['客户星级标识']

temp4 = temp4.fillna(0)
d4_55 = {-1: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 98: 1, 99: 1}  # 第4-11列对应字典
d_yesno = {0: 0, '是': 1, '否': 0}
d_sex = {'女': 0, '性别不详': 1, '男': 1}
for i in range(1, 61):
    temp4.iloc[:, i] = temp4.iloc[:, i].map(d4_55)

for i in [0, 64, 67, 69, 81, 82]:
    temp4.iloc[:, i] = temp4.iloc[:, i].map(d_yesno)
temp4['性别'] = temp4['性别'].map(d_sex)


# x1.to_excel('x1_top.xlsx')
# x2.to_excel('x2_top.xlsx')
# [markdown]
# 根据1相关矩阵筛选变量

# 根据各个影响因素对于整体满意度的相关系数，我们将删去下面这些变量，并将筛选后的文件保存到f1_.xlsx和f2_.xlsx
print(temp3.shape, temp4.shape)
dd1 = ['当月MOU', '语音通话-时长（分钟）', '前3月MOU', '省际漫游-时长（分钟）', '外省流量占比',
       '外省语音占比', 'GPRS-国内漫游-流量（KB）', 'GPRS总流量（KB）', '套外流量（MB）',
       '是否关怀用户', '客户星级标识', '是否5G网络客户', '套外流量费（元）', '4\\5G用户']
for i in dd1:
    del temp3[i]
dd2 = ['搜狐视频', '梦幻西游', '是否不限量套餐到达用户', '套外流量费（元）', '微信质差次数', '上网质差次数',
       '当月MOU', '套外流量（MB）',  '是否5G网络客户', '性别']
for i in dd2:
    del temp4[i]
print(temp3.shape, temp4.shape)



#  保留测试集中有的属性

for i in temp3:
    if i not in temp1.columns:
        del temp3[i]
for i in temp4:
    if i not in temp2.columns:
        del temp4[i]
temp3.to_csv('pre1.csv',index=None)
temp4.to_csv('pre2.csv',index=None)

for i in temp1.columns:
    if i not in temp3.columns:
        del temp1[i]
for i in temp2.columns:
    if i not in temp4.columns:
        del temp2[i]
temp1.to_csv('train1_.csv',index=None)
temp2.to_csv('train2_.csv',index=None)
print(temp1.shape, temp3.shape)
print(temp2.shape, temp4.shape)