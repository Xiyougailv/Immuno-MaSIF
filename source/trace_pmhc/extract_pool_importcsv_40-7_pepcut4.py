import os
import sys
import numpy as np
import pandas as pd
import csv
## 通过sys导入自定义的模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source") ## For Linux
from default_config.masif_opts import masif_opts

####################################
##  本程序用于： 数据后处理
##      MaSIF_pMHC解释性：Trace patches on surface（在pMHC表面上对verts的重要程度可视化）
##      本脚本：产生pool.csv/weigh_pool.csv文件（包含 poolvert index/importance）：
##            分别对应于不添加/添加预测置信度(confidence)作为权重因子，从而影响surfvert频次（重要程度）统计
###   SC，2023/10/26
####################################

##----------------##
##  迭代部分（传入）
##----------------##
pdb_info = sys.argv[1] 
fields = pdb_info.split("_")
pdb = fields[0] ## PDB ID
label = fields[3] ## TCR Label

params = masif_opts["pmhc_fingerprint"]

## 11.11 TCR-cutoff改为4A
## 24-03-05 对旧31-6分类器溯源
csv_dir = params["predict_trace_40-7_pepcut4_all-test"] ## 0305修改
## 包含：我们为了溯源重新test的.npy文件 & 记录每个结构sample/logits的.csv大表
sample_path = os.path.join(csv_dir, f"b'{pdb_info}'_sample.csv") ## 文件名的形式 ？
logits_path = os.path.join(csv_dir, f"b'{pdb_info}'_logits.csv")
csv_out_file = os.path.join(csv_dir, f'pool_{pdb}.csv')
weigh_csv_out_file = os.path.join(csv_dir, f'weigh_pool_{pdb}.csv')

## 处理标签
labels_dict = {"A6": 0, "1E6": 1, "DMF5": 2, "JM22": 3,"a24b17": 4, "T4H2": 5, "868": 6} ### 40-7 24.Dec.17，注意最后两个标签顺序
corr_label = labels_dict[label]
print("正在处理 %s ... 其TCR标签为 %d ... :)\n" % (pdb_info, corr_label))


print("------------------------- <<<不添加权重>>> ---------------------------")
##----------------##
##  处理logits.csv
##----------------##
# 打开文件
with open(logits_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  

## 求每一行最大值及对应索引 (注意执行顺序！！！)
df['sec_val'] = np.sort(df.to_numpy())[:, -2] ## 第二大值
df['max_val'] = df.max(axis=1) ## 最大值
df['max_idx'] = df.idxmax(axis=1) ## 最大值索引
df['confidence'] = df['max_val'] / df['sec_val'] ##置信度
## 获取max_idx为label的行
df_add = df[df['max_idx'] == corr_label]
## 将置信度降序排列
df_sort = df_add.sort_values(by='confidence',ascending=False ) 

print("** 1. 共采样 %d 次，正确预测 %d 次\n" % (df.shape[0], df_sort.shape[0])) ## %d代表整数

## 预测正确行数：
corr_row = df_sort.shape[0]
## 抽取前30%，则抽取行数：
extc_row = int(corr_row * 0.1) ## 24.Dec 对pep溯源一致
## 抽取行号
df_extc = df_sort.iloc[0:extc_row]
## 得到抽取的行号列表
extc_rowlist = []
extc_rowlist.append(df_extc.index.tolist())

print("** 2. 将logits.csv按照置信度排序，并抽取前30%：")
print(df_extc)

##----------------##
##  处理sample.csv
##----------------##
# 打开文件
with open(sample_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  
rowlist = np.squeeze(extc_rowlist)
##  抽取出预测正确 & 把握高 的sample情况
df_samp = df.loc[rowlist]

print("\n** 3. 抽取的sample.csv形状为： %d 行 * %d 列" % (df_samp.shape[0], df_samp.shape[1]))

## 统计vert频次：转换成array处理
arr_samp = np.array(df_samp)
arr_samp =  arr_samp.reshape(-1)
#print(arr_samp.shape)

## 用array进行vert频次排序：字典格式降序排列
vimpo_dict = {}
for i in arr_samp:
    if i in vimpo_dict:
        vimpo_dict[i] +=1
    else:
        vimpo_dict[i] =1

sort_vimpo_dict = dict(sorted(vimpo_dict.items(), key=lambda x: x[1], reverse=True)) ######### vert频次字典


## “标准化处理”：线性缩放
##    最大频次 --> 1.0; 最小频次 --> 0.0 
vert_impomin = min(sort_vimpo_dict.values())
vert_impomax = max(sort_vimpo_dict.values())
vert_imporange = vert_impomax - vert_impomin
for key in sort_vimpo_dict:
    sort_vimpo_dict[key] -= vert_impomin
    sort_vimpo_dict[key] /= vert_imporange
    sort_vimpo_dict[key] = np.round(sort_vimpo_dict[key], 6)## 6位浮点数
    
#print(sort_vimpo_dict) ############ 最终结果：vert频次字典 --> normalized:)
print("\n** 4. 提取poolvert频次字典：vert数目为 %d，原始频次区间为 %d ~ %d；已标准化到[0, 1]区间" \
      % (len(sort_vimpo_dict), vert_impomin, vert_impomax ))


##----------------##
##  写入pool.csv
##----------------##
vert_idx = [k for k in sort_vimpo_dict]
vert_impo = [v for v in sort_vimpo_dict.values()]
rows = zip(vert_idx, vert_impo)
with open(csv_out_file, "w", newline="")as csvfile:
    header = ["vert_idx", "vert_impo"] ##表头：字段列表
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
        
print("\n** 5. vertpool中共有 %d 个vert，已写入pool.csv文件，处理完成:)\n" % len(sort_vimpo_dict))
print("------------------------- <<< 完成 >>> ---------------------------\n")


print("------------------------- <<<添加权重>>> ---------------------------")
##----------------##
##  处理logits.csv
##----------------##
# 打开文件
with open(logits_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  

## 求每一行最大值及对应索引 (注意执行顺序！！！)
df['sec_val'] = np.sort(df.to_numpy())[:, -2] ## 第二大值
df['max_val'] = df.max(axis=1) ## 最大值
df['max_idx'] = df.idxmax(axis=1) ## 最大值索引
df['confidence'] = df['max_val'] / df['sec_val'] ##置信度
## 获取max_idx为label的行
df_add = df[df['max_idx'] == corr_label]
## 将置信度降序排列
df_sort = df_add.sort_values(by='confidence',ascending=False ) 

print("** 1. 共采样 %d 次，正确预测 %d 次\n" % (df.shape[0], df_sort.shape[0])) ## %d代表整数

## 预测正确行数：
corr_row = df_sort.shape[0]
## 抽取前30%，则抽取行数：
extc_row = int(corr_row * 0.1)
## 抽取行号
df_extc = df_sort.iloc[0:extc_row]
## 得到抽取的行号列表
extc_rowlist = []
extc_rowlist.append(df_extc.index.tolist())

print("** 2. 将logits.csv按照置信度排序，并抽取前10%：")
print(df_extc)

###--------------------------------------------------------- <<< 添加权重

## 添加权重-->“超参数” 
### 置信度 --> 取对数 --> softmax归一
weigh_conf = df_extc['confidence'].values ##置信度转为数组
log_base = min(weigh_conf) ## 以最小置信度为对数底数（注：此值一定 > 1）
log_upval = max(weigh_conf)
print("\n*** 2a. 抽取的置信度取值范围：%f ~ %f" % (log_base, log_upval))

## 取对数
import math
weigh_impo = []
for i in weigh_conf:
    weigh_impo.append(math.log(i, log_base))    
up_val = max(weigh_impo)
down_val = min(weigh_impo)
print("*** 2b. 将置信度以下界为底取对数，映射到：%f ~ %f" % (down_val, up_val))

## softmax归一（希望权重差距在一个数量级之内）
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
weigh_impo = softmax(weigh_impo) ## softmax归一
weigh_ratio = max(weigh_impo) / min(weigh_impo)
print("*** 2c. Softmax函数处理，得到归一化权重：权重差距为：%f" % (weigh_ratio))

###--------------------------------------------------------- >>>

##----------------##
##  处理sample.csv
##----------------##
# 打开文件
with open(sample_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  
rowlist = np.squeeze(extc_rowlist)
##  抽取出预测正确 & 把握高 的sample情况
df_samp = df.loc[rowlist]

print("\n** 3. 抽取的sample.csv形状为： %d 行 * %d 列\n" % (df_samp.shape[0], df_samp.shape[1]))

## 统计vert频次：转换成array处理
arr_samp = np.array(df_samp)
arr_samp =  arr_samp.reshape(-1)
#print(arr_samp.shape)

## 用array进行vert频次排序：字典格式降序排列
###--------------------------------------------------------- <<< 添加权重
## 对照weigh_impo，添加权重
vimpo_dict = {}
## 为了直观比较加权频次，我们将归一化的weigh_impo乘以抽取的sample次数
weigh_impo = np.multiply(weigh_impo, df_samp.shape[0])

for i,index in enumerate(arr_samp):
    weigh_idx = int(i / 32) ## 向下取整：得到应取得的权重在weigh_impo数组中的索引值
    if index in vimpo_dict:
        vimpo_dict[index] += weigh_impo[weigh_idx]
    else:
        vimpo_dict[index] = weigh_impo[weigh_idx]
###--------------------------------------------------------- >>>

sort_vimpo_dict = dict(sorted(vimpo_dict.items(), key=lambda x: x[1], reverse=True)) ######### vert频次字典


## “标准化处理”：线性缩放
##    最大频次 --> 1.0; 最小频次 --> 0.0 
vert_impomin = min(sort_vimpo_dict.values())
vert_impomax = max(sort_vimpo_dict.values())
vert_imporange = vert_impomax - vert_impomin
for key in sort_vimpo_dict:
    sort_vimpo_dict[key] -= vert_impomin
    sort_vimpo_dict[key] /= vert_imporange
    sort_vimpo_dict[key] = np.round(sort_vimpo_dict[key], 6)## 6位浮点数
    
#print(sort_vimpo_dict) ############ 最终结果：vert频次字典 --> normalized:)
print("\n** 4. 提取poolvert频次字典：vert数目为 %d，加权重的“原始”频次区间为 %f ~ %f；已标准化：线性缩放" \
      % (len(sort_vimpo_dict), vert_impomin, vert_impomax ))


##----------------##
##  写入pool.csv
##----------------##
vert_idx = [k for k in sort_vimpo_dict]
vert_impo = [v for v in sort_vimpo_dict.values()]
rows = zip(vert_idx, vert_impo)
with open(weigh_csv_out_file, "w", newline="")as csvfile:
    header = ["vert_idx", "vert_impo"] ##表头：字段列表
    writer = csv.writer(csvfile)
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
        
print("\n** 5. vertpool中共有 %d 个vert，已写入pool.csv文件，处理完成:)\n" % len(sort_vimpo_dict))

print("------------------------- <<< 完成 >>> ---------------------------\n")







