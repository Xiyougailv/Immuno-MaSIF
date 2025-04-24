import os
import numpy as np
import sys
from Bio.PDB import * ## 未使用
from biopandas.pdb import PandasPdb
from scipy import spatial
import pandas as pd
import csv

## 通过sys导入自定义的模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source") ## For Linux
from default_config.masif_opts import masif_opts

####################################
##  本程序用于： 数据后处理
##      MaSIF_pMHC解释性：Trace patches on PEPTIDE（对pep各号位进行重要程度打分）
##      本脚本：产生scorepep.csv/weigh_scorepep.csv文件（包含 PDB ID && 各号位分数）：
##            分别对应于不添加/添加预测置信度(confidence)作为权重因子，从而影响surfvert频次（重要程度）统计
###   SC，2023/11/08

## 11.11 B27迁移（微调脚本参数）
####################################

##--------------------------##
##  迭代部分（传入）&& 超参设置
##--------------------------##
pdb_info = sys.argv[1] 
fields = pdb_info.split("_")
pdbid = fields[0] ## PDB ID
pmhc_chain = fields[1] ## AC
label = fields[3] ## TCR Label
## 拆分pMHC
pmhc = list(pmhc_chain)
pep = pmhc[1] 

params = masif_opts["pmhc_fingerprint"]

## 质子化后的PDB文件：用于统计pep各号位氨基酸的原子坐标
# pdb_file = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/00-raw_pdbs/1AO7.pdb" 
pdb_file = masif_opts['raw_pdb_dir']+"/"+pdbid+".pdb"
pep_chainid = pep # "C"

## 预处理数据文件夹：用于加载三角剖分后pMHC上所有surface-vert的坐标，用于对应于pep各号位氨基酸的sub-vert-pool的搜寻
# precom_dir = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/04sc-precomputation_12A/precomputation/1AO7_AC_DE_A6"
precom_dir = params["masif_precomputation_dir"]+"/"+pdb_info+"/"
atom_cutoff = 4 ## 超参，24.Dec

## 表面打分溯源：用于原始频次数据的获取
## csv_dir = params["predict_trace-all-test"] ## 包含：我们为了溯源重新test的.npy文件 & 记录每个结构sample/logits的.csv大表

# 11.09 换用4A-TCR-cutoff数据
csv_dir = params["predict_trace_40-7_pepcut4_b27-rigid"] ## 24.Dec
sample_path = os.path.join(csv_dir, f"b'{pdb_info}'_sample.csv") ## 文件名的形式 ？
logits_path = os.path.join(csv_dir, f"b'{pdb_info}'_logits.csv")
sum_csv_file = os.path.join(csv_dir, "C6_scorepep_sum.csv") ##0308 ## 11.11 注意修改：L2 --> Label 2
weigh_sum_csv_file = os.path.join(csv_dir, "C6_weigh_scorepep_sum.csv")

extc_ratio = 0.05 ## 抽取行号百分比

## 处理标签
labels_dict = {"A6": 0, "1E6": 1, "DMF5": 2, "JM22": 3,"a24b17": 4, "T4H2": 5, "868": 6,"newb27": 7} ##0308 ## 与.csv文件的索引格式呼应：从0开始
corr_label = 5 ## 溯源指定标签 C5--a24b17 && C6--T4H2
print("正在处理 %s ... 其TCR标签为 %d ... :)\n" % (pdb_info, corr_label))
print("超参设置：surf-vert的pep-atom-cutoff为 %s A ...抽取行号比例为 %s ...\n" % (atom_cutoff, extc_ratio))


print("------------------------- <<<不添加权重>>> ---------------------------")
##-----------------##
##  存储pep原子坐标
##-----------------##
'''
    得到： 不规则列表 pep_coords --> pep各号位氨基酸中所有原子的坐标列表
             列表 atomnum_list --> pep各号位氨基酸包含的原子数目列表
'''

## BioPandas：通过Pandas中的DataFrames来方便的处理PDB文件
data = PandasPdb().read_pdb(pdb_file)
# data.df.keys() ## ['ATOM', 'HETATM', 'ANISOU', 'OTHERS']
df = data.df['ATOM']
## 找出符合条件的raws
t = df[(df.chain_id == pep_chainid)]
# print('t.shape：{}'.format(t.shape))

## 获取pep残基编号 （注意：可以不从1开始；9/10mer-pep）
resid_list = list(set(t.residue_number)) ## 将residue_number降重排序

atomnum_list = []
pep_coords = []
for resid in resid_list:    
    df_id = df[(df.chain_id == pep_chainid) & (df.residue_number == resid)] ## 选出每个pep氨基酸对应的子df
    atomnum_list.append(len(list(df_id.x_coord))) ## 得到每个氨基酸包含的原子数目
    pep_id_coords = list(zip(list(df_id.x_coord), list(df_id.y_coord), list(df_id.z_coord)))
    pep_coords.append(pep_id_coords)

#print(atomnum_list) ## 各个氨基酸包含的原子数目列表
#print(pep_coords) ## 各个氨基酸中所有原子的坐标列表（共9个子列表，每个子列表包含若干原子的坐标（格式：三维元组）

print("** 1. 存储Peptide各号位氨基酸中所有原子的坐标：Pep共 %d 位；各号位氨基酸包含的原子数目：%s \n" % (len(atomnum_list), atomnum_list)) ## %d代表整数

##-----------------------------##
##  搜寻各号位pep的sub-vert-pool
##-----------------------------##
'''
    得到： 字典 vertidx_dict --> key：氨基酸位置序号；value：SUB-pool-vert索引列表
          列表 vertnum_list --> 2A-atom-cutoff范围内，每号位pep附近的pMHC-surf-vert数目
'''


## 加载pMHC所有surf-vert的坐标
X = np.load(os.path.join(precom_dir, "p1_X.npy"))
Y = np.load(os.path.join(precom_dir, "p1_Y.npy"))
Z = np.load(os.path.join(precom_dir, "p1_Z.npy"))

xyz_coords = np.vstack([X, Y, Z]).T
tree = spatial.KDTree(xyz_coords)

pep_coords = np.array(pep_coords, dtype=object) ## 形状不规则
# print(pep_coords.shape[0]) ## 9mer-pep

vertnum_list = []
surfidx_list = []
for i in range(pep_coords.shape[0]):
    pep_icoords = pep_coords[i]
    surf_iverts = tree.query_ball_point(pep_icoords, atom_cutoff) ##### atom-cutoff: 2A--> pMHC-surf-vert VS 某号位pep残基中的任意atom
    surf_ivertidx = list(set([pp for p in surf_iverts for pp in p]))  ## 返回符合条件的所有patch center vert索引：去重排序
    resid = i+1
    vertnum_list.append(len(surf_ivertidx))
    surfidx_list.append(surf_ivertidx)
    
#print(vertnum_list) ## 2A-atom-cutoff范围内，每个氨基酸附近的pMHC-surf-vert数目

## 写入字典
keys = [(i+1) for i in range(pep_coords.shape[0])] ## pep位置序号（索引从0开始）
values = surfidx_list
vertidx_dict = dict(zip(keys, values))  ## key：氨基酸位置序号；value：sub-pool-vert索引列表
# print(vertidx_dict) 

print("** 2. 搜寻各号位Pep的SUB-vert-pool：Atom-cutoff设置为 %sA；各号位Pep附近的pMHC-surf-vert数目：%s \n" % (atom_cutoff, vertnum_list)) ## %d代表整数


##----------------------##
##  获取原始频次统计数据
##----------------------##
'''
  摘取自：表面打分脚本
  得到： 字典 sort_vimpo_dict --> key：SUM-pool-vert索引号；value：标签关联程度值（原始统计频次）  
'''

print("** 3. 获取原始频次统计数据：")
##++++++++++++++++##
##  处理logits.csv
##++++++++++++++++##
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

print("  *** 3a. 共采样 %d 次，正确预测 %d 次" % (df.shape[0], df_sort.shape[0])) ## %d代表整数

## 预测正确行数：
corr_row = df_sort.shape[0]
## 抽取前30%，则抽取行数：
extc_row = int(corr_row * extc_ratio)
## 抽取行号
df_extc = df_sort.iloc[0:extc_row]
## 得到抽取的行号列表
extc_rowlist = []
extc_rowlist.append(df_extc.index.tolist())

percent_ratio = extc_ratio * 100
print("  *** 3b. 将logits.csv按照置信度排序，并抽取前 %d-percent ：" % percent_ratio)
#print(df_extc)

##++++++++++++++++##
##  处理sample.csv
##++++++++++++++++##
# 打开文件
with open(sample_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  
rowlist = np.squeeze(extc_rowlist)
##  抽取出预测正确 & 把握高 的sample情况
df_samp = df.loc[rowlist]

print("          --> 抽取的sample.csv形状为： %d 行 * %d 列" % (df_samp.shape[0], df_samp.shape[1]))

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

vert_impomin = min(sort_vimpo_dict.values())
vert_impomax = max(sort_vimpo_dict.values())

print("  *** 3c. 提取poolvert频次字典：vert数目为 %d，原始频次区间为 %d ~ %d\n" \
      % (len(sort_vimpo_dict), vert_impomin, vert_impomax ))


##-------------------------------##
##  打分：各号位Pep的平均标签关联度
##-------------------------------##
'''
  得到： 字典 pep_avg_dict --> key：Pep氨基酸位置序号；value：平均标签关联程度打分值（已标准化）  
'''

## vertidx_dict：key：氨基酸位置序号；value：sub-pool-vert索引列表
## sort_vimpo_dict：key：pool-vert索引号；value：标签关联程度值（原始统计频次）

print("** 4. 打分：各号位Pep的标签关联度：")
sumidx = list(sort_vimpo_dict) ## sum-pool-vert索引列表
pep_vidx_dict = {} ## 创建空字典
pep_vimpo_dict = {} 

ovlap_list = []
for pepidx, pool in vertidx_dict.items():
    pep_vimpo_dict[pepidx] = []
    pep_vidx_dict[pepidx] = []
    for vertidx in pool:
        if vertidx in sumidx:
            pep_vidx_dict[pepidx].append(vertidx) ## 若SUBvert位于SUMpool中，记录其索引值
            pep_vimpo_dict[pepidx].append(sort_vimpo_dict[vertidx]) ## 记录其对应原始统计频次
    ovlap_list.append(len(pep_vidx_dict[pepidx])) ## sub&sum pool 重叠vert数
            
print("  *** 4a. 各号位Pep的SUBpool与SUMpool的重叠vert数目：%s" % ovlap_list) ## 

## 规范化处理：求平均 + 频次数值标准化
colnum = df_samp.shape[0] ## 总抽取行数
vertnum = len(sumidx)
coeff = colnum * 32 / vertnum ### 标准化系数：总抽取行数 * 32 / SUM-pool-vert数目

pep_avg_dict = {}
print_avg_dict = {}
for pepidx, impool in pep_vimpo_dict.items():
    if len(impool) == 0:
        pep_avg_dict[pepidx] = 0 ## 此号位pep周围没有位于SUMpool中的vert
        print_avg_dict[pepidx] = 0
    else:
        arr_impool = np.array(impool)
        avg_val = np.mean(arr_impool) / coeff
        avg_val_6 = round(avg_val, 6) ## 保留六位小数
        pep_avg_dict[pepidx] = avg_val_6
        avg_val_3 = round(avg_val, 3)
        print_avg_dict[pepidx] = avg_val_3 ## 打印方便看：保留两位小数

print("  *** 4b. 各号位Pep的平均标签关联度分值（以1为基准）：%s" % print_avg_dict) ## 

##-------------------------##
## （追加）写入scorepep.csv
##-------------------------##

row = []
row.append(pdbid)
for idnum, score in pep_avg_dict.items():
    row.append(score)
#print(row)

with open(sum_csv_file, "a", newline="")as csvfile:
    #header = ["PDBID", "1", "2", "3", "4", "5", "6", "7", "8", "9"] ##表头：字段列表
    writer = csv.writer(csvfile)
    writer.writerow(row)
        
print("\n** 5. 追加写入scorepep.csv文件，%s 处理完成:) " % (pdbid)) ## 追加写入时，注意不要重复运行程序！！！
print("------------------------- <<< 完成 >>> ---------------------------\n")


print("------------------------- <<<添加权重>>> ---------------------------")
##-----------------##
##  存储pep原子坐标
##-----------------##
'''
    得到： 不规则列表 pep_coords --> pep各号位氨基酸中所有原子的坐标列表
             列表 atomnum_list --> pep各号位氨基酸包含的原子数目列表
'''

## BioPandas：通过Pandas中的DataFrames来方便的处理PDB文件
data = PandasPdb().read_pdb(pdb_file)
# data.df.keys() ## ['ATOM', 'HETATM', 'ANISOU', 'OTHERS']
df = data.df['ATOM']
## 找出符合条件的raws
t = df[(df.chain_id == pep_chainid)]
# print('t.shape：{}'.format(t.shape))

## 获取pep残基编号 （注意：可以不从1开始；9/10mer-pep）
resid_list = list(set(t.residue_number)) ## 将residue_number降重排序

atomnum_list = []
pep_coords = []
for resid in resid_list:    
    df_id = df[(df.chain_id == pep_chainid) & (df.residue_number == resid)] ## 选出每个pep氨基酸对应的子df
    atomnum_list.append(len(list(df_id.x_coord))) ## 得到每个氨基酸包含的原子数目
    pep_id_coords = list(zip(list(df_id.x_coord), list(df_id.y_coord), list(df_id.z_coord)))
    pep_coords.append(pep_id_coords)

#print(atomnum_list) ## 各个氨基酸包含的原子数目列表
#print(pep_coords) ## 各个氨基酸中所有原子的坐标列表（共9个子列表，每个子列表包含若干原子的坐标（格式：三维元组）

print("** 1. 存储Peptide各号位氨基酸中所有原子的坐标：Pep共 %d 位；各号位氨基酸包含的原子数目：%s \n" % (len(atomnum_list), atomnum_list)) ## %d代表整数

##-----------------------------##
##  搜寻各号位pep的sub-vert-pool
##-----------------------------##
'''
    得到： 字典 vertidx_dict --> key：氨基酸位置序号；value：SUB-pool-vert索引列表
          列表 vertnum_list --> 2A-atom-cutoff范围内，每号位pep附近的pMHC-surf-vert数目
'''


## 加载pMHC所有surf-vert的坐标
X = np.load(os.path.join(precom_dir, "p1_X.npy"))
Y = np.load(os.path.join(precom_dir, "p1_Y.npy"))
Z = np.load(os.path.join(precom_dir, "p1_Z.npy"))

xyz_coords = np.vstack([X, Y, Z]).T
tree = spatial.KDTree(xyz_coords)

pep_coords = np.array(pep_coords, dtype=object) ## 形状不规则
# print(pep_coords.shape[0]) ## 9mer-pep

vertnum_list = []
surfidx_list = []
for i in range(pep_coords.shape[0]):
    pep_icoords = pep_coords[i]
    surf_iverts = tree.query_ball_point(pep_icoords, atom_cutoff) ##### atom-cutoff: 2A--> pMHC-surf-vert VS 某号位pep残基中的任意atom
    surf_ivertidx = list(set([pp for p in surf_iverts for pp in p]))  ## 返回符合条件的所有patch center vert索引：去重排序
    resid = i+1
    vertnum_list.append(len(surf_ivertidx))
    surfidx_list.append(surf_ivertidx)
    
#print(vertnum_list) ## 2A-atom-cutoff范围内，每个氨基酸附近的pMHC-surf-vert数目

## 写入字典
keys = [(i+1) for i in range(pep_coords.shape[0])] ## pep位置序号（索引从0开始）
values = surfidx_list
vertidx_dict = dict(zip(keys, values))  ## key：氨基酸位置序号；value：sub-pool-vert索引列表
# print(vertidx_dict) 

print("** 2. 搜寻各号位Pep的SUB-vert-pool：Atom-cutoff设置为 %sA；各号位Pep附近的pMHC-surf-vert数目：%s \n" % (atom_cutoff, vertnum_list)) ## %d代表整数


##----------------------##
##  获取原始频次统计数据
##----------------------##
'''
  摘取自：表面打分脚本
  得到： 字典 sort_vimpo_dict --> key：SUM-pool-vert索引号；value：标签关联程度值（原始统计频次）  
'''

print("** 3. 获取原始频次统计数据：")
##++++++++++++++++##
##  处理logits.csv
##++++++++++++++++##
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

print("  *** 3a. 共采样 %d 次，正确预测 %d 次" % (df.shape[0], df_sort.shape[0])) ## %d代表整数

## 预测正确行数：
corr_row = df_sort.shape[0]
## 抽取前30%，则抽取行数：
extc_row = int(corr_row * extc_ratio)
## 抽取行号
df_extc = df_sort.iloc[0:extc_row]
## 得到抽取的行号列表
extc_rowlist = []
extc_rowlist.append(df_extc.index.tolist())

percent_ratio = extc_ratio * 100
print("  *** 3b. 将logits.csv按照置信度排序，并抽取前 %d-percent ：" % percent_ratio)
#print(df_extc)

###--------------------------------------------------------- <<< 添加权重

## 添加权重-->“超参数” 
### 置信度 --> 取对数 --> softmax归一
weigh_conf = df_extc['confidence'].values ##置信度转为数组
log_base = min(weigh_conf) ## 以最小置信度为对数底数（注：此值一定 > 1）
log_upval = max(weigh_conf)
print("          --> 抽取的置信度取值范围：%f ~ %f" % (log_base, log_upval))

## 取对数
import math
weigh_impo = []
for i in weigh_conf:
    weigh_impo.append(math.log(i, log_base))    
up_val = max(weigh_impo)
down_val = min(weigh_impo)
print("          --> 将置信度以下界为底取对数，映射到：%f ~ %f" % (down_val, up_val))

## softmax归一（希望权重差距在一个数量级之内）
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
weigh_impo = softmax(weigh_impo) ## softmax归一
weigh_ratio = max(weigh_impo) / min(weigh_impo)
print("          --> Softmax函数处理，得到归一化权重：权重差距为：%f" % (weigh_ratio))

###--------------------------------------------------------- >>>


##++++++++++++++++##
##  处理sample.csv
##++++++++++++++++##
# 打开文件
with open(sample_path, 'r') as lf:
    # 读取文件
    df = pd.read_csv(lf, header=None)  
rowlist = np.squeeze(extc_rowlist)
##  抽取出预测正确 & 把握高 的sample情况
df_samp = df.loc[rowlist]

print("          --> 抽取的sample.csv形状为： %d 行 * %d 列" % (df_samp.shape[0], df_samp.shape[1]))

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

vert_impomin = min(sort_vimpo_dict.values())
vert_impomax = max(sort_vimpo_dict.values())

print("  *** 3c. 提取poolvert频次字典：vert数目为 %d，原始频次区间为 %f ~ %f\n" \
      % (len(sort_vimpo_dict), vert_impomin, vert_impomax ))


##-------------------------------##
##  打分：各号位Pep的平均标签关联度
##-------------------------------##
'''
  得到： 字典 pep_avg_dict --> key：Pep氨基酸位置序号；value：平均标签关联程度打分值（已标准化）  
'''

## vertidx_dict：key：氨基酸位置序号；value：sub-pool-vert索引列表
## sort_vimpo_dict：key：pool-vert索引号；value：标签关联程度值（原始统计频次）

print("** 4. 打分：各号位Pep的标签关联度：")
sumidx = list(sort_vimpo_dict) ## sum-pool-vert索引列表
pep_vidx_dict = {} ## 创建空字典
pep_vimpo_dict = {} 

ovlap_list = []
for pepidx, pool in vertidx_dict.items():
    pep_vimpo_dict[pepidx] = []
    pep_vidx_dict[pepidx] = []
    for vertidx in pool:
        if vertidx in sumidx:
            pep_vidx_dict[pepidx].append(vertidx) ## 若SUBvert位于SUMpool中，记录其索引值
            pep_vimpo_dict[pepidx].append(sort_vimpo_dict[vertidx]) ## 记录其对应原始统计频次
    ovlap_list.append(len(pep_vidx_dict[pepidx])) ## sub&sum pool 重叠vert数
            
print("  *** 4a. 各号位Pep的SUBpool与SUMpool的重叠vert数目：%s" % ovlap_list) ## 

## 规范化处理：求平均 + 频次数值标准化
colnum = df_samp.shape[0] ## 总抽取行数
vertnum = len(sumidx)
coeff = colnum * 32 / vertnum ### 标准化系数：总抽取行数 * 32 / SUM-pool-vert数目

pep_avg_dict = {}
print_avg_dict = {}
for pepidx, impool in pep_vimpo_dict.items():
    if len(impool) == 0:
        pep_avg_dict[pepidx] = 0 ## 此号位pep周围没有位于SUMpool中的vert
        print_avg_dict[pepidx] = 0
    else:
        arr_impool = np.array(impool)
        avg_val = np.mean(arr_impool) / coeff
        avg_val_6 = round(avg_val, 6) ## 保留六位小数
        pep_avg_dict[pepidx] = avg_val_6
        avg_val_3 = round(avg_val, 3)
        print_avg_dict[pepidx] = avg_val_3 ## 打印方便看：保留两位小数

print("  *** 4b. 各号位Pep的平均标签关联度分值（以1为基准）：%s" % print_avg_dict) ## 


##-------------------------##
## （追加）写入scorepep.csv
##-------------------------##

row = []
row.append(pdbid)
for idnum, score in pep_avg_dict.items():
    row.append(score)
#print(row)

with open(weigh_sum_csv_file, "a", newline="")as csvfile:
    #header = ["PDBID", "1", "2", "3", "4", "5", "6", "7", "8", "9"] ##表头：字段列表
    writer = csv.writer(csvfile)
    writer.writerow(row)
        
print("\n** 5. 追加写入weigh_scorepep.csv文件，%s 处理完成:) " % (pdbid)) ## 追加写入时，注意不要重复运行程序！！！
print("------------------------- <<< 完成 >>> ---------------------------\n")







