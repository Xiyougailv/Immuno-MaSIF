import os
import numpy as np
import sys
from Bio.PDB import *
from biopandas.pdb import PandasPdb
from scipy import spatial
import pandas as pd
import csv

#### ---------------------------------------
## 20241129 对pMHC圈出“Peptide框”
##   本脚本：质子化后，储存Pep（所有原子的 ##可按需改动）坐标：
##         后面需要依据Pep-cutoff筛选可训练的pMHC-patch
#### ---------------------------------------

## from SBI.structure import PDB
## 通过sys导入自定义的default_config/input_output模块（识别标志：__init__.py文件）
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")
from default_config.masif_opts import masif_opts


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
pdb_file = masif_opts['pdb_chain_dir']+"/"+pdbid+"_"+pmhc_chain+".pdb" ## 241203, 确保加载的pdb经过质子化；eg.1AO7_AC.pdb
pep_chainid = pep # "C"

if not os.path.exists(masif_opts["pmhc_fingerprint"]["pep_coords_dir"]):
    os.mkdir(masif_opts["pmhc_fingerprint"]["pep_coords_dir"])

## 预处理数据文件夹：用于加载三角剖分后pMHC上所有surface-vert的坐标，用于对应于pep各号位氨基酸的sub-vert-pool的搜寻
# precom_dir = "/Users/shangchun/Desktop/MyMaSIF_tolinux/source/data_preparation_pmhc/04sc-precomputation_12A/precomputation/1AO7_AC_DE_A6"
precom_dir = params["masif_precomputation_dir"]+"/"+pdb_info+"/"
atom_cutoff = 5 ## 超参 
## 24.Dec 注意，此处的超参设置仅为打印检查之用，本步骤的pep坐标存储与超参取值无关！
## 25.Jan 在B27例中，尝试将cutoff4->5


print("正在处理 %s ... :)\n" % (pdb_info))
print("超参设置：surf-vert的pep-atom-cutoff为 %s A ...\n" % (atom_cutoff))

##---------------------------##
##  存储所有pep原子坐标，写入文件
##---------------------------##

structure_pep_coords = []
parser = PDBParser()
struct = parser.get_structure(pdbid, pdb_file)
### 选出Pep上的所有原子，并存储其坐标
for atom in struct.get_atoms():
    residue = atom.get_parent()
    chain = residue.get_parent()
    if chain.get_id() in pep_chainid:
        coords = "{:.06f} {:.06f} {:.06f}".format(
            atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
            ) ## 需要这么高的精度吗？？
        structure_pep_coords.append(coords)

np.save(
    os.path.join(
        masif_opts["pmhc_fingerprint"]["pep_coords_dir"], "{}_pep_coords.npy".format(pdbid)
    ),
    structure_pep_coords,
)

print("------------------------- <<< COPY >>> ---------------------------")
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

## 241203 补充
from collections import Counter
# 展平所有子列表
flattened_values = [point for sublist in values for point in sublist]
point_counts = Counter(flattened_values)  # Counter 会生成 {点编号: 出现次数} 的字典
distribution = Counter(point_counts.values())  # Counter 会统计 {重复次数: 对应点编号数量}
#print("每个点编号的重复次数:", dict(point_counts))  # 转为普通字典便于查看
#print("重复次数的整体分布:", dict(distribution))

print("** 2. 搜寻各号位Pep的SUB-vert-pool：Atom-cutoff设置为 %sA；各号位Pep附近的pMHC-surf-vert数目：%s \n" % (atom_cutoff, vertnum_list)) ## %d代表整数
print("** A.1 使用Pep-cutoff方式所选出的Pep-surf-pool包含的vert数目：\n", len(dict(point_counts)) ) 
print("** A.2 统计各号位Pep附近的vert分布情况：不同pep之间选出的vert重叠次数整体分布：\n", dict(distribution)) 
