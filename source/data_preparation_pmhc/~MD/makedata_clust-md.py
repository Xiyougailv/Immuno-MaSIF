import numpy as np
import tensorflow as tf #### 2.13.0, SC
from random import shuffle
import os
import glob
from scipy import spatial #### 1.10.1
## 通过sys导入自定义的default_config/input_output模块（识别标志：__init__.py文件）
import sys
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")
from default_config.masif_opts import masif_opts

#####################################
##  此脚本功能：Make tensorflow records
#####################################
import random ## 随机抽取face

gamenum = sys.argv[1]

params = masif_opts["pmhc_fingerprint"]
## ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"] 
tcr_labels = ["A6", "JM22", "868", "1E6", "a24b17"]
# List all structures that have been preprocessed

## 我的储存位置： precomputation/MD/10_n/pdb_info/xxx.npy (人为选择了10-7)
precomputed_pdbs = glob.glob(
    os.path.join(params["masif_precomputation_dir_clustmd"], "10_7", "*", "p1_X.npy")
) ## glob.glob: 返回所有匹配的文件路径列表
precomputed_pdbs = [p.split("/")[-2] for p in precomputed_pdbs] ### 表示所有处理过的PDB(文件夹名): 1AO7_AC_DE_A6

# Only use the ones selected based on sequence homology
'''
selected_pdbs = np.load(os.path.join("lists", "selected_pdb_ids_30.npy"))
selected_pdbs = selected_pdbs.astype(str)
all_pdbs = [p for p in precomputed_pdbs if p.split("_")[0] in selected_pdbs] ### 处理过且同源筛选过的PDB（包含ID与chain信息：1AO7_AC）
''' ##SC
## 再次筛选
selected_pdbs = []
f = open("./data_preparation_pmhc/md_list_20_5.txt", "r")
line = f.readline() # 读取第一行
while line:
    pdb_str = line.strip('\n')
    selected_pdbs.append(pdb_str) # 列表增加
    line = f.readline() # 读取下一行
print(selected_pdbs)
f.close()
all_pdbs = [p for p in precomputed_pdbs if p in selected_pdbs] ### 处理过且在名单上的PDB（包含ID与chain信息：1AO7_AC_DE_A6）

##labels_dict = {"ADP": 1, "COA": 2, "FAD": 3, "HEM": 4, "NAD": 5, "NAP": 6, "SAM": 7}
labels_dict = {"A6": 1, "JM22": 2, "868": 3, "1E6": 4, "a24b17": 5}

# Structures are randomly assigned to train, validation and test sets
'''
shuffle(all_pdbs)
train = int(len(all_pdbs) * params["train_fract"])
val = int(len(all_pdbs) * params["val_fract"])
test = int(len(all_pdbs) * params["test_fract"])
print("Train", train)
print("Validation", val)
print("Test", test)
train_pdbs = all_pdbs[:train]
val_pdbs = all_pdbs[train : train + val]
test_pdbs = all_pdbs[train + val : train + val + test]
# np.save('lists/train_pdbs_sequence.npy',train_pdbs)
# np.save('lists/val_pdbs_sequence.npy',val_pdbs)
# np.save('lists/test_pdbs_sequence.npy',test_pdbs)
'''

## 挑选train/test set （应该可以简写 -_-
train_pdbs = []
test_pdbs = []
class1 = []
class2 = []
class3 = []
#class4 = []
class5 = []
class6 = []
shuffle(all_pdbs)
for p in all_pdbs:
    in_fields = p.split('_')
    ## print(in_fields)
    tcr_type = in_fields[3]
    if tcr_type == "A6":
        class1.append(p)
    if tcr_type == "JM22":
        class2.append(p)
    if tcr_type == "868":
        class3.append(p)
    #if tcr_type == "T4H2":
    #    class4.append(p)
    if tcr_type == "1E6":
        class5.append(p)
    if tcr_type == "a24b17":
        class6.append(p)
train_pdbs.extend(class1[:6]) ## 前6个
train_pdbs.extend(class2[:2]) ## 前2个
train_pdbs.extend(class3[:2])
#train_pdbs.extend(class4[:2])
train_pdbs.extend(class5[:2])
train_pdbs.extend(class6[:2])
test_pdbs.extend(class1[6:]) ## 后2个
test_pdbs.extend(class2[2:]) ## 后1个
test_pdbs.extend(class3[2:])
#test_pdbs.extend(class4[2:])
test_pdbs.extend(class5[2:])
test_pdbs.extend(class6[2:])
print(train_pdbs)
print(test_pdbs)

datalist_dir = params["datalist_clust-md"]
if not os.path.exists(datalist_dir):
    os.makedirs(datalist_dir)
np.save(os.path.join(datalist_dir, f"{gamenum}_train_pdbs_pmhc.npy"), train_pdbs)
np.save(os.path.join(datalist_dir, f"{gamenum}_test_pdbs_pmhc.npy"), test_pdbs)

### ---------------------
##   给出train/val/test set的PDB信息列表（包含ID与chain信息：1AO7_AC）
### ---------------------
# For this run use the train, validation and test sets actually used
train_pdbs = np.load(os.path.join(datalist_dir, f"{gamenum}_train_pdbs_pmhc.npy")).astype(str)
test_pdbs = np.load(os.path.join(datalist_dir, f"{gamenum}_test_pdbs_pmhc.npy")).astype(str)
## val_pdbs = np.load("lists/val_pdbs_sequence.npy").astype(str)


success = 0
## 这里需要改动：对应/Clust_MD/子文件夹 (已经创建过)
precom_dir = params["masif_precomputation_dir_clustmd"]
tcr_coord_dir = params["tcr_coords_dir_clustmd"]

tfrecords_dir = params["tfrecords_clust-md"]
if not os.path.exists(tfrecords_dir):
    os.mkdir(tfrecords_dir)

### ---------------------
##   将train/val/test set打包生成TFRecord文件：
##   后续可以配合TF中相关的API实现数据的加载、处理、训练等一系列工作。
#        TFRecord是Google专为Tensorflow设计的一种数据格式；
#        本质上是二进制文件（二进制存储占有空间少）（由一行行字节字符串byte-string构成的样本数据)：
#        -- 一条TFRecord数据代表一个Example(即一个样本数据)；
#        -- 每个Example内部由一个字典构成，字典的每一个key（特征字段名）对应一种Feature（特征数据）；
#        -- Feature有三种数据类型：ByteList/FloatList/Int64List       
### ---------------------

##### ------------------------------
####### 逐个pdb写入训练集数据文件：training_data_sequenceSplit_30.tfrecord
### 重点操作：对相关pocket vertice打上了ligand种类数字标签
##### ------------------------------


## 指定要写入的TFRecord文件的地址：
## Tensorflow 1.x旧版命令: tf.python_io.TFRecordWriter
with tf.compat.v1.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, f"{gamenum}_training_data_14.tfrecord")
) as writer:
    for i, pdb in enumerate(train_pdbs):
        print("Working on Training pdb: ", pdb)
        ## ！！对一个PDB文件操作
        ## ！！随机选择 10 张脸中的 1 张
        sum_face = 10
        face_num = random.randint(1, sum_face)  # 取1到10之间的随机整数（1 <= num <= 10）
        print("Choosing face num: ", face_num)
        #### 下面需要把所有输入文件对应定位到"THIS choosed-face"
        try:
            # Load precomputed data
            input_feat = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_input_feat.npy") ## 删去了"_" ???
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_Z.npy"))
            all_tcr_coords = np.load(
                os.path.join(
                    tcr_coord_dir, "{}_tcr_coords_{}_{}.npy".format(pdb.split("_")[0], sum_face, face_num)
                )
            )
            all_tcr_types = np.load(
                os.path.join(
                    tcr_coord_dir, "{}_tcr_type_{}_{}.npy".format(pdb.split("_")[0], sum_face, face_num)
                )
            ).astype(str) ### ligand可能有多个~   ## tcr只有一个标签
        except:
            continue

        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pmhc_labels = np.zeros(
            (xyz_coords.shape[0], len(all_tcr_types)), dtype=int
        )  ## (n_vertices, 1) 

        print("Sum pMHC surface verts: "+str(xyz_coords.shape[0])) ## 检查错误 ~5900 vertices

        ## 处理格式是string的tcr原子坐标列表：转为浮点型
        tcr_coords = []
        for i, coord_str in enumerate(all_tcr_coords):
            coord_float = coord_str.strip().split()
            tcr_coords.append(list(map(float,coord_float)))


        #############################
        ## 打标签 --> verts on pMHC surface within 3A from corresponding TCR
        ##   给此结构中符合条件的vertice（patch center vert）打上对应的TCR种类数字标签
        ##   最终的pmhc_labels形状: (n_vertices, 1)
        #############################
        for j, structure_tcr in enumerate(all_tcr_types):
            ## 注意：此列表只有一个元素（tcr type唯一）
            pmhc_points = tree.query_ball_point(tcr_coords, 3.0)
            ## 此时，要搜索其邻居（距离3.0以内）的tcr_coords为多个点：
            ##   返回的pmhc_points为：包含(每点)邻居索引列表的形状元组的对象数组
            pmhc_points_flatten = list(set([pp for p in pmhc_points for pp in p]))  ## 返回符合条件的所有patch center vert索引
            print("Labeled pmhc surface verts: "+str(len(pmhc_points_flatten)))  ## 检查错误 ~350 vertices
            ## list(set())：可以将原列表去重，并按从小到大排序
            pmhc_labels[pmhc_points_flatten, j] = labels_dict[structure_tcr]


        '''
        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )  ## (n_vertices, ligand_type_num_within_a_pdb) (！！注意，同一个结构中可能有多个ligand)
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        #############################
        ## 给此结构中符合条件的vertice（patch center vert）打上对应的ligand种类数字标签
        ##   最终的pocket_labels形状: (n_vertices, ligand_type_num_within_a_pdb)
        #############################
        for j, structure_ligand in enumerate(all_ligand_types):
            ## 处理此结构中的一种ligand
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0) 
            ## 此时，要搜索其邻居（距离3.0以内）的ligand_coords为多个点：
            ## 返回的pocket_points为：包含(每点)邻居索引列表的形状元组的对象数组
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))  ## 返回符合条件的所有patch center vert索引
            ## list(set())：可以将原列表去重，并按从小到大排序
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]
        ''' ## SC

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pmhc_labels.shape) ## 注意：这里没改名字了：保持统一
        pocket_labels = tf.train.Int64List(value=pmhc_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        } ## 对于此种PDB结构的每一vertice（即数据行数为：n_vertices）

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())  ## 写入TFRecord文件
        ## Example --->(serialize) ---> byte-string
        if i % 1 == 0:
            print("Training data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(train_pdbs))



success = 0
with tf.compat.v1.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, f"{gamenum}_testing_data_6.tfrecord")
) as writer:
    for i, pdb in enumerate(test_pdbs):
        print("Working on Testing pdb: ", pdb)
        ## ！！对一个PDB文件操作
        ## ！！随机选择 10 张脸中的 1 张
        sum_face = 10
        face_num = random.randint(1, sum_face)  # 取1到10之间的随机整数（1 <= num <= 10）
        print("Choosing face num: ", face_num)
        #### 下面需要把所有输入文件对应定位到"THIS choosed-face"
        try:
            # Load precomputed data
            input_feat = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_input_feat.npy") ## 删去了"_" ???
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, f"{sum_face}_{face_num}", pdb, "p1_Z.npy"))
            all_tcr_coords = np.load(
                os.path.join(
                    tcr_coord_dir, "{}_tcr_coords_{}_{}.npy".format(pdb.split("_")[0], sum_face, face_num)
                )
            )
            all_tcr_types = np.load(
                os.path.join(
                    tcr_coord_dir, "{}_tcr_type_{}_{}.npy".format(pdb.split("_")[0], sum_face, face_num)
                )
            ).astype(str) ### ligand可能有多个~   ## tcr只有一个标签
        except:
            continue

        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pmhc_labels = np.zeros(
            (xyz_coords.shape[0], len(all_tcr_types)), dtype=int
        )  ## (n_vertices, 1) 

        print("Sum pMHC surface verts: "+str(xyz_coords.shape[0]))

        ## 处理格式是string的tcr原子坐标列表：转为浮点型
        tcr_coords = []
        for i, coord_str in enumerate(all_tcr_coords):
            coord_float = coord_str.strip().split()
            tcr_coords.append(list(map(float,coord_float)))


        #############################
        ## 打标签 --> verts on pMHC surface within 3A from corresponding TCR
        ##   给此结构中符合条件的vertice（patch center vert）打上对应的TCR种类数字标签
        ##   最终的pmhc_labels形状: (n_vertices, 1)
        #############################
        for j, structure_tcr in enumerate(all_tcr_types):
            ## 注意：此列表只有一个元素（tcr type唯一）
            pmhc_points = tree.query_ball_point(tcr_coords, 3.0)
            ## 此时，要搜索其邻居（距离3.0以内）的tcr_coords为多个点：
            ##   返回的pmhc_points为：包含(每点)邻居索引列表的形状元组的对象数组
            pmhc_points_flatten = list(set([pp for p in pmhc_points for pp in p]))  ## 返回符合条件的所有patch center vert索引
            print("Labeled pmhc surface verts: "+str(len(pmhc_points_flatten)))
            ## list(set())：可以将原列表去重，并按从小到大排序
            pmhc_labels[pmhc_points_flatten, j] = labels_dict[structure_tcr]


        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pmhc_labels.shape)
        pocket_labels = tf.train.Int64List(value=pmhc_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        } ## 对于此种PDB结构的每一vertice（即数据行数为：n_vertices）

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())  ## 写入TFRecord文件
        ## Example --->(serialize) ---> byte-string
        if i % 1 == 0:
            print("Testing data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(test_pdbs))

'''
success = 0
with tf.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, "validation_data_sequenceSplit_30.tfrecord")
) as writer:
    for i, pdb in enumerate(val_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_", "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Validation data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(val_pdbs))


success = 0
with tf.python_io.TFRecordWriter(
    os.path.join(tfrecords_dir, "testing_data_sequenceSplit_30.tfrecord")
) as writer:
    for i, pdb in enumerate(test_pdbs):
        try:
            input_feat = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_input_feat.npy")
            )
            rho_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_rho_wrt_center.npy")
            )
            theta_wrt_center = np.load(
                os.path.join(precom_dir, pdb + "_", "p1_theta_wrt_center.npy")
            )
            mask = np.expand_dims(np.load(os.path.join(precom_dir, pdb + "_", "p1_mask.npy")),-1)
            X = np.load(os.path.join(precom_dir, pdb + "_", "p1_X.npy"))
            Y = np.load(os.path.join(precom_dir, pdb + "_", "p1_Y.npy"))
            Z = np.load(os.path.join(precom_dir, pdb + "_", "p1_Z.npy"))
            all_ligand_coords = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_coords.npy".format(pdb.split("_")[0])
                )
            )
            all_ligand_types = np.load(
                os.path.join(
                    ligand_coord_dir, "{}_ligand_types.npy".format(pdb.split("_")[0])
                )
            ).astype(str)
        except:
            continue

        if len(all_ligand_types) == 0:
            continue
        xyz_coords = np.vstack([X, Y, Z]).T
        tree = spatial.KDTree(xyz_coords)
        pocket_labels = np.zeros(
            (xyz_coords.shape[0], len(all_ligand_types)), dtype=np.int
        )
        # Label points on surface within 3A distance from ligand with corresponding ligand type
        for j, structure_ligand in enumerate(all_ligand_types):
            ligand_coords = all_ligand_coords[j]
            pocket_points = tree.query_ball_point(ligand_coords, 3.0)
            pocket_points_flatten = list(set([pp for p in pocket_points for pp in p]))
            pocket_labels[pocket_points_flatten, j] = labels_dict[structure_ligand]

        input_feat_shape = tf.train.Int64List(value=input_feat.shape)
        input_feat_list = tf.train.FloatList(value=input_feat.reshape(-1))
        rho_wrt_center_shape = tf.train.Int64List(value=rho_wrt_center.shape)
        rho_wrt_center_list = tf.train.FloatList(value=rho_wrt_center.reshape(-1))
        theta_wrt_center_shape = tf.train.Int64List(value=theta_wrt_center.shape)
        theta_wrt_center_list = tf.train.FloatList(value=theta_wrt_center.reshape(-1))
        mask_shape = tf.train.Int64List(value=mask.shape)
        mask_list = tf.train.FloatList(value=mask.reshape(-1))
        pdb_list = tf.train.BytesList(value=[pdb.encode()])
        pocket_labels_shape = tf.train.Int64List(value=pocket_labels.shape)
        pocket_labels = tf.train.Int64List(value=pocket_labels.reshape(-1))

        features_dict = {
            "input_feat_shape": tf.train.Feature(int64_list=input_feat_shape),
            "input_feat": tf.train.Feature(float_list=input_feat_list),
            "rho_wrt_center_shape": tf.train.Feature(int64_list=rho_wrt_center_shape),
            "rho_wrt_center": tf.train.Feature(float_list=rho_wrt_center_list),
            "theta_wrt_center_shape": tf.train.Feature(
                int64_list=theta_wrt_center_shape
            ),
            "theta_wrt_center": tf.train.Feature(float_list=theta_wrt_center_list),
            "mask_shape": tf.train.Feature(int64_list=mask_shape),
            "mask": tf.train.Feature(float_list=mask_list),
            "pdb": tf.train.Feature(bytes_list=pdb_list),
            "pocket_labels_shape": tf.train.Feature(int64_list=pocket_labels_shape),
            "pocket_labels": tf.train.Feature(int64_list=pocket_labels),
        }

        features = tf.train.Features(feature=features_dict)
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())
        if i % 1 == 0:
            print("Testing data")
            success += 1
            print(success)
            print(pdb)
            print(float(i) / len(test_pdbs))
''' ## SC，我们没有设置val set :)

