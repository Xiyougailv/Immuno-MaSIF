import tempfile

masif_opts = {}
# Default directories
masif_opts["raw_pdb_dir"] = "data_preparation_pmhc/00-raw_pdbs/"
masif_opts["pdb_chain_dir"] = "data_preparation_pmhc/01-benchmark_pdbs/"
masif_opts["ply_chain_dir"] = "data_preparation_pmhc/01-benchmark_surfaces/"
masif_opts["tmp_dir"] = tempfile.gettempdir()
masif_opts["ply_file_template"] = masif_opts["ply_chain_dir"] + "/{}_{}.ply"
## 添加187个pHLA原始pdb文件的路径
masif_opts["init_pdb_dir"] = "data_preparation_pmhc/init_pdb_23"

'''240921 dMaSIF preparation for 40 structures
'''
masif_opts["dmasif_compute_iface"] = True
masif_opts["dmasif_pdb_chain_dir"] = "/home/alcohol/dMaSIF-design/surface_data/raw/01-benchmark_pdbs/"
masif_opts["dmasif_ply_chain_dir"] = "/home/alcohol/dMaSIF-design/surface_data/raw/01-benchmark_surfaces/"


## 240712 Docking 5850 strucs
## 240714：改用绝对路径，否则出现函数调用时，文件相对该功能函数的位置可能发生变化
masif_opts["dock-raw_pdb_dir"] = "/home/alcohol/data/Pandora_5850/pdb_pandora_5850/"
masif_opts["dock-pdb_chain_dir"] = "/home/alcohol/data/Pandora_5850/data_preparation/01-benchmark_pdbs/"
masif_opts["dock-ply_chain_dir"] = "/home/alcohol/data/Pandora_5850/data_preparation/01-benchmark_surfaces/"
masif_opts["dock-ply_file_template"] = masif_opts["dock-ply_chain_dir"] + "/{}_{}.ply"

# Surface features
masif_opts["use_hbond"] = True
masif_opts["use_hphob"] = True
masif_opts["use_apbs"] = True
'''
masif_opts["compute_iface"] = True
''' ## SC, For pMHC-case
masif_opts["compute_iface"] = False
# Mesh resolution. Everything gets very slow if it is lower than 1.0
masif_opts["mesh_res"] = 1.0
masif_opts["feature_interpolation"] = True


# Coords params
masif_opts["radius"] = 12.0

## 20250408 PIP-PocketFingerprint
masif_opts["pip_pocketfp"] = {} ## 先定义！
masif_opts["pip_pocketfp"]["pip_coords_dir"] = "data_preparation_pmhc/00e-pip_coords/"
masif_opts["pip_pocketfp"]["ply_pocket"] = "../data/masif_pmhc/nn_models/PIP/ply_pocket/"



########## Parameters for pMHC: SC
masif_opts["pmhc_fingerprint"] = {} ## 先定义！
masif_opts["pmhc_fingerprint"]["tcr_coords_dir"] = "data_preparation_pmhc/00c-tcr_coords/"
masif_opts["pmhc_fingerprint"]["pep_coords_dir"] = "data_preparation_pmhc/00c-pep_coords/"
masif_opts["pmhc_fingerprint"]["max_shape_size"] = 200
masif_opts["pmhc_fingerprint"]["max_distance"] = 12.0
masif_opts["pmhc_fingerprint"][
    "masif_precomputation_dir"
] = "data_preparation_pmhc/04sc-precomputation_12A/precomputation/"
masif_opts["pmhc_fingerprint"]["tfrecords_dir"] = "data_preparation_pmhc/tfrecords"
masif_opts["pmhc_fingerprint"]["model_dir"] = "../data/masif_pmhc/nn_models/all_feat/"  ##/source/下
masif_opts["pmhc_fingerprint"]["n_classes"] = 5
masif_opts["pmhc_fingerprint"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["pmhc_fingerprint"]["costfun"] = "dprime" ## ???
masif_opts["pmhc_fingerprint"]["test_set_out_dir"] = "../data/masif_pmhc/nn_models/test_set_predictions/"


###### Featurization for 5850 Pandora docking structures :)    Jul 2024
#masif_opts["pmhc_fingerprint"]["tcr_coords_dir_dock"] = "../../data/Pandora_5850/data_preparation/00c-tcr_coords/"
masif_opts["pmhc_fingerprint"][
    "masif_precomputation_dir_dock"
] = "/home/alcohol/data/Pandora_5850/data_preparation/04sc-precomputation_12A/precomputation/"
## Statistical feature distribution
masif_opts["pmhc_fingerprint"]["feature_distribution_dir_dock"] = "/home/alcohol/data/Pandora_5850/feature_distribution/"



###### Take MD-flexibility in to account! :)    Aug 2023
masif_opts["pmhc_fingerprint"]["tcr_coords_dir_md"] = "data_preparation_pmhc/00c-tcr_coords/MD/"
masif_opts["pmhc_fingerprint"][
    "masif_precomputation_dir_md"
] = "data_preparation_pmhc/04sc-precomputation_12A/precomputation/MD/"
##### MD-Faces
masif_opts["pmhc_fingerprint"]["datalist_md-faces"] = "../data/masif_pmhc/lists/md-faces/"
masif_opts["pmhc_fingerprint"]["tfrecords_md-faces"] = "data_preparation_pmhc/tfrecords/md-faces/"
masif_opts["pmhc_fingerprint"]["model_md-faces"] = "../data/masif_pmhc/nn_models/all_feat/md-faces/"
masif_opts["pmhc_fingerprint"]["predict_md-faces"] = "../data/masif_pmhc/nn_models/test_set_predictions/md-faces/"

###### Take MD-flexibility in to account! --> After **Clustering** :D   Sep 2023
masif_opts["pmhc_fingerprint"]["tcr_coords_dir_clustmd"] = "data_preparation_pmhc/00c-tcr_coords/Clust_MD/"
masif_opts["pmhc_fingerprint"][
    "masif_precomputation_dir_clustmd"
] = "data_preparation_pmhc/04sc-precomputation_12A/precomputation/Clust_MD/"
##### Clust-MD
masif_opts["pmhc_fingerprint"]["datalist_clust-md"] = "../data/masif_pmhc/lists/clust-md/"
masif_opts["pmhc_fingerprint"]["tfrecords_clust-md"] = "data_preparation_pmhc/tfrecords/clust-md/"
masif_opts["pmhc_fingerprint"]["model_clust-md"] = "../data/masif_pmhc/nn_models/all_feat/clust-md/"
masif_opts["pmhc_fingerprint"]["predict_clust-md"] = "../data/masif_pmhc/nn_models/test_set_predictions/clust-md/"


##### All-Test 
masif_opts["pmhc_fingerprint"]["datalist_all-test"] = "../data/masif_pmhc/lists/all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_all-test"] = "data_preparation_pmhc/tfrecords/all-test/"
masif_opts["pmhc_fingerprint"]["model_all-test"] = "../data/masif_pmhc/nn_models/all_feat/all-test/"
masif_opts["pmhc_fingerprint"]["predict_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/all-test/"
##### Trace-All-Test
masif_opts["pmhc_fingerprint"]["predict_trace-all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace-all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace-all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace-all-test/"
##### 4A_All-Test
masif_opts["pmhc_fingerprint"]["datalist_4A_all-test"] = "../data/masif_pmhc/lists/4A_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_4A_all-test"] = "../../data/MaSIF_pMHC/tfrecords/4A_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_4A_all-test"] = "../data/masif_pmhc/nn_models/all_feat/4A_all-test/"
masif_opts["pmhc_fingerprint"]["predict_4A_trace-all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/4A_trace-all-test/"
masif_opts["pmhc_fingerprint"]["ply_4A_trace-all-test"] = "../data/masif_pmhc/nn_models/trace_ply/4A_trace-all-test/"

##### 40-7_All-Test ## 23-12-07
masif_opts["pmhc_fingerprint"]["n_classes_40-7_all-test"] = 7
masif_opts["pmhc_fingerprint"]["datalist_40-7_all-test"] = "../data/masif_pmhc/lists/40-7_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_40-7_all-test"] = "../../data/MaSIF_pMHC/tfrecords/40-7_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_40-7_all-test"] = "../data/masif_pmhc/nn_models/all_feat/40-7_all-test/"
masif_opts["pmhc_fingerprint"]["predict_40-7_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/40-7_all-test/"

##### 40-7_PepCut5_All-Test ## 24-12-03 ## 注：为简化修改，本次并未修改名单位置（设置「gamenum」不重复：101-）
masif_opts["pmhc_fingerprint"]["tfrecords_40-7_pepcut5_all-test"] = "../../data/MaSIF_pMHC/tfrecords/40-7_pepcut5_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_40-7_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/all_feat/40-7_pepcut5_all-test/"
masif_opts["pmhc_fingerprint"]["predict_40-7_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/40-7_pepcut5_all-test/"
##### Trace_PepCut5_All-Test ## 24-12-04
masif_opts["pmhc_fingerprint"]["predict_trace_40-7_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_40-7_pepcut5_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_40-7_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_40-7_pepcut5_all-test/"
##### 40-7_PepCut6_All-Test ## 24-12-05 ## 注：为简化修改，本次并未修改名单位置（设置「gamenum」不重复：201-）
masif_opts["pmhc_fingerprint"]["tfrecords_40-7_pepcut6_all-test"] = "../../data/MaSIF_pMHC/tfrecords/40-7_pepcut6_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_40-7_pepcut6_all-test"] = "../data/masif_pmhc/nn_models/all_feat/40-7_pepcut6_all-test/"
masif_opts["pmhc_fingerprint"]["predict_40-7_pepcut6_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/40-7_pepcut6_all-test/"
##### Trace_PepCut6_All-Test ## 24-12-05
masif_opts["pmhc_fingerprint"]["predict_trace_40-7_pepcut6_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_40-7_pepcut6_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_40-7_pepcut6_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_40-7_pepcut6_all-test/"
##### 40-7_PepCut4_All-Test ## 24-12-06 ## 注：为简化修改，本次并未修改名单位置（设置「gamenum」不重复：301-） ## 24-12-15 问题结构重新特征工程后，训练50个All-Test模型，并覆盖先前的20个
masif_opts["pmhc_fingerprint"]["tfrecords_40-7_pepcut4_all-test"] = "../../data/MaSIF_pMHC/tfrecords/40-7_pepcut4_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_40-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/all_feat/40-7_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["predict_40-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/40-7_pepcut4_all-test/"
##### Trace_PepCut4_All-Test ## 24-12-06
masif_opts["pmhc_fingerprint"]["predict_trace_40-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_40-7_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_40-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_40-7_pepcut4_all-test/"

##### 40-7_PepCut4_B27-Rigid ## 24-12-17
masif_opts["pmhc_fingerprint"]["n_classes_40-7_b27-rigid"] = 8 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_40-7_b27-rigid"] = "../data/masif_pmhc/lists/40-7_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_40-7_pepcut4_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/40-7_pepcut4_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_40-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/40-7_pepcut4_b27-rigid/"
#masif_opts["pmhc_fingerprint"]["predict_40-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/40-7_pepcut4_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_trace_40-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_40-7_pepcut4_b27-rigid/"
masif_opts["pmhc_fingerprint"]["ply_trace_40-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/trace_ply/trace_40-7_pepcut4_b27-rigid/"



##### 31-6_All-Test (旧版)## 23-12-10
masif_opts["pmhc_fingerprint"]["n_classes_31-6_all-test"] = 6
masif_opts["pmhc_fingerprint"]["datalist_31-6_all-test"] = "../data/masif_pmhc/lists/31-6_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_31-6_all-test"] = "../../data/MaSIF_pMHC/tfrecords/31-6_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_31-6_all-test"] = "../data/masif_pmhc/nn_models/all_feat/31-6_all-test/"
masif_opts["pmhc_fingerprint"]["predict_31-6_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/31-6_all-test/"
##### Trace_31-6_All-Test ## 24-03-05
masif_opts["pmhc_fingerprint"]["predict_trace_31-6_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_31-6_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_31-6_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_31-6_all-test/"

##### 31-6_PepCut5_All-Test ## 24-12-07 ## 注：为简化修改，本次并未修改名单位置（设置「gamenum」不重复：101-）
masif_opts["pmhc_fingerprint"]["tfrecords_31-6_pepcut5_all-test"] = "../../data/MaSIF_pMHC/tfrecords/31-6_pepcut5_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_31-6_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/all_feat/31-6_pepcut5_all-test/"
masif_opts["pmhc_fingerprint"]["predict_31-6_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/31-6_pepcut5_all-test/"
##### Trace_PepCut5_All-Test ## 24-12-07
masif_opts["pmhc_fingerprint"]["predict_trace_31-6_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_31-6_pepcut5_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_31-6_pepcut5_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_31-6_pepcut5_all-test/"
##### 31-6_PepCut4_All-Test ## 24-12-07 ## 注：为简化修改，本次并未修改名单位置（设置「gamenum」不重复：301-）
masif_opts["pmhc_fingerprint"]["tfrecords_31-6_pepcut4_all-test"] = "../../data/MaSIF_pMHC/tfrecords/31-6_pepcut4_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_31-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/all_feat/31-6_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["predict_31-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/31-6_pepcut4_all-test/"
##### Trace_PepCut4_All-Test ## 24-12-07
masif_opts["pmhc_fingerprint"]["predict_trace_31-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_31-6_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_31-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_31-6_pepcut4_all-test/"


##### 32-6_PepCut4_All-Test ## 24-12-08 
masif_opts["pmhc_fingerprint"]["n_classes_32-6_all-test"] = 6
masif_opts["pmhc_fingerprint"]["datalist_32-6_all-test"] = "../data/masif_pmhc/lists/32-6_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_32-6_pepcut4_all-test"] = "../../data/MaSIF_pMHC/tfrecords/32-6_pepcut4_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_32-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/all_feat/32-6_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["predict_32-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/32-6_pepcut4_all-test/"
##### Trace_PepCut4_All-Test ## 24-12-09
masif_opts["pmhc_fingerprint"]["predict_trace_32-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_32-6_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["ply_trace_32-6_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/trace_ply/trace_32-6_pepcut4_all-test/"

##### 32-6_PepCut4_B27-Rigid ## 24-12-09
masif_opts["pmhc_fingerprint"]["n_classes_32-6_b27-rigid"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_32-6_b27-rigid"] = "../data/masif_pmhc/lists/32-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_32-6_pepcut4_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/32-6_pepcut4_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_32-6_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/32-6_pepcut4_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_32-6_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/32-6_pepcut4_b27-rigid/"
##### Trace_32-6_PepCut4_B27-Rigid ## 24-12-11
masif_opts["pmhc_fingerprint"]["predict_trace_32-6_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_32-6_pepcut4_b27-rigid/"
masif_opts["pmhc_fingerprint"]["ply_trace_32-6_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/trace_ply/trace_32-6_pepcut4_b27-rigid/"
##### Trace_32-6_PepCut4_A02-Rest ## 24-12-12
masif_opts["pmhc_fingerprint"]["predict_trace_32-6_pepcut4_a02-rest"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_32-6_pepcut4_a02-rest/"
masif_opts["pmhc_fingerprint"]["ply_trace_32-6_pepcut4_a02-rest"] = "../data/masif_pmhc/nn_models/trace_ply/trace_32-6_pepcut4_a02-rest/"


## >>>>>>>
##### 35-7_PepCut4_All-Test ## 24-12-14 （尽量不删掉label为原则）
masif_opts["pmhc_fingerprint"]["n_classes_35-7_all-test"] = 7
masif_opts["pmhc_fingerprint"]["datalist_35-7_all-test"] = "../data/masif_pmhc/lists/35-7_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_35-7_pepcut4_all-test"] = "../../data/MaSIF_pMHC/tfrecords/35-7_pepcut4_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_35-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/all_feat/35-7_pepcut4_all-test/"
masif_opts["pmhc_fingerprint"]["predict_35-7_pepcut4_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/35-7_pepcut4_all-test/"

##### 35-7_PepCut4_B27-Rigid ## 24-12-17
masif_opts["pmhc_fingerprint"]["n_classes_35-7_b27-rigid"] = 8 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_35-7_b27-rigid"] = "../data/masif_pmhc/lists/35-7_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_35-7_pepcut4_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/35-7_pepcut4_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_35-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/35-7_pepcut4_b27-rigid/"

masif_opts["pmhc_fingerprint"]["predict_trace_35-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_35-7_pepcut4_b27-rigid/"
masif_opts["pmhc_fingerprint"]["ply_trace_35-7_pepcut4_b27-rigid"] = "../data/masif_pmhc/nn_models/trace_ply/trace_35-7_pepcut4_b27-rigid/"


## <<<<<<<

##### 26-5_All-Test ## 23-12-19
masif_opts["pmhc_fingerprint"]["n_classes_26-5_all-test"] = 5
masif_opts["pmhc_fingerprint"]["datalist_26-5_all-test"] = "../data/masif_pmhc/lists/26-5_all-test/"
masif_opts["pmhc_fingerprint"]["tfrecords_26-5_all-test"] = "../../data/MaSIF_pMHC/tfrecords/26-5_all-test/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_26-5_all-test"] = "../data/masif_pmhc/nn_models/all_feat/26-5_all-test/"
masif_opts["pmhc_fingerprint"]["predict_26-5_all-test"] = "../data/masif_pmhc/nn_models/test_set_predictions/26-5_all-test/"




##### Zero-Shot 
masif_opts["pmhc_fingerprint"]["datalist_zero-shot"] = "../data/masif_pmhc/lists/zero-shot/"
masif_opts["pmhc_fingerprint"]["tfrecords_zero-shot"] = "data_preparation_pmhc/tfrecords/zero-shot/"
masif_opts["pmhc_fingerprint"]["model_zero-shot"] = "../data/masif_pmhc/nn_models/all_feat/zero-shot/"
masif_opts["pmhc_fingerprint"]["predict_zero-shot"] = "../data/masif_pmhc/nn_models/test_set_predictions/zero-shot/"
##### Leave-One-Out 
masif_opts["pmhc_fingerprint"]["datalist_leave-one-out"] = "../data/masif_pmhc/lists/leave-one-out/"
masif_opts["pmhc_fingerprint"]["tfrecords_leave-one-out"] = "data_preparation_pmhc/tfrecords/leave-one-out/"
masif_opts["pmhc_fingerprint"]["model_leave-one-out"] = "../data/masif_pmhc/nn_models/all_feat/leave-one-out/"
masif_opts["pmhc_fingerprint"]["predict_leave-one-out"] = "../data/masif_pmhc/nn_models/test_set_predictions/leave-one-out/"

##### B27-Rigid
masif_opts["pmhc_fingerprint"]["n_classes_b27-rigid"] = 6 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_b27-rigid"] = "../data/masif_pmhc/lists/b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_b27-rigid"] = "data_preparation_pmhc/tfrecords/b27-rigid/"
masif_opts["pmhc_fingerprint"]["model_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/b27-rigid/"
##### Trace-B27-Rigid
masif_opts["pmhc_fingerprint"]["predict_trace-b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace-b27-rigid/"
masif_opts["pmhc_fingerprint"]["ply_trace-b27-rigid"] = "../data/masif_pmhc/nn_models/trace_ply/trace-b27-rigid/"

##### 26-5_B27-Rigid ## 23-12-21
masif_opts["pmhc_fingerprint"]["n_classes_26-5_b27-rigid"] = 6 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_26-5_b27-rigid"] = "../data/masif_pmhc/lists/26-5_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_26-5_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/26-5_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_26-5_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/26-5_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_26-5_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/26-5_b27-rigid/"

##### 29-6_B27-Rigid ## 23-12-22
masif_opts["pmhc_fingerprint"]["n_classes_29-6_b27-rigid"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_29-6_b27-rigid"] = "../data/masif_pmhc/lists/29-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_29-6_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/29-6_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_29-6_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/29-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_29-6_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/29-6_b27-rigid/"

##### 29-6_YLQ-Infer ## 24-01-09 ## 注意：其实，此case下训出的模型与29-6_B27-Rigid case在理论上是一样的（训练集均为：29-6）
masif_opts["pmhc_fingerprint"]["n_classes_29-6_YLQ-infer"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_29-6_YLQ-infer"] = "../data/masif_pmhc/lists/29-6_YLQ-infer/"
masif_opts["pmhc_fingerprint"]["tfrecords_29-6_YLQ-infer"] = "../../data/MaSIF_pMHC/tfrecords/29-6_YLQ-infer/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_29-6_YLQ-infer"] = "../data/masif_pmhc/nn_models/all_feat/29-6_YLQ-infer/"
masif_opts["pmhc_fingerprint"]["predict_29-6_YLQ-infer"] = "../data/masif_pmhc/nn_models/test_set_predictions/29-6_YLQ-infer/"

##### 29-6_Dimer-Infer ## 24-01-12 ## 注：训练集为：29-6
masif_opts["pmhc_fingerprint"]["n_classes_29-6_dimer-infer"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_29-6_dimer-infer"] = "../data/masif_pmhc/lists/29-6_dimer-infer/"
masif_opts["pmhc_fingerprint"]["tfrecords_29-6_dimer-infer"] = "../../data/MaSIF_pMHC/tfrecords/29-6_dimer-infer/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_29-6_dimer-infer"] = "../data/masif_pmhc/nn_models/all_feat/29-6_dimer-infer/"
masif_opts["pmhc_fingerprint"]["predict_29-6_dimer-infer"] = "../data/masif_pmhc/nn_models/test_set_predictions/29-6_dimer-infer/"


##### 31-6_B27-Rigid ## 24-03-03 ## 重新挑选数据，并测试 (with 6TMO)
masif_opts["pmhc_fingerprint"]["n_classes_31-6_b27-rigid"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_31-6_b27-rigid"] = "../data/masif_pmhc/lists/31-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_31-6_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/31-6_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_31-6_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/31-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_31-6_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/31-6_b27-rigid/"
##### Trace_31-6_B27-Rigid ## 24-03-07
masif_opts["pmhc_fingerprint"]["predict_trace_31-6_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/trace_31-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["ply_trace_31-6_b27-rigid"] = "../data/masif_pmhc/nn_models/trace_ply/trace_31-6_b27-rigid/"



##### 30-6_B27-Rigid ## 24-03-03 ## 重新挑选数据，并测试 (without 6TMO)
masif_opts["pmhc_fingerprint"]["n_classes_30-6_b27-rigid"] = 7 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_30-6_b27-rigid"] = "../data/masif_pmhc/lists/30-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["tfrecords_30-6_b27-rigid"] = "../../data/MaSIF_pMHC/tfrecords/30-6_b27-rigid/" ## 最占空间的tfrecords文件移到80根目录/data下
masif_opts["pmhc_fingerprint"]["model_30-6_b27-rigid"] = "../data/masif_pmhc/nn_models/all_feat/30-6_b27-rigid/"
masif_opts["pmhc_fingerprint"]["predict_30-6_b27-rigid"] = "../data/masif_pmhc/nn_models/test_set_predictions/30-6_b27-rigid/"






##### B27-Flex
masif_opts["pmhc_fingerprint"]["n_classes_b27-flex"] = 6 ##  Add a "fake" class
masif_opts["pmhc_fingerprint"]["datalist_b27-flex"] = "../data/masif_pmhc/lists/b27-flex/"
masif_opts["pmhc_fingerprint"]["tfrecords_b27-flex"] = "data_preparation_pmhc/tfrecords/b27-flex/"
masif_opts["pmhc_fingerprint"]["model_b27-flex"] = "../data/masif_pmhc/nn_models/all_feat/b27-flex/"
masif_opts["pmhc_fingerprint"]["predict_b27-flex"] = "../data/masif_pmhc/nn_models/test_set_predictions/b27-flex/"


# Neural network patch application specific parameters.
masif_opts["ppi_search"] = {}
masif_opts["ppi_search"]["training_list"] = "lists/training.txt"
masif_opts["ppi_search"]["testing_list"] = "lists/testing.txt"
masif_opts["ppi_search"]["max_shape_size"] = 200
masif_opts["ppi_search"]["max_distance"] = 12.0  # Radius for the neural network.
masif_opts["ppi_search"][
    "masif_precomputation_dir"
] = "data_preparation/04b-precomputation_12A/precomputation/"
masif_opts["ppi_search"]["feat_mask"] = [1.0] * 5
masif_opts["ppi_search"]["max_sc_filt"] = 1.0
masif_opts["ppi_search"]["min_sc_filt"] = 0.5
masif_opts["ppi_search"]["pos_surf_accept_probability"] = 1.0
masif_opts["ppi_search"]["pos_interface_cutoff"] = 1.0
masif_opts["ppi_search"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["ppi_search"]["cache_dir"] = "nn_models/sc05/cache/"
masif_opts["ppi_search"]["model_dir"] = "nn_models/sc05/all_feat/model_data/"
masif_opts["ppi_search"]["desc_dir"] = "descriptors/sc05/all_feat/"
masif_opts["ppi_search"]["gif_descriptors_out"] = "gif_descriptors/"
# Parameters for shape complementarity calculations.
masif_opts["ppi_search"]["sc_radius"] = 12.0
masif_opts["ppi_search"]["sc_interaction_cutoff"] = 1.5
masif_opts["ppi_search"]["sc_w"] = 0.25

# Neural network patch application specific parameters.
masif_opts["site"] = {}
masif_opts["site"]["training_list"] = "lists/training.txt"
masif_opts["site"]["testing_list"] = "lists/testing.txt"
#masif_opts["site"]["pmhc_testing_list"] = "lists/testing.txt" ## SC
masif_opts["site"]["max_shape_size"] = 100
masif_opts["site"]["n_conv_layers"] = 3
masif_opts["site"]["max_distance"] = 9.0  # Radius for the neural network.
masif_opts["site"][
    "masif_precomputation_dir"
] = "data_preparation_site/04a-precomputation_9A/precomputation/" ##SC
masif_opts["site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
masif_opts["site"]["model_dir"] = "../data/masif_site/nn_models/all_feat_3l/model_data" ## SC, load trained masif-site
masif_opts["site"]["out_pred_dir"] = "../../data/MaSIF_pMHC/masif_site/output/all_feat_3l/pred_data/" ## SC
masif_opts["site"]["out_surf_dir"] = "../../data/MaSIF_pMHC/masif_site/output/all_feat_3l/pred_surfaces/" ## SC
#masif_opts["site"]["pmhc_out_pred_dir"] = "output/all_feat_3l/pred_data/" ## SC
#masif_opts["site"]["pmhc_out_surf_dir"] = "output/all_feat_3l/pred_surfaces/" ## SC
masif_opts["site"]["feat_mask"] = [1.0] * 5

# Neural network ligand application specific parameters.
masif_opts["ligand"] = {}
masif_opts["ligand"]["assembly_dir"] = "data_preparation/00b-pdbs_assembly"
masif_opts["ligand"]["ligand_coords_dir"] = "data_preparation/00c-ligand_coords"
masif_opts["ligand"][
    "masif_precomputation_dir"
] = "data_preparation/04a-precomputation_12A/precomputation/" 
masif_opts["ligand"]["max_shape_size"] = 200
masif_opts["ligand"]["feat_mask"] = [1.0] * 5
masif_opts["ligand"]["train_fract"] = 0.9 * 0.8
masif_opts["ligand"]["val_fract"] = 0.1 * 0.8
masif_opts["ligand"]["test_fract"] = 0.2
masif_opts["ligand"]["tfrecords_dir"] = "data_preparation/tfrecords"
masif_opts["ligand"]["max_distance"] = 12.0
masif_opts["ligand"]["n_classes"] = 7
masif_opts["ligand"]["feat_mask"] = [1.0, 1.0, 1.0, 1.0, 1.0]
masif_opts["ligand"]["costfun"] = "dprime"
masif_opts["ligand"]["model_dir"] = "nn_models/all_feat/"
masif_opts["ligand"]["test_set_out_dir"] = "test_set_predictions/"

