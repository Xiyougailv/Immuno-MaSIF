## SC, 23/11/06

-- 本文件夹 source/data_preparation_pmhc 用于：

## 1. 特征工程：准备Feature-riched patches （总流程控制脚本：pmhc_batch_prepare.sh；批处理名单：xxx.txt）
### 1a). 00c-(md/clustmd)-save_TCR_coords.py:  质子化，存储TCR坐标 // IN >> ./00-raw_pdbs ; OUT >> ./00c-tcr_coords
### 1b). 01-pdb_extract_and_triangulate.py: 质子化，抽取pMHC表面，三角剖分，加入三种物化特征 // IN >> ./00-raw_pdbs ; OUT >> ./01-benchmark_surfaces && ./01-benchmark_pdbs
### 1c). 04sc-masif_precompute.py: 划出patch(计算测地极坐标)，加入几何特征（patch可视化） // IN >> ./01-benchmark_surfaces ; OUT >> ./04sc-precomputation_12A

## 2. 标签解释性溯源预处理：Score surface-patches （流程控制脚本位于source/trace_pmhc/下）
### 2a). 05-tracesurf.py: 产生可载入pymol的.ply文件， // IN >> ; OUT >>

## 3. 深度网络数据预处理：Make tensorflow records (流程控制脚本位于source/workflow_xxx.sh)
### 3a). makedata_xxx-xxx.py: 写入训练/测试集对应的train/test.tfrecords文件（<注意>：在此脚本中，依据TCR-cutoff选出patch-center-pool） // IN >> ./00c-tcr_coords && ./04sc-precomputation_12A ; OUT >> ./tfrecords（<注意>：由于80上空间的问题，将此文件夹移到根目录/data下）


## 深度网络：预处理&&训练模型&&测试模型 (总流程控制脚本：source/workflow_xxx.sh)
### a). data_preparation_pmhc/makedata_xxx-xxx.py: 产生训练/测试集对应的train/test.tfrecords文件
### b). masif_pmhc/trainmodel_xxx-xxx.py: 训练模型，保存模型 // IN >> /data/.../tfrecords ; OUT >> ./data/.../nn_models/all_feat/
### c). masif_pmhc/testmodel_xxx-xxx.py: 测试模型，保存预测 // IN >> ./data/.../nn_models/all_feat/ ; OUT >> ./data/.../nn_models/test_set_predictions/


### d). masif_pmhc/testmodel_trace-xxx-xxx.py: 保存所有采样与预测结果