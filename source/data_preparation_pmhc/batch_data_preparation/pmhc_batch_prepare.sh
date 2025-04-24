#!/bin/bash

#file="./data_preparation_pmhc/b27_list_24-Dec.txt"  ## 24.Dec.11
file="./data_preparation_pmhc/tmp_list.txt" ## 24.Dec.15 

## 25.1.2 改变MSMS参数，重新处理B27两个WT结构；25.1.3 align结构集，改变MSMS及pepcut参数，重新处理B27结构集6个结构
## 24.12.15 重新处理含有非标准氨基酸的3D39和3D3V
## 24.12.11 处理GMX模拟500ns、轨迹聚类产生的B27结构集
## 24.12.3 使用pep-cutoff选取surf-pool
## 24.7.12 处理5850个Pandora docking结构：特征工程
## 24.3.7 处理7N2R09的新（正确）结构：7N2R2H.pdb

## 24.1.12 处理Dimer-Inference（TCR-induced Case）的四组体系八种结构（其中YLQ体系有两个结构已经处理过，改用“dimer”标签重新处理...）
## 24.1.9 处理YLQ-Inference的四个实验结构P4和四个建模结构L4
## 23.9.26 处理B27-Inference的四个实验结构27:05和三个建模结构27:09
#file="./data_preparation_pmhc/all-test_add_40_7.txt" 
## 23.12.7 处理All-Test新增的20个结构：增加后，数据集：7 Class 40 Structures


i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    ## 240714：-W ignore::FutureWarning 消除屏幕上的warning
    ## 24.Dec ## 为了获取TCR标签，依然需要存储TCR坐标
    python3  -W ignore::FutureWarning /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/00c-save_TCR_coords.py $pdb_info
    python3 -W ignore::FutureWarning /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-pdb_extract_and_triangulate.py $pdb_info
    python3 -W ignore::FutureWarning /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/04sc-masif_precompute.py $pdb_info masif_pmhc
    ## 24.Dec ## 为了确保结构经过质子化，将存储坐标放在特征工程最后一步
    python3  -W ignore::FutureWarning /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/00d-save_pep_coords.py $pdb_info
    

    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




