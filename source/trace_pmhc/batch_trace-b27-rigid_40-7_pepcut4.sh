#!/bin/bash

#file="./data_preparation_pmhc/sum_list_20_5.txt"
#file="./trace_pmhc/trace_corr_list_20.txt" 
## 23.10.26  批量处理：20个结构在静态All-Test中的patch溯源——本标签解释性

file="./trace_pmhc/trace_b27_list_pepcut4_6.txt" 
## 25.01.04  更换调参及align后的结构
## 24.12.22  40-7_pepcut4 标签解释性
## 24.03.08  批量处理：6个B27结构（包含问题结构：7N2R09）在新31-6分类器的标签解释性
## 23.11.06  批量处理：10个B27结构基于静态A2表面的patch溯源——标签解释性

####### 注意：本流程无需tensorflow环境

i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/trace_pmhc/extract_pool_importcsv_b27-rigid_40-7_pepcut4.py $pdb_info
    #python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-pdb_extract_and_triangulate.py $pdb_info
    python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/05-tracesurf_b27-rigid.py $pdb_info masif_pmhc
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




