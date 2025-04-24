#!/bin/bash

#file="./data_preparation_pmhc/sum_list_20_5.txt"
file="./trace_pmhc/trace_corr_list_31.txt" 

## 23.10.26  批量处理：20个结构在静态All-Test中的patch溯源——本标签解释性
## 24.03.05  批量处理：旧31-6分类器在All-Test中的patch溯源——本标签解释性

i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/trace_pmhc/extract_pool_importcsv.py $pdb_info
    python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-pdb_extract_and_triangulate.py $pdb_info
    python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/05-tracesurf.py $pdb_info masif_pmhc
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




