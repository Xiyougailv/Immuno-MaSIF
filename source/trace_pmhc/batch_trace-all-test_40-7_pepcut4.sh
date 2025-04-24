#!/bin/bash

#file="./data_preparation_pmhc/sum_list_20_5.txt"
file="./trace_pmhc/trace_40-7_list.txt" 

## 24.Dec 批量处理：40-7结构，pepcut为4A，本标签解释性

i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/trace_pmhc/extract_pool_importcsv_40-7_pepcut4.py $pdb_info
    ## 24.Dec 注意确认['ply_file_template']是否有更动，有则重新计算01-pdb_extract_and_triangulate.py
    #python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-pdb_extract_and_triangulate.py $pdb_info
    python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/05-tracesurf.py $pdb_info masif_pmhc
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




