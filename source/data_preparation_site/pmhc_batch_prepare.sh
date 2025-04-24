#!/bin/bash

#file="./data_preparation_pmhc/sum_list_20_5.txt"
file="./data_preparation_site/site_list.txt"  ## 5.27



i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_site/04sc-masif_precompute.py $pdb_info masif_site
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




