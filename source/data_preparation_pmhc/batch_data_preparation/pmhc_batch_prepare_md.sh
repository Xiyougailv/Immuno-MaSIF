#!/bin/bash

# file="./data_preparation_pmhc/md_list_20_5.txt"
file="./data_preparation_pmhc/tmp_list.txt" ## 出错补充处理

i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    ## 对每个结构，逐个处理其10张“脸”
    nf=10
    for ((f=1; f<=$nf; f ++))
    do
      python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/00c-md-save_TCR_coords.py $pdb_info $nf $f
      python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-md-pdb_extract_and_triangulate.py $pdb_info $nf $f 
      python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/04sc-md-masif_precompute.py $pdb_info masif_pmhc $nf $f
      echo " $i: $pdb_info $nf $f 处理完成^_^" ## 能不能打印出计数变量？？？
    done
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




