#!/bin/bash

'''
使用MaSIF流程预处理40个cplx，产生代表mesh形式的.ply文件与质子化后的.pdb文件,
用于dMaSIF流程中：将mesh的iface标签投射到pointcloud
'''

file="./data_preparation_pmhc/dMaSIF_40.txt"  ## 24.9.21 



i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    #python3 /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/00c-save_TCR_coords.py $pdb_info
    ## 240714：-W ignore::FutureWarning 消除屏幕上的warning
    python3 -W ignore::FutureWarning /home/alcohol/MyMaSIF_tolinux/source/data_preparation_pmhc/01-dMaSIF-pdb_extract_and_triangulate.py $pdb_info
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




