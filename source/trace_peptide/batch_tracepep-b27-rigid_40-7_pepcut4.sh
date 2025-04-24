#!/bin/bash

## 24.12.18 2组B27结构在40-7分类器（pepcut:4A）的标签解释性：Class5 && Class 6
file="./trace_peptide/trace_b27_list_pepcut4_6.txt" 

## 24.03.08 批量处理：6个B27结构（包含问题结构：7N2RO9）在新31-6分类器的标签解释性：氨基酸打分




i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件 
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/trace_peptide/score_pep_csv_b27_40-7_pepcut4.py $pdb_info
    echo " $i: $pdb_info 处理完成^_^" ## 能不能打印出计数变量？？？
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




