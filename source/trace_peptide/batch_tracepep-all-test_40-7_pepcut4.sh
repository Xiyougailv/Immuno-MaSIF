#!/bin/bash

#file="./data_preparation_pmhc/sum_list_20_5.txt"
#file="./trace_pmhc/trace_corr_list_20.txt" 
## 23.10.26  批量处理：20个结构在静态All-Test中的patch溯源——本标签解释性：表面打分

file="./trace_peptide/trace_corr_list_40-7.txt" 

## 23.11.08  批量处理：20个结构在静态All-Test中的patch溯源——本标签解释性：氨基酸打分
## 24.03.06  批量处理：31个结构在旧All-Test中的patch溯源——本标签解释性：氨基酸打分
## 24.12.09  32-6分类器（pepcut: 4A） 本标签解释性 （注意：.txt名单的顺序将决定生成的汇总pepscore.csv的顺序！）
## 24.12.17  40-7分类器（pepcut: 4A） 本标签解释性（共50个模型*400）

i=1
# 检查文件是否存在
if [ -f "$file" ]; then
  # 逐行读取文件
  while IFS= read -r line; do
    echo "$line"
    pdb_info="$line"
    python3 /home/alcohol/MyMaSIF_tolinux/source/trace_peptide/score_pep_csv_40-7_pepcut4.py $pdb_info
    echo " $i: $pdb_info 处理完成^_^" 
    ((i=i+1))
  done < "$file"
else
  echo "文件 $file 不存在"
fi




