#!/bin/bash

##-----------------------------
##  溯源第一步：
##  产生并记录大量的采样和预测结果，保存在.csv大表中
##-----------------------------

## 24.Dec 改动：快速解释性测试：对40-7分类器（4A pep cutoff）做All-Test溯源（一共20个模型，每个模型对某一结构做出500次预测）
#### 24.Dec.17 重新对含有非标准残基的结构（3D3V/3D39）进行特征工程后，使用40-7数据集 && 50关训练 
####         做All-Test溯源（每个模型对某一结构做出200次预测）
####         注意标签顺序： T4H2(6), 868(7)

for ((i=301; i<=350; i ++))
do 
  echo "此刻来到Trace_40-7_pepcut4_All-Test探险游戏第 $((i)) 关，共50关 :)"
  echo "测试模型&保存采样和预测："
  ## python3 /home/alcohol/MyMaSIF_tolinux/source/masif_pmhc/testmodel_trace-all-test.py $i
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_all-test/testmodel_trace_40-7_pepcut4_all-test.py $i
  echo "恭喜！第 $((i)) 轮已经通关，欢迎继续 ^_^"
done



