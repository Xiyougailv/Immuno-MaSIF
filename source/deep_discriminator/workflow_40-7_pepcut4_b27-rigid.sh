#!/bin/bash

## 25-01-03 使用前面训练出的40-7分类器，重新完成B27-Infer：结构集经过align处理、调节MSMS(probe:2.0)及cutoff参数(5A)
## 24-12-17 使用40-7分类器（pepcut--4A）训练一个模型，完成B27-Infer
##    （为确认重复性，训练3个模型 & 每个模型推断 2000次/结构，同时直接为后续溯源记录预测情况）

## 24-12-09 更换为pep cutoff并进行参数/结构集测试后，进行B27-Infer：
##          32-6分类器 && pepcut 4A && 训练20个模型

### 23-12-22 进行29-6分类器的B27-Infer：共推断六个结构（3实验晶体+3建模）；共20关 
##         （注：29-6分类器的All-Test正确率还未经测试 -> Baseline应为31-6分类器）
### 24-03-03 重新挑选并测试数据，进行新31-6分类器的B27-Infer：同上

for ((i=1; i<=2; i ++)) 
do 
  echo "此刻来到 40-7_PepCut4_B27-Rigid探险游戏第 $((i)) 关，共2关 :)"
  echo "Step 1: 产生训练/测试数据："
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_b27-rigid/makedata_40-7_pepcut4_b27-rigid.py $i
  echo "Step 2: 训练模型&保存模型："
  #python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_b27-rigid/trainmodel_40-7_pepcut4_b27-rigid.py $i
  echo "Step 3: 测试模型&保存预测："
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_b27-rigid/testmodel_trace_40-7_pepcut4_b27-rigid.py $i
  echo "恭喜！第 $((i)) 轮已经通关，欢迎继续 ^_^"
done



    