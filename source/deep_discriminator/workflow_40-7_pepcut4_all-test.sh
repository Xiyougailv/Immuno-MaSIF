#!/bin/bash

#### 23.12.07 用所有结构（7 Classes & 40 Structures）训练分类器 —— All-Test
#### 24.Dec 改变为pep-cutoff方式后，首先尝试使用40-7数据集 && 20关浅训练 && 4A cutoff超参
#### 24.Dec.15 重新对含有非标准残基的结构（3D3V/3D39）进行特征工程后，使用40-7数据集 && 50关训练 
####         注意标签顺序： T4H2(6), 868(7)

# 注：Dec.15 覆盖之前的文件（「gamenum」从301开始）
for ((i=301; i<=350; i ++))
do 
  echo "此刻来到 40-7_PepCut4_All-Test 探险游戏第 $((i)) 关，共 50 关 :)"
  echo "Step 1: 产生训练/测试数据："
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_all-test/makedata_40-7_pepcut4_all-test.py $i
  echo "Step 2: 训练模型&保存模型："
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_all-test/trainmodel_40-7_pepcut4_all-test.py $i
  echo "Step 3: 测试模型&保存预测："
  python3 /home/alcohol/MyMaSIF_tolinux/source/deep_network_40-7_pepcut4_all-test/testmodel_40-7_pepcut4_all-test.py $i
  echo "恭喜！第 $((i)) 轮已经通关，欢迎继续 ^_^"
done



