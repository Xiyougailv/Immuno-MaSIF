import os
import numpy as np
##from IPython.core.debugger import set_trace
import importlib
import sys
import csv ## For溯源: 2023.11
sys.path.append("/home/alcohol/MyMaSIF_tolinux/source")
from default_config.masif_opts import masif_opts
##from masif_modules.MaSIF_ligand import MaSIF_ligand
from masif_modules.MaSIF_pmhc import MaSIF_pmhc
from masif_modules.read_pmhc_tfrecords import _parse_function
from sklearn.metrics import confusion_matrix
##import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() ## “一劳永逸”法！！！！！


"""
masif_ligand_evaluate_test: Evaluate and test MaSIF-ligand.
Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

#####################
## 测试 MaSIF-pmhc 模型 ！
##     尚醇 202307

## patch的pMHC界面溯源： 202311
#####################

gamenum = sys.argv[1]

params = masif_opts["pmhc_fingerprint"]

# Load testing data
#### 溯源：由于空间不够，将tfrecords数据移到了80上的：
####    /data/alcohol/MaSIF_pMHC/tfrecords/b27-rigid/文件夹下
tfrecord_path = params["tfrecords_40-7_pepcut4_b27-rigid"] ## 24.Dec
testing_data = tf.data.TFRecordDataset(
    os.path.join(tfrecord_path, f"{gamenum}_testing_data_6.tfrecord") ## 24.Dec ## 0307，改动（注意，此时已经加载新测试集）
)
testing_data = testing_data.map(_parse_function)


model_dir = params["model_40-7_pepcut4_b27-rigid"] ## 24.Dec
output_model = model_dir + f"{gamenum}_model"


## 溯源改动1：结果文件夹
test_set_out_dir = params["predict_trace_40-7_pepcut4_b27-rigid"] ## 24.Dec
if not os.path.exists(test_set_out_dir):
    os.makedirs(test_set_out_dir)


with tf.Session() as sess:
    # Build network
    learning_obj = MaSIF_pmhc(
        sess,
        params["max_distance"],
        params["n_classes_40-7_b27-rigid"],
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        costfun=params["costfun"],
    )
    # Load pretrained network
    learning_obj.saver.restore(learning_obj.session, output_model)

    num_test_samples = 6  ## 注意修改 #0307
    testing_iterator = testing_data.make_one_shot_iterator()
    testing_next_element = testing_iterator.get_next()

    all_logits_softmax = []
    all_labels = []
    all_pdbs = []
    all_data_loss = []
    for num_test_sample in range(num_test_samples):
        try:
            data_element = sess.run(testing_next_element)
        except:
            continue

        print("Now in Test: ", num_test_sample)

        labels = data_element[4]
        n_ligands = labels.shape[1]
        pdb_logits_softmax = []
        pdb_labels = []
        for ligand in range(n_ligands):
            # Rows indicate point number and columns ligand type
            pocket_points = np.where(labels[:, ligand] != 0.0)[0]
            label = np.max(labels[:, ligand]) - 1
            pocket_labels = np.zeros(8, dtype=np.float32) ##################### 注意修改class数量 ## 0307，add a "fake" class
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            pdb_labels.append(label)
            pdb = data_element[5]
            # all_pdbs.append(pdb)

            ## 溯源改动2：以pdb_info区别的每个结构两张.csv大表：
            ###  第一张：存储每次sample的32个patches索引（一次sample一行 每次：1*32）
            ###  第二张：存储每次预测的logits结果，逐行一一对应于第一张表的sample（每次：1*5）
            sample_csv = os.path.join(test_set_out_dir, f'{pdb}_sample.csv')
            logits_csv = os.path.join(test_set_out_dir, f'{pdb}_logits.csv')
            ### 总索引表
            patch_pool = os.path.join(test_set_out_dir, f'{pdb}_pool.csv')

            samples_logits_softmax = []
            samples_data_loss = []
            # Make 100 predictions
            ## 溯源：将每一模型的预测次数增加到2000, 24.Dec 
            for i in range(5000):
                # Sample pocket randomly
                sample = np.random.choice(pocket_points, 32, replace=False)
                feed_dict = {
                    learning_obj.input_feat: data_element[0][sample, :, :],
                    learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                        sample, :, :
                    ],
                    learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                        sample, :, :
                    ],
                    learning_obj.mask: data_element[3][sample, :, :],
                    learning_obj.labels: pocket_labels,
                    learning_obj.keep_prob: 1.0,
                }

                logits_softmax, data_loss = learning_obj.session.run(
                    [learning_obj.logits_softmax, learning_obj.data_loss],
                    feed_dict=feed_dict,
                )


                ## 溯源改动3：追加模式将相应数据写入两张.csv大表 (注：with open会在使用完毕后自动关闭文件，无需close)
                with open(sample_csv, 'a', newline='') as f1:
                    writer1 = csv.writer(f1) ## 实例化
                    writer1.writerow(sample) ## 以行为单位写入
                    
                ## logits数据规范化：
                logits_0 = np.squeeze(logits_softmax) ## 修改格式：移除维度为1的数组
                logits = []
                ## 转为六位浮点数：
                for lg in logits_0:
                    lg = np.round(lg, 6)
                    logits.append(lg)
                with open(logits_csv, 'a', newline='') as f2:
                    writer2 = csv.writer(f2) ## 实例化
                    writer2.writerow(logits) ## 以行为单位写入


                samples_logits_softmax.append(logits_softmax)
                samples_data_loss.append(data_loss)

            pdb_logits_softmax.append(samples_logits_softmax)
        #np.save(test_set_out_dir + "{}_labels.npy".format(pdb), pdb_labels)
        #np.save(test_set_out_dir + "{}_logits.npy".format(pdb), pdb_logits_softmax)
        np.save(test_set_out_dir + f"{gamenum}_{pdb}_labels.npy", pdb_labels) ## pdb即pdb_info
        np.save(test_set_out_dir + f"{gamenum}_{pdb}_logits.npy", pdb_logits_softmax)

