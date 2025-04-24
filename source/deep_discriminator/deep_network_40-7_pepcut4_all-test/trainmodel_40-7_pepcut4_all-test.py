# Header variables and parameters.
import os
import numpy as np
##from IPython.core.debugger import set_trace
import importlib
import sys
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
masif_ligand_train.py: Train MaSIF-ligand. 
Freyr Sverrisson - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""

#####################
## 训练 MaSIF-pmhc 模型 ！
##     尚醇 202307
#####################

gamenum = sys.argv[1]

params = masif_opts["pmhc_fingerprint"]

# Load dataset
training_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_40-7_pepcut4_all-test"], f"{gamenum}_training_data_27.tfrecord") ## 载入40-7 tfrecords文件 12.07
)
##validation_data = tf.data.TFRecordDataset(
#    os.path.join(params["tfrecords_dir"], "validation_data_sequenceSplit_30.tfrecord")
#)
testing_data = tf.data.TFRecordDataset(
    os.path.join(params["tfrecords_40-7_pepcut4_all-test"], f"{gamenum}_testing_data_13.tfrecord") ## 载入40-7 tfrecords文件 12.07
)
training_data = training_data.map(_parse_function)
## Dataset.map(f) ：对数据集中的每个元素应用函数 f ，得到一个新的数据集
## validation_data = validation_data.map(_parse_function)
testing_data = testing_data.map(_parse_function)

out_dir = params["model_40-7_pepcut4_all-test"] ## 24.Dec
output_model = out_dir + f"{gamenum}_model"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

## tf.compat.v1.Session() tf.Session()
with tf.compat.v1.Session() as sess:
    # Build the neural network model
    learning_obj = MaSIF_pmhc(
        sess,
        params["max_distance"],
        params["n_classes_40-7_all-test"],
        idx_gpu="/gpu:0",
        feat_mask=params["feat_mask"],
        costfun=params["costfun"],
    )
    # learning_obj.saver.restore(learning_obj.session, 'monet_models/model')
    ##best_validation_loss = 1000
    ##best_validation_accuracy = 0.0
    total_iterations = 0
    num_epochs = 100
    for num_epoch in range(num_epochs):
        num_training_samples = 27 ## 40-7 12.07
        #num_validation_samples = 120
        num_testing_samples = 13 ## 40-7 12.07
        ##td = tf.data.Dataset.from_tensor_slices(training_data)
        training_iterator = training_data.make_one_shot_iterator()
        ##training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_data)
        training_next_element = training_iterator.get_next()
        ##validation_iterator = validation_data.make_one_shot_iterator()
        ##validation_next_element = validation_iterator.get_next()
        testing_iterator = tf.data.make_one_shot_iterator(testing_data)
        testing_next_element = testing_iterator.get_next()

        ##print("Total iterations", total_iterations)
        print("Now in Epoch: ", num_epoch)

        # Train the network
        training_losses = []
        training_ytrue = []
        training_ypred = []
        #training_iterator = training_data.make_one_shot_iterator()
        #training_next_element = training_iterator.get_next()
        for num_sample in range(num_training_samples):
            try:
                data_element = sess.run(training_next_element)
            except:
                continue
            labels = data_element[4]
            n_ligands = labels.shape[1]
            random_ligand = np.random.choice(n_ligands, 1)
            pocket_points = np.where(labels[:, random_ligand] != 0.0)[0] ## 所有范围内patch center vert
            label = np.max(labels[:, random_ligand]) - 1 ## 其实值都一样？？
            pocket_labels = np.zeros(7, dtype=np.float32)  ## 注意修改class数量 For 40-7, 12.07
            pocket_labels[label] = 1.0
            npoints = pocket_points.shape[0]
            if npoints < 32:
                continue
            # Sample 32 points randomly
            sample = np.random.choice(pocket_points, 32, replace=False)
            feed_dict = {
                learning_obj.input_feat: data_element[0][sample, :, :],
                learning_obj.rho_coords: np.expand_dims(data_element[1], -1)[
                    sample, :, :
                ],
                learning_obj.theta_coords: np.expand_dims(data_element[2], -1)[
                    sample, :, :
                ],
                learning_obj.mask: data_element[3][pocket_points[:32], :, :],
                learning_obj.labels: pocket_labels,
                learning_obj.keep_prob: 1.0,
            }

            _, training_loss, norm_grad, logits, logits_softmax, computed_loss = learning_obj.session.run(
                [
                    learning_obj.optimizer,
                    learning_obj.data_loss,
                    learning_obj.norm_grad,
                    learning_obj.logits,
                    learning_obj.logits_softmax,
                    learning_obj.computed_loss,
                ],
                feed_dict=feed_dict,
            )
            training_losses.append(training_loss)
            # training_ytrue.append(label)
            # training_ypred.append(np.argmax(logits_softmax))
            print(
                "Num sample {}\tTraining loss {}\nLabels {}\tSoftmax logits {}\tComputed loss {}\n".format(
                    num_sample,
                    training_loss,
                    pocket_labels,
                    logits_softmax,
                    computed_loss,
                )
            )
            '''
            if num_sample % 50 == 0:
                print(
                    "Mean training loss {}, median training loss {}".format(
                        np.mean(training_losses), np.median(training_losses)
                    )
                )
            ''' ## SC
            total_iterations += 1
            if total_iterations == 40000:
                break

        #### 每十代保存一个模型
        if (num_epoch + 1) % 10 == 0:
            print("^_^  Epoch: ", num_epoch)
            print(
                "Mean training loss {}, median training loss {}".format(
                    np.mean(training_losses), np.median(training_losses)
                )
            )
            print("Saving model")
            learning_obj.saver.save(learning_obj.session, output_model)


