import tensorflow as tf
import os
import time
import numpy as np
import pickle
from model import DeepSim
import pandas as pd


#####CHANGE
ModelPath = './Data/model_128/'


#
tf.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate.")
tf.flags.DEFINE_float("alpha",0.6,"alpha.")
tf.flags.DEFINE_integer("sequence_batch_size", 120, "sequence batch size.")  #
tf.flags.DEFINE_integer("batch_size", 1, "batch size.")
tf.flags.DEFINE_integer("n_hidden_gru", 32, "hidden gru size.")
tf.flags.DEFINE_float("l1", 3e-5, "l1.")
tf.flags.DEFINE_float("l2", 2e-8, "l2.")
tf.flags.DEFINE_float("l1l2", 0.001, "l1l2.")
tf.flags.DEFINE_string("activation", "relu", "activation function.")
tf.flags.DEFINE_integer("n_sequences", 1, "num of sequences.")  # each source to destination has ten path
tf.flags.DEFINE_integer("training_iters", 100000, "max training iters.")
tf.flags.DEFINE_integer("display_step", 50, "display step.")
tf.flags.DEFINE_integer("embedding_size", 128, "embedding size.")
tf.flags.DEFINE_integer("n_input", 128, "input size.")  # embedding dimension
tf.flags.DEFINE_integer("n_steps", 60, "num of step.")  # length of sequence
tf.flags.DEFINE_integer("n_hidden_dense1",  32, "dense1 size.")
tf.flags.DEFINE_integer("n_hidden_dense2",  16, "dense2 size.")
tf.flags.DEFINE_integer("n_hidden_dense3", 166, "dense3 size.") ###driver ID
tf.flags.DEFINE_integer("n_hidden_dense4",   8, "dense4_size.")
tf.flags.DEFINE_string("version", "v4", "data version.")
tf.flags.DEFINE_integer("max_grad_norm", 100, "gradient clip.")
tf.flags.DEFINE_float("stddev", 0.01, "initialization stddev.")
tf.flags.DEFINE_float("emb_learning_rate", 5e-05, "embedding learning_rate.")
tf.flags.DEFINE_float("dropout_prob", 1., "dropout probability.")

config = tf.flags.FLAGS


def get_batch(x, x_driver, x_temporal, y, y_tt, y_fc, y_len,step, batch_size=config.batch_size):

    batch_x = np.zeros((batch_size, len(x[0]), len(x[0][0])))
    batch_x_driver = np.zeros((batch_size, len(x_driver[0]), len(x_driver[0][0])))
    batch_x_temporal = np.zeros((batch_size, len(x_temporal[0]), len(x_temporal[0][0])))
    batch_y = np.zeros((batch_size, 1))
    batch_y_tt = np.zeros((batch_size, 1))
    batch_y_fc = np.zeros((batch_size, 1))
    batch_y_len = np.zeros((batch_size, 1))
    start = step * batch_size % len(x)

    for i in range(batch_size):
        batch_y[i, 0] = y[(i + start) % len(x)]
        batch_y_tt[i, 0] = y_tt[(i + start) % len(x)]
        batch_y_fc[i, 0] = y_fc[(i + start) % len(x)]
        batch_y_len[i, 0] = y_len[(i + start) % len(x)]
        batch_x[i, :] = np.array(x[(i + start) % len(x)])
        batch_x_driver[i, :] = np.array(x_driver[(i + start) % len(x)])
        batch_x_temporal[i, :] = np.array(x_temporal[(i + start) % len(x)])
    return batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_tt, batch_y_fc, batch_y_len


version = config.version


###change this to the testing data
x_train, x_temporal,x_driver,y_train,y_tt_train,y_fc_train,y_len_train = pickle.load(open('./Data/'
                                                                  'data_DT200915_example_train.pkl', 'rb'))
deep_walk = pickle.load(open('./Data/road_network_200703_128.pkl', 'rb'))
driverID = pickle.load(open('./Data/driverid_onehot_0823_166.pkl', 'rb'))
temporalNode = pickle.load(open('./Data/temporalDT_node2vec_0826_new_16.pkl', 'rb'))

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
model = DeepSim(config, sess, deep_walk, driverID, temporalNode)
saver = tf.train.Saver()

#Model Loade
print('=====================================')
print('             Model Loading!            ')
print('=====================================\n')
model_out_dir = ModelPath
ckpt = tf.train.get_checkpoint_state(model_out_dir)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(sess, os.path.join(model_out_dir, ckpt_name))
else:
    raise FileNotFoundError('Not Found')

sim_pre = []
tt_pre = []
len_pre = []
fc_pre = []

PT = []
Step = []
print('=====================================')
print('               Testing!            ')
print('=====================================\n')
for iter in range(len(x_train)//config.batch_size):
    batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_tt, batch_y_fc, batch_y_len = get_batch(x_train,
                                                                                                        x_driver,
                                                                                                        x_temporal,
                                                                                                        y_train,
                                                                                                        y_tt_train,
                                                                                                        y_fc_train,
                                                                                                        y_len_train,
                                                                                                        iter,
                                                                                                        batch_size=config.batch_size)

    start = time.time()

    current_pred_sim = model.prediction_sim(batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_len, batch_y_tt,
                                            batch_y_fc)
    current_pred_len = model.prediction_len(batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_len, batch_y_tt,
                                            batch_y_fc)
    current_pred_tt = model.prediction_tt(batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_len, batch_y_tt,
                                          batch_y_fc)
    current_pred_fc = model.prediction_fc(batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_len, batch_y_tt,
                                          batch_y_fc)

    processing_time = time.time() - start
    sim_pre.append(current_pred_sim[0][0])
    tt_pre.append(current_pred_tt[0][0])
    fc_pre.append(current_pred_fc[0][0])
    len_pre.append(current_pred_len[0][0])
    PT.append(processing_time)
    Step.append(iter)
    print("step:", iter, "Sim Prediction: ", current_pred_sim[0][0],'Processing Time:', processing_time)
print('=====================================')
print('             Finishing!            ')
print('=====================================\n')


