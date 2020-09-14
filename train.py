import sys
import numpy as np
import tensorflow as tf
from model import DeepSim
import six.moves.cPickle as pickle
import gzip

tf.set_random_seed(0)
import time
import pandas as pd
import os

NUM_THREADS = 20

#Where to save the trained model
ModelPath = '../Data/model_128/'

isExists = os.path.exists(ModelPath)
if not isExists:
    os.makedirs(ModelPath)


tf.flags.DEFINE_float("learning_rate", 0.0001, "learning_rate.")
tf.flags.DEFINE_float("alpha",0.6,"alpha.")
tf.flags.DEFINE_integer("sequence_batch_size", 120, "sequence batch size.")  #
tf.flags.DEFINE_integer("batch_size", 32, "batch size.")
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

x_train, x_temporal,x_driver,y_train,y_tt_train,y_fc_train,y_len_train = pickle.load(open('./Data/'
                                                                  'data_DT200915_example_train.pkl', 'rb'))
x_val, x_val_temporal,x_val_driver,y_val,y_tt_val,y_fc_val,y_len_val= pickle.load(open('./Data/'
                                                                      'data_DT200915_example_train.pkl', 'rb'))
deep_walk = pickle.load(open('./Data/road_network_200703_128.pkl', 'rb'))
driverID = pickle.load(open('./Data/driverid_onehot_0823_166.pkl', 'rb'))
temporalNode = pickle.load(open('./Data/temporalDT_node2vec_0826_new_16.pkl', 'rb'))


training_iters = len(x_train)
batch_size = config.batch_size
display_step = min(config.display_step, batch_size)

np.set_printoptions(precision=2)


sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
start = time.time()
model = DeepSim(config, sess, deep_walk, driverID, temporalNode) ####



old_val_error=1e9
old_error = 1e9

# Keep training until reach max iterations or max_try
train_error =[]
max_try = config.n_steps
patience = max_try
TTrainLoss = []
TrainError = []
TestError = []
ValError = []
TStep = []
TTStep = []
Dic ={}


saver = tf.train.Saver(max_to_keep=5)
for epoch in range (1,60+1):
    for step in range(training_iters // batch_size):
        Dic1 ={}
        batch_x, batch_x_driver, batch_x_temporal, batch_y, batch_y_tt, batch_y_fc, batch_y_len = get_batch(x_train, x_driver, x_temporal,y_train,y_tt_train, y_fc_train, y_len_train, step, batch_size=batch_size)
        
        model.train_batch(batch_x, batch_x_driver, batch_x_temporal,batch_y,batch_y_len, batch_y_tt,batch_y_fc)
        train_error.append(model.get_error(batch_x, batch_x_driver, batch_x_temporal,batch_y,batch_y_len, batch_y_tt,batch_y_fc))
        
        TrainLoss = model.train_loss(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        RN_Embedding = model.RN_Embedding(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        Temporal_Embedding =model.Temporal_Embedding(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        TTrainLoss.append(TrainLoss)
        TTStep.append(step)
        
        TrainError.append(np.mean(train_error))
        current_pred_sim = model.prediction_sim(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        current_pred_len = model.prediction_len(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        current_pred_tt = model.prediction_tt(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        current_pred_fc = model.prediction_fc(batch_x, batch_x_driver, batch_x_temporal, batch_y,batch_y_len, batch_y_tt, batch_y_fc)
        print("Epoch:" +"{}".format(epoch) + ", Training Step:" + "{}".format(step) +
          ", Training Loss= " + "{:.6f}".format(TrainLoss) +
            ", Train Error=" + "{:.6f}".format(np.mean(train_error)) +
          ", Prediction= " + "{:.6f}".format(np.mean(current_pred_sim)) +
          ", Prediction_tt= " + "{:.6f}".format(np.mean(current_pred_tt)) +
          ", Prediction_len= " + "{:.6f}".format(np.mean(current_pred_len)) +
          ", Prediction_fc= " + "{:.6f}".format(np.mean(current_pred_fc)))

        if TrainLoss <= old_error:
            old_error = TrainLoss
            model_name = 'model'

            # Calculate batch loss
            val_error = []
            pre_val = []
            for val_step in range(len(y_val)//batch_size):
                val_x, val_x_driver,val_x_temporal, val_y,val_y_tt,val_y_fc,val_y_len = get_batch(x_val, x_val_driver,x_val_temporal,y_val,y_tt_val, y_fc_val, y_len_val, val_step, batch_size=batch_size)
                current_error = model.get_error(val_x, val_x_driver,val_x_temporal,val_y,val_y_len, val_y_tt, val_y_fc)
                val_error.append(current_error)
                current_pred_val_sim = model.prediction_sim(val_x, val_x_driver,val_x_temporal,val_y,val_y_len, val_y_tt, val_y_fc)
                current_pred_val_len = model.prediction_len(val_x, val_x_driver,val_x_temporal,val_y,val_y_len, val_y_tt, val_y_fc)
                current_pred_val_tt = model.prediction_tt(val_x, val_x_driver,val_x_temporal,val_y,val_y_len, val_y_tt, val_y_fc)
                current_pred_val_fc = model.prediction_fc(val_x, val_x_driver,val_x_temporal,val_y,val_y_len, val_y_tt, val_y_fc)
                pre_val.append((current_pred_val_sim, current_pred_val_len,current_pred_val_tt, current_pred_val_fc))
                if np.mean(current_error) < old_val_error:
                    old_val_error = np.mean(current_error)
                    saver.save(sess, os.path.join(ModelPath, model_name), global_step=epoch)

                    print('=====================================')
                    print('               Model Saved!            ')
                    print('=====================================\n')
                    print("Validation Step: " + str(step) +
                        ", Training Error= " + "{:.6f}".format(np.mean(train_error)) +
                        ", Validation Error= " + "{:.6f}".format(np.mean(current_error))
                        )

print("Time:", time.time() - start)
print("Finished!\n----------------------------------------------------------------")


