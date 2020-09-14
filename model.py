import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def batched_scalar_mul(w, x):
    x_t = tf.transpose(x, [2, 0, 1, 3])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.multiply(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2]), int(shape[3])])
    res = tf.transpose(res, [1, 2, 0, 3])
    return res

def batched_scalar_mul3(w, x):
    x_t = tf.transpose(x, [1, 0, 2])
    shape = x_t.get_shape()
    x_t = tf.reshape(x_t, [int(shape[0]), -1])
    wx_t = tf.multiply(w, x_t)
    res = tf.reshape(wx_t, [int(shape[0]), -1, int(shape[2])])
    res = tf.transpose(res, [1, 0, 2])
    return res

class DeepSim(object):
    def __init__(self, config, sess, node_embed, driver_embed, temporal_embed):
        
        self.n_sequences = config.n_sequences
        self.alpha = config.alpha
        self.learning_rate = config.learning_rate
        self.emb_learning_rate = config.emb_learning_rate
        self.training_iters = config.training_iters
        self.sequence_batch_size = config.sequence_batch_size 
        self.batch_size = config.batch_size
        self.display_step = config.display_step

        self.embedding_size = config.embedding_size
        self.n_input = config.n_input
        self.n_steps = config.n_steps
        self.n_hidden_gru = config.n_hidden_gru
        self.n_hidden_dense1 = config.n_hidden_dense1
        self.n_hidden_dense2 = config.n_hidden_dense2
        self.n_hidden_dense3 = config.n_hidden_dense3 #driverID onehot size
        self.n_hidden_dense4 =config.n_hidden_dense4
        self.scale1 = config.l1
        self.scale2 = config.l2
        self.scale = config.l1l2
        if config.activation == "tanh":
            self.activation = tf.tanh
        else:
            self.activation = tf.nn.relu
        self.max_grad_norm = config.max_grad_norm
        self.initializer = tf.random_normal_initializer(stddev=config.stddev)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.regularizer = tf.contrib.layers.l1_l2_regularizer(self.scale1, self.scale2)
        self.dropout_prob = config.dropout_prob
        self.sess = sess

        #######embedding#############
        self.deep_walk = node_embed
        self.driver = driver_embed
        self.temporal = temporal_embed
        # self.deep_walk = tf.cast(node_embed, tf.float32)

        self.name = "deepsim"
        self.size = tf.cast(160, tf.float32)
        
        
        self.build_input()
        self.build_var()
        self.pred_sim, self.pred_len, self.pred_tt, self.pred_fc = self.build_model()
        
        sim = self.y
        tt = self.y_tt
        fc = self.y_fc
        leng = self.y_len
        RN_EmbeddingUpdate = self.embedding
        Temporal_EmbeddingUpdate = self.embedding_temporal

        cost_sim = tf.reduce_mean(tf.pow(self.pred_sim - sim, 2))
        cost_len = tf.reduce_mean(tf.pow(self.pred_len - leng, 2)) 
        cost_tt = tf.reduce_mean(tf.pow(self.pred_tt - tt, 2))
        cost_fc = tf.reduce_mean(tf.pow(self.pred_fc - fc, 2))
   
        cost = (1-self.alpha)*cost_sim + self.alpha*(cost_len + cost_tt + cost_fc) + self.scale*tf.add_n([self.regularizer(var) for var in tf.trainable_variables()])
        error =(1-self.alpha)*cost_sim + self.alpha*(cost_len + cost_tt + cost_fc)


        
        ##embedding learning
        var_list1 = [var for var in tf.trainable_variables() if 'BiGRU' in var.name]
        var_list2 = [var for var in tf.trainable_variables() if 'dense' in var.name]
        var_list3 = [var for var in tf.trainable_variables() if 'driverF' in var.name]
        var_list4 = [var for var in tf.trainable_variables() if 'temporalDT' in var.name]
        var_list5 = [var for var in tf.trainable_variables() if 'tt_dense' in var.name]
        var_list6 = [var for var in tf.trainable_variables() if 'fc_dense' in var.name]
        var_list7 = [var for var in tf.trainable_variables() if 'len_dense' in var.name]
        var_list8 = [var for var in tf.trainable_variables() if 'embedding' in var.name]
        var_list9 = [var for var in tf.trainable_variables() if 'embedding_driver' in var.name]
        var_list10 = [var for var in tf.trainable_variables() if 'embedding_temporal' in var.name]
      
        
        opt1 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt2 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt3 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt4 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt5 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt6 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt7 = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        opt8 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        opt9 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)
        opt10 = tf.train.AdamOptimizer(learning_rate=self.emb_learning_rate)

        grads = tf.gradients(cost, var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + 
            var_list6 + var_list7 +var_list8 + var_list9 +var_list10)

        grads1 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[:len(var_list1)]]
        grads2 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1):len(var_list1+var_list2)]]
        grads3 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1+var_list2):len(var_list1+var_list2+var_list3)]]
        grads4 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1+var_list2+var_list3):len(var_list1+var_list2+var_list3+var_list4)]]
        grads5 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(var_list1 + var_list2 + var_list3+var_list4):len(var_list1 + var_list2 + var_list3+var_list4+var_list5)]]
        grads6 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(
            var_list1 + var_list2 + var_list3 + var_list4+var_list5):len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5+var_list6)]]
        grads7 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6): len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7)]]

        grads8 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7): len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7 + var_list8)]]

        grads9 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7 + var_list8): len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7 + var_list8 + var_list9)]]

        grads10 = [tf.clip_by_norm(grad, self.max_grad_norm) for grad in grads[len(
            var_list1 + var_list2 + var_list3 + var_list4 + var_list5 + var_list6 + var_list7 + var_list8 + var_list9):]]

        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train_op3 = opt3.apply_gradients(zip(grads3, var_list3))
        train_op4 = opt4.apply_gradients(zip(grads4, var_list4))
        train_op5 = opt5.apply_gradients(zip(grads5, var_list5))
        train_op6 = opt6.apply_gradients(zip(grads6, var_list6))
        train_op7 = opt7.apply_gradients(zip(grads7, var_list7))
        train_op8 = opt8.apply_gradients(zip(grads8, var_list8))
        train_op9 = opt9.apply_gradients(zip(grads9, var_list9))
        train_op10 = opt10.apply_gradients(zip(grads10, var_list10))

        train_op = tf.group(train_op1, train_op2, train_op3, train_op4, train_op5, train_op6, train_op7, train_op8, train_op9,train_op10)


        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.cost = cost
        self.error = error

        ###road network
        self.RN_NewEmbedding = RN_EmbeddingUpdate
        self.Temporal_NewEmbedding = Temporal_EmbeddingUpdate
        self.train_op = train_op
    
    def build_input(self):
        self.x = tf.placeholder(tf.int32, [None, self.n_sequences, self.n_steps], name="x")
        self.x_driver = tf.placeholder(tf.int32, [None, 1,1], name="x_driver")
        self.x_temporal = tf.placeholder(tf.int32, [None, 1,1], name="x_temporal")
        self.y = tf.placeholder(tf.float32, [None, 1], name="y")
        self.y_tt = tf.placeholder(tf.float32, [None, 1], name="y_tt")
        self.y_fc = tf.placeholder(tf.float32, [None, 1], name="y_fc")
        self.y_len = tf.placeholder(tf.float32, [None, 1], name="y_len")
        
    def build_var(self):
        with tf.variable_scope(self.name) as scope:
            ####PATH NODE
            with tf.variable_scope('embedding'):
                self.embedding = tf.get_variable('embedding', initializer=tf.constant(self.deep_walk, dtype=tf.float32))

            ####driver ID
            with tf.variable_scope('embedding_driver'):
                self.embedding_driver = tf.get_variable('embedding_driver',initializer=tf.constant(self.driver, dtype=tf.float32))
                
            ####temporal graph DT
            with tf.variable_scope('embedding_temporal'):
                self.embedding_temporal = tf.get_variable('embedding_temporal',initializer=tf.constant(self.temporal, dtype=tf.float32))

            with tf.variable_scope('BiGRU'):
                self.gru_fw_cell = rnn.GRUCell(self.n_hidden_gru)
                self.gru_bw_cell = rnn.GRUCell(self.n_hidden_gru)
           
            with tf.variable_scope('dense'):
                self.weights = {
                    'dense1': tf.get_variable('dense1_weight', initializer=self.initializer([5*self.n_hidden_dense2,
                                                                                             2*self.n_hidden_dense1])),
                    ####196*128
                    'dense2': tf.get_variable('dense2_weight', initializer=self.initializer([2*self.n_hidden_dense1,
                                                                                             self.n_hidden_dense1])),
                    #####128*64
                    'dense3': tf.get_variable('dense3_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                             self.n_hidden_dense2])),
                    #####64*16
                    'out': tf.get_variable('out_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases = {
                    'dense1': tf.get_variable('dense1_bias', initializer=self.initializer([2*self.n_hidden_dense1])),
                    'dense2': tf.get_variable('dense2_bias', initializer=self.initializer([self.n_hidden_dense1])),
                    'dense3': tf.get_variable('dense3_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out': tf.get_variable('out_bias', initializer=self.initializer([1]))
                }


            with tf.variable_scope('driverF'):
                self.weights_Dri = {
                    'denseF1': tf.get_variable('denseF1_weight', initializer=self.initializer([self.n_hidden_dense3,
                                                                                             4*self.n_hidden_dense1])),
                    'denseF2': tf.get_variable('denseF2_weight', initializer=self.initializer([4*self.n_hidden_dense1,
                                                                                             2*self.n_hidden_dense1])),
                    'denseF3':tf.get_variable('denseF3_weight', initializer=self.initializer([2*self.n_hidden_dense1,
                                                                                              self.n_hidden_dense4]))
                }
                self.biases_Dri = {
                    'denseF1': tf.get_variable('denseF1_bias', initializer=self.initializer([4*self.n_hidden_dense1])),
                    'denseF2': tf.get_variable('denseF2_bias', initializer=self.initializer([2*self.n_hidden_dense1])),
                    'denseF3': tf.get_variable('denseF3_bias', initializer=self.initializer([self.n_hidden_dense4]))
                }

            with tf.variable_scope('temporalDT'):
                self.weights_DT = {
                    'denseDT1': tf.get_variable('denseDT1_weight', initializer=self.initializer([self.n_hidden_dense2,
                                                                                             self.n_hidden_dense4])),
                }
                self.biases_DT = {
                    'denseDT1': tf.get_variable('denseDT1_bias', initializer=self.initializer([self.n_hidden_dense4])),
                }

            with tf.variable_scope('tt_dense'):
                self.weights_tt = {
                    'dense_tt_1': tf.get_variable('dense1_tt_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.n_hidden_dense1])),
                    'dense_tt_2': tf.get_variable('dense2_tt_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                             self.n_hidden_dense2])),
                    'out_tt': tf.get_variable('out_tt_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases_tt = {
                    'dense_tt_1': tf.get_variable('dense1_tt_bias', initializer=self.initializer([self.n_hidden_dense1])),
                    'dense_tt_2': tf.get_variable('dense2_tt_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out_tt': tf.get_variable('out_tt_bias', initializer=self.initializer([1])) #tt, fc, len
                }

            with tf.variable_scope('fc_dense'):
                self.weights_fc = {
                    'dense_fc_1': tf.get_variable('dense1_fc_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.n_hidden_dense1])),
                    'dense_fc_2': tf.get_variable('dense2_fc_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                             self.n_hidden_dense2])),
                    'out_fc': tf.get_variable('out_fc_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases_fc = {
                    'dense_fc_1': tf.get_variable('dense1_fc_bias', initializer=self.initializer([self.n_hidden_dense1])),
                    'dense_fc_2': tf.get_variable('dense2_fc_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out_fc': tf.get_variable('out_fc_bias', initializer=self.initializer([1])) #tt, fc, len
                }

            with tf.variable_scope('len_dense'):
                self.weights_len = {
                    'dense_len_1': tf.get_variable('dense1_len_weight', initializer=self.initializer([2 * self.n_hidden_gru,
                                                                                             self.n_hidden_dense1])),
                    'dense_len_2': tf.get_variable('dense2_len_weight', initializer=self.initializer([self.n_hidden_dense1,
                                                                                             self.n_hidden_dense2])),
                    'out_len': tf.get_variable('out_len_weight', initializer=self.initializer([self.n_hidden_dense2, 1]))
                }
                self.biases_len = {
                    'dense_len_1': tf.get_variable('dense1_len_bias', initializer=self.initializer([self.n_hidden_dense1])),
                    'dense_len_2': tf.get_variable('dense2_len_bias', initializer=self.initializer([self.n_hidden_dense2])),
                    'out_len': tf.get_variable('out_len_bias', initializer=self.initializer([1])) #tt, fc, len
                }
    
    def build_model(self):
        with tf.device('/gpu:0'):
            with tf.variable_scope('deepsim') as scope:
                with tf.variable_scope('embedding'):
                    x_vector = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding, self.x), 
                                             self.dropout_prob)

                    # (batch_size, n_sequences, n_steps, n_input)
                with tf.variable_scope('embedding_driver'):
                    x_driverN = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_driver, self.x_driver), 
                                             self.dropout_prob)
        
                    #(batch_size, n_input)
                with tf.variable_scope('embedding_temporal'):
                    x_temporalN = tf.nn.dropout(tf.nn.embedding_lookup(self.embedding_temporal, self.x_temporal),
                                             self.dropout_prob)
                    #(batch_size, n_input)
        
                with tf.variable_scope('BiGRU'):
                    x_vector = tf.transpose(x_vector, [1, 0, 2, 3])
                    # (n_sequences, batch_size, n_steps, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_steps, self.n_input])
                    # (n_sequences*batch_size, n_steps, n_input)
            
                    x_vector = tf.transpose(x_vector, [1, 0, 2])
                    # (n_steps, n_sequences*batch_size, n_input)
                    x_vector = tf.reshape(x_vector, [-1, self.n_input])
                    # (n_steps*n_sequences*batch_size, n_input)
            
                    # Split to get a list of 'n_steps' tensors of shape (n_sequences*batch_size, n_input)
                    x_vector = tf.split(x_vector, self.n_steps, 0)

                    outputs, _, _ = tf.nn.static_bidirectional_rnn(self.gru_fw_cell, self.gru_bw_cell, x_vector,
                                                          dtype=tf.float32)
                    hidden_states = tf.transpose(tf.stack(outputs), [1, 0, 2])
                    # (n_sequences*batch_size, n_steps, 2*n_hidden_gru)
                    hidden_states = tf.transpose(tf.reshape(hidden_states, [self.n_sequences, -1, self.n_steps, 2*self.n_hidden_gru]), [1, 0, 2, 3])
                    # (batch_size, n_sequences, n_steps, 2*n_hiddent_gru)

                    hidden_res = batched_scalar_mul(1.00, hidden_states)
                    shape = hidden_res.get_shape()
                    shape = [-1, int(shape[1]), int(shape[2]), int(shape[3])]
                    hidden_res = tf.reshape(hidden_res, [-1, shape[1], shape[2], shape[3]])
                    hidden_graph = tf.reduce_sum(hidden_res, reduction_indices=[1, 2])
                    ###bath_size 2*n_hidden_gru

                with tf.variable_scope('driverF'):
                    x_driverN1 = tf.reduce_sum(x_driverN, reduction_indices=[1,2])
                    denseF1 = self.activation(tf.add(tf.matmul(x_driverN1, self.weights_Dri['denseF1']), self.biases_Dri['denseF1']))
                    denseF2 = self.activation(tf.add(tf.matmul(denseF1,self.weights_Dri['denseF2']), self.biases_Dri['denseF2']))
                    driver_hidden = self.activation(tf.add(tf.matmul(denseF2, self.weights_Dri['denseF3']), self.biases_Dri['denseF3']))

                with tf.variable_scope('temporalDT'):
                    x_temporalN1 = tf.reduce_sum(x_temporalN, reduction_indices=[1,2])
                    temporal_hidden = self.activation(tf.add(tf.matmul(x_temporalN1, self.weights_DT['denseDT1']), self.biases_DT['denseDT1']))
        
                with tf.variable_scope('prediction'):
                    hidden_graphN = tf.concat([hidden_graph,driver_hidden,temporal_hidden],1) ###(batchsize 3*64)
                    dense1 = self.activation(tf.add(tf.matmul(hidden_graphN, self.weights['dense1']), self.biases['dense1']))
                    dense2 = self.activation(tf.add(tf.matmul(dense1, self.weights['dense2']), self.biases['dense2']))
                    dense3 = self.activation(tf.add(tf.matmul(dense2, self.weights['dense3']), self.biases['dense3']))
                    pred_sim = self.activation(tf.add(tf.matmul(dense3, self.weights['out']), self.biases['out']))

                #tt
                with tf.variable_scope('tt_prediction'):
                    dense_tt_1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights_tt['dense_tt_1']), self.biases_tt['dense_tt_1']))
                    dense_tt_2 = self.activation(tf.add(tf.matmul(dense_tt_1, self.weights_tt['dense_tt_2']), self.biases_tt['dense_tt_2']))
                    pred_tt = self.activation(tf.add(tf.matmul(dense_tt_2, self.weights_tt['out_tt']), self.biases_tt['out_tt']))

                #fc
                with tf.variable_scope('fc_prediction'):
                    dense_fc_1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights_fc['dense_fc_1']), self.biases_fc['dense_fc_1']))
                    dense_fc_2 = self.activation(tf.add(tf.matmul(dense_fc_1, self.weights_fc['dense_fc_2']), self.biases_fc['dense_fc_2']))
                    pred_fc = self.activation(tf.add(tf.matmul(dense_fc_2, self.weights_fc['out_fc']), self.biases_fc['out_fc']))

                #len
                with tf.variable_scope('len_prediction'):
                    dense_len_1 = self.activation(tf.add(tf.matmul(hidden_graph, self.weights_len['dense_len_1']), self.biases_len['dense_len_1']))
                    dense_len_2 = self.activation(tf.add(tf.matmul(dense_len_1, self.weights_len['dense_len_2']), self.biases_len['dense_len_2']))
                    pred_len = self.activation(tf.add(tf.matmul(dense_len_2, self.weights_len['out_len']), self.biases_len['out_len']))


                return pred_sim,pred_len, pred_tt, pred_fc
        
    def train_batch(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        self.sess.run(self.train_op, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def get_error(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.error, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def train_loss(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.cost, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def prediction_sim(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.pred_sim, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def prediction_len(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.pred_len, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def prediction_tt(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.pred_tt, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def prediction_fc(self, x, x_driver, x_temporal, y,y_len, y_tt, y_fc):
        return self.sess.run(self.pred_fc, feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y, self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def RN_Embedding(self, x, x_driver, x_temporal, y, y_len, y_tt, y_fc):
        return self.sess.run(self.RN_NewEmbedding,
                             feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y,
                                        self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})
    def Temporal_Embedding(self, x, x_driver, x_temporal, y, y_len, y_tt, y_fc):
        return self.sess.run(self.Temporal_NewEmbedding,
                             feed_dict={self.x: x, self.x_driver: x_driver, self.x_temporal: x_temporal, self.y: y,
                                        self.y_len: y_len, self.y_tt: y_tt, self.y_fc: y_fc})

