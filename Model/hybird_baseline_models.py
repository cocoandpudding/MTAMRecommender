import tensorflow as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
import numpy as np


class Hybird_Baseline_Models(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(Hybird_Baseline_Models, self).__init__(FLAGS, Embeding)  # 此处修改了
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.num_units = self.FLAGS.num_units
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout
        self.regulation_rate = self.FLAGS.regulation_rate
        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb, \
        self.category_list_emb, \
        self.position_list_emb, \
        self.time_list, \
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.build_model()
        self.init_variables(sess, self.checkpoint_path_dir)


class LSTUR(Hybird_Baseline_Models):
    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net_initial(hidden_units=self.num_units,
                                                                           initial_state=self.user_embedding,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()
class LSTUR_time_rnn(Hybird_Baseline_Models):
    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.gru_net_initial_time(hidden_units=self.num_units,
                                                                           initial_state=self.user_embedding,
                                                                           input_data=self.behavior_list_embedding_dense,
                                                                           input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()
class NARM_time_att_time_rnn(Hybird_Baseline_Models):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            user_history = self.short_term_intent_temp
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                1, 1, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = tf.concat([self.short_term_intent, hybird_preference], 1)
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output_concat()
class NARM_time_att(Hybird_Baseline_Models):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            user_history = self.short_term_intent_temp
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                1, 1, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = tf.concat([self.short_term_intent, hybird_preference], 1)
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output_concat()
class NARM(Hybird_Baseline_Models):
    def build_model(self):
        attention = Attention()
        self.gru_net_ins = GRU()

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            user_history = self.short_term_intent_temp
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = layer_norm(self.short_term_intent)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)

        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                1, 1, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32))
            self.predict_behavior_emb = tf.concat([self.short_term_intent,hybird_preference],1)
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output_concat()
class STAMP(Hybird_Baseline_Models):
    def build_model(self):
        user_history = self.behavior_list_embedding_dense
        external_memory = layer_norm(tf.reduce_sum(user_history,1))
        last_click = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=user_history,
                                                    positions=self.mask_index-1)
        with tf.variable_scope('AttentionNet'):
            att_w0 = variable_scope.get_variable("att_w0",
                                                 shape=[self.num_units,1],
                                                 dtype=user_history.dtype)
            att_w1 = variable_scope.get_variable("att_w1",
                                                 shape=[self.num_units, self.num_units],
                                                 dtype=user_history.dtype)
            att_w2 = variable_scope.get_variable("att_w2",
                                                 shape=[self.num_units, self.num_units],
                                                 dtype=user_history.dtype)
            att_w3 = variable_scope.get_variable("att_w3",
                                                 shape=[self.num_units, self.num_units],
                                                 dtype=user_history.dtype)
            att_b = variable_scope.get_variable("att_b",
                                                 shape=[ 1,self.num_units],
                                                 dtype=user_history.dtype)
            a_history= tf.matmul(user_history,att_w1)
            a_external_memory = tf.matmul(external_memory, att_w2)
            a_last_click = tf.matmul(last_click, att_w3)
            att = a_history+ tf.expand_dims(a_external_memory,1)
            att = att + tf.expand_dims(a_last_click,1)
            att = tf.sigmoid(att)

            #att=tf.sigmoid(tf.matmul(user_history,att_w1)+\
            #tf.matmul(external_memory,att_w2)+ \
            #tf.matmul(last_click, att_w3)+att_b)
            att=tf.squeeze(tf.matmul(att,att_w0),2)
            ms= tf.matmul(att,user_history)
            ms = tf.reduce_sum(ms,1)

        with tf.variable_scope('MLPcellA'):
            self.hs = tf.layers.dense(ms, self.num_units,
                                 activation=tf.nn.relu, use_bias=False,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        with tf.variable_scope('MLPcellB'):
            self.ht = tf.layers.dense(last_click, self.num_units,
                                 activation=tf.nn.relu, use_bias=False,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        self.predict_behavior_emb =layer_norm(self.hs*self.ht)
        self.output()










