import tensorflow as tf
from tensorflow.python.ops import variable_scope

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
import numpy as np


class MTAMRec_model(base_model):

    def __init__(self, FLAGS,Embeding,sess):

        super(MTAMRec_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)

class MTAM_only_time_aware_RNN(MTAMRec_model):

    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='T-SeqRec')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()
# Multi-hop Time-aware Attentive Memory network (MTAM)
class MTAM(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()
class MTAM_no_time_aware_rnn(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                                              input_data=self.behavior_list_embedding_dense,
                                                                              input_length=tf.add(self.seq_length, -1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            #z = tf.concat([self.short_term_intent, hybird_preference], 1)
            #z = tf.layers.dropout(tf.layers.dense(z, self.num_units, activation=tf.nn.relu), rate=self.FLAGS.dropout,
                                  #training=True)
            #z = tf.sigmoid(tf.layers.dense(z, 1))
            #self.predict_behavior_emb = layer_norm(z * hybird_preference + (1 - z) * self.short_term_intent)
        #self.output()
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()
class MTAM_no_time_aware_att(MTAMRec_model):
    def build_model(self):
        attention = Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32))
            #self.predict_behavior_emb = layer_norm(hybird_preference)
            self.predict_behavior_emb = hybird_preference
            #z = tf.concat([self.short_term_intent, hybird_preference], 1)
            #z = tf.layers.dropout(tf.layers.dense(z, self.num_units, activation=tf.nn.tanh), rate=self.FLAGS.dropout,
                                  #training=True)
            #z = tf.sigmoid(tf.layers.dense(z, 1))
            #self.predict_behavior_emb = layer_norm(z * hybird_preference + (1 - z) * self.short_term_intent)
        self.output()


class MTAM_via_T_GRU(MTAMRec_model):
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
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
            #z = tf.concat([self.short_term_intent, hybird_preference], 1)
            #z = tf.layers.dropout(tf.layers.dense(z, self.num_units, activation=tf.nn.relu), rate=self.FLAGS.dropout,
                                  #training=True)
            #z = tf.sigmoid(tf.layers.dense(z, 1))
            #self.predict_behavior_emb = layer_norm(z * hybird_preference + (1 - z) * self.short_term_intent)
        self.output()
        #self.output()
class MTAM_via_rnn(MTAMRec_model):
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
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
            #z = tf.concat([self.short_term_intent, hybird_preference], 1)
            #z = tf.layers.dropout(tf.layers.dense(z, self.num_units, activation=tf.nn.relu), rate=self.FLAGS.dropout,
                                  #training=True)
            #z = tf.sigmoid(tf.layers.dense(z, 1))
            #self.predict_behavior_emb = layer_norm(z * hybird_preference + (1 - z) * self.short_term_intent)
        self.output()
        #self.output()
class MTAM_hybird(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense

        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='new')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            #self.predict_behavior_emb = layer_norm(hybird_preference)
            self.predict_behavior_emb = tf.concat([self.short_term_intent,layer_norm(hybird_preference)],1)
        self.output_concat()

class MTAM_with_T_SeqRec(MTAMRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = self.behavior_list_embedding_dense
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                       tf.expand_dims(self.timelast_list, 2),
                                                       tf.expand_dims(self.timenow_list, 2)], 2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                                              input_data=self.time_aware_gru_net_input,
                                                                              input_length=tf.add(self.seq_length, -1),
                                                                              type='T-SeqRec')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index - 1)
            self.short_term_intent = self.short_term_intent


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, self.num_units,
                                                self.num_heads, self.num_blocks, self.dropout_rate,is_training=True,
                                                reuse=False,key_length=self.seq_length,
                                                query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                t_querys = tf.expand_dims(self.target[2],1),t_keys = self.time_list,
                                                t_keys_length=self.max_len,t_querys_length=1 )
            self.predict_behavior_emb = layer_norm(hybird_preference)
        self.output()











