import tensorflow as tf

from Model.Modules.gru import GRU
from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
import numpy as np


class PISTRec_model(base_model):
    def __init__(self, FLAGS,Embeding,sess):

        super(PISTRec_model, self).__init__(FLAGS, Embeding)  # 此处修改了
        self.now_bacth_data_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.num_units = self.FLAGS.num_units
        self.num_heads = self.FLAGS.num_heads
        self.num_blocks = self.FLAGS.num_blocks
        self.dropout_rate = self.FLAGS.dropout
        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb, \
        self.category_list_emb, \
        self.position_list_emb, \
        self.time_list, \
        self.timelast_list,                                                                       \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)


class Time_Aware_self_Attention_model(PISTRec_model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense, num_units=128,
                                                               num_heads=self.num_heads, num_blocks=self.num_blocks,
                                                               dropout_rate=self.dropout_rate, is_training=True, reuse=False,
                                                               key_length=self.seq_length, query_length=self.seq_length,
                                                               t_querys=self.time_list, t_keys=self.time_list,
                                                               t_keys_length=self.max_len, t_querys_length=self.max_len
                                                               )
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        with tf.name_scope('CrossEntropyLoss'):

            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.item_list_emb),
                tf.nn.l2_loss(self.category_list_emb),
                tf.nn.l2_loss(self.position_list_emb)
            ])
            regulation_rate = self.FLAGS.regulation_rate
            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.target[0], [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=self.embedding.item_count + 3, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
            self.loss = regulation_rate * l2_norm + tf.reduce_mean(self.loss_origin)
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('Training Loss', self.loss)
            tf.summary.scalar('Learning_rate', self.learning_rate)

        self.cal_gradient(tf.trainable_variables())

class Time_Aware_Hybird_model(PISTRec_model):
    def build_model(self):

        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout

        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb,\
        self.category_list_emb, \
        self.position_list_emb,\
        self.time_list, \
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(num_units)
        max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length-1,[self.now_bacth_data_size,1])

        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,num_units=128,
                                                               num_heads=num_heads, num_blocks = num_blocks,
                                                               dropout_rate= dropout_rate, is_training=True,reuse=False,
                                                        key_length=self.seq_length, query_length = self.seq_length,
                                                               t_querys= self.time_list, t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=max_len
                                                               )

            self.user_h = user_history
        with tf.variable_scope('ShortTermIntentEncoder'):

            timelast_list = tf.expand_dims(self.timelast_list, 2)
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list,2),
                                                  tf.expand_dims(self.timenow_list,2)],2)
            self.error = tf.reshape(tf.add(self.seq_length, -1),[-1])
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=num_units,
                                                              input_data=self.time_aware_gru_net_input,
                                                              input_length=tf.add(self.seq_length,-1))

            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.FLAGS.max_len,
                                                    width=self.FLAGS.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, num_units,
                                                               num_heads, num_blocks, dropout_rate,is_training=True,
                                                               reuse=False,key_length=self.seq_length,
                                                               query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                               t_querys = tf.expand_dims(self.target[2],1),
                                                               t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=1 )

            #self.hybird_preference = hybird_preference


        with tf.variable_scope('OutputLayer'):
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.FLAGS.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.long_term_prefernce = long_term_prefernce
            self.short_term_intent = self.short_term_intent
            self.hybird_preference = hybird_preference

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.hybird_preference], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=3,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.regulation_rate))
            self.z = tf.nn.softmax(self.z)




            if self.FLAGS.PISTRec_type == 'hard':
                if tf.argmax(self.z) == 0:
                    self.predict_behavior_emb = self.long_term_prefernce
                elif tf.argmax(self.z) == 1:
                    self.predict_behavior_emb = self.short_term_intent
                else:
                    self.predict_behavior_emb = self.hybird_preference

            elif self.FLAGS.PISTRec_type == 'soft':
                self.predict_behavior_emb = tf.multiply(tf.slice(self.z,[0,0],[-1,1]),self.long_term_prefernce)+\
                                            tf.multiply(tf.slice(self.z, [0, 1], [-1, 1]), self.short_term_intent)+\
                                            tf.multiply(tf.slice(self.z, [0, 2], [-1, 1]), self.hybird_preference)
            elif self.FLAGS.PISTRec_type == 'short':
                self.predict_behavior_emb = self.short_term_intent
            elif self.FLAGS.PISTRec_type == 'long':
                self.predict_behavior_emb = self.long_term_prefernce
            elif self.FLAGS.PISTRec_type == 'hybird':
                self.predict_behavior_emb = self.hybird_preference

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.item_list_emb),
                tf.nn.l2_loss(self.category_list_emb),
                tf.nn.l2_loss(self.position_list_emb)
            ])
            regulation_rate = self.FLAGS.regulation_rate

            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.target[0], [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=self.embedding.item_count+3, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

            pistloss = regulation_rate * l2_norm +tf.reduce_mean(self.loss_origin)


        with tf.name_scope('LearningtoRankLoss'):
            self.loss = pistloss
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('Training Loss', self.loss)
            tf.summary.scalar('Learning_rate',self.learning_rate)

        self.cal_gradient(tf.trainable_variables())

class Time_Aware_RNN_model(PISTRec_model):
    def build_model(self):

        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout

        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb,\
        self.category_list_emb, \
        self.position_list_emb,\
        self.time_list, \
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(num_units)
        max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length-1,[self.now_bacth_data_size,1])

        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,num_units=128,
                                                               num_heads=num_heads, num_blocks = num_blocks,
                                                               dropout_rate= dropout_rate, is_training=True,reuse=False,
                                                        key_length=self.seq_length, query_length = self.seq_length,
                                                               t_querys= self.time_list, t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=max_len
                                                               )

            self.user_h = user_history
        with tf.variable_scope('ShortTermIntentEncoder'):

            timelast_list = tf.expand_dims(self.timelast_list, 2)
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list,2),
                                                  tf.expand_dims(self.timenow_list,2)],2)
            self.error = tf.reshape(tf.add(self.seq_length, -1),[-1])
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=num_units,
                                                              input_data=self.time_aware_gru_net_input,
                                                              input_length=tf.add(self.seq_length,-1))

            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.FLAGS.max_len,
                                                    width=self.FLAGS.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, num_units,
                                                               num_heads, num_blocks, dropout_rate,is_training=True,
                                                               reuse=False,key_length=self.seq_length,
                                                               query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                               t_querys = tf.expand_dims(self.target[2],1),
                                                               t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=1 )

            #self.hybird_preference = hybird_preference


        with tf.variable_scope('OutputLayer'):
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.FLAGS.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.long_term_prefernce = long_term_prefernce
            self.short_term_intent = self.short_term_intent
            self.hybird_preference = hybird_preference

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.hybird_preference], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=3,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.regulation_rate))
            self.z = tf.nn.softmax(self.z)




            if self.FLAGS.PISTRec_type == 'hard':
                if tf.argmax(self.z) == 0:
                    self.predict_behavior_emb = self.long_term_prefernce
                elif tf.argmax(self.z) == 1:
                    self.predict_behavior_emb = self.short_term_intent
                else:
                    self.predict_behavior_emb = self.hybird_preference

            elif self.FLAGS.PISTRec_type == 'soft':
                self.predict_behavior_emb = tf.multiply(tf.slice(self.z,[0,0],[-1,1]),self.long_term_prefernce)+\
                                            tf.multiply(tf.slice(self.z, [0, 1], [-1, 1]), self.short_term_intent)+\
                                            tf.multiply(tf.slice(self.z, [0, 2], [-1, 1]), self.hybird_preference)
            elif self.FLAGS.PISTRec_type == 'short':
                self.predict_behavior_emb = self.short_term_intent
            elif self.FLAGS.PISTRec_type == 'long':
                self.predict_behavior_emb = self.long_term_prefernce
            elif self.FLAGS.PISTRec_type == 'hybird':
                self.predict_behavior_emb = self.hybird_preference

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.item_list_emb),
                tf.nn.l2_loss(self.category_list_emb),
                tf.nn.l2_loss(self.position_list_emb)
            ])
            regulation_rate = self.FLAGS.regulation_rate

            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.target[0], [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=self.embedding.item_count+3, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

            pistloss = regulation_rate * l2_norm +tf.reduce_mean(self.loss_origin)


        with tf.name_scope('LearningtoRankLoss'):
            self.loss = pistloss
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('Training Loss', self.loss)
            tf.summary.scalar('Learning_rate',self.learning_rate)

        self.cal_gradient(tf.trainable_variables())

class Time_Aware_Recommender_via_switch_network(PISTRec_model):
    def build_model(self):

        num_units = self.FLAGS.num_units
        num_heads = self.FLAGS.num_heads
        num_blocks = self.FLAGS.num_blocks
        dropout_rate = self.FLAGS.dropout

        time_aware_attention = Time_Aware_Attention()
        self.gru_net_ins = GRU()

        self.user_embedding, \
        self.behavior_list_embedding_dense, \
        self.item_list_emb,\
        self.category_list_emb, \
        self.position_list_emb,\
        self.time_list, \
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(num_units)
        max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length-1,[self.now_bacth_data_size,1])

        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(enc=self.behavior_list_embedding_dense,num_units=128,
                                                               num_heads=num_heads, num_blocks = num_blocks,
                                                               dropout_rate= dropout_rate, is_training=True,reuse=False,
                                                        key_length=self.seq_length, query_length = self.seq_length,
                                                               t_querys= self.time_list, t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=max_len
                                                               )

            self.user_h = user_history
        with tf.variable_scope('ShortTermIntentEncoder'):

            timelast_list = tf.expand_dims(self.timelast_list, 2)
            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list,2),
                                                  tf.expand_dims(self.timenow_list,2)],2)
            self.error = tf.reshape(tf.add(self.seq_length, -1),[-1])
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=num_units,
                                                              input_data=self.time_aware_gru_net_input,
                                                              input_length=tf.add(self.seq_length,-1))

            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.FLAGS.max_len,
                                                    width=self.FLAGS.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)


            short_term_intent4vallina = tf.expand_dims(self.short_term_intent, 1)
        with tf.variable_scope('NextItemDecoder'):
            hybird_preference = time_aware_attention.vanilla_attention(user_history, short_term_intent4vallina, num_units,
                                                               num_heads, num_blocks, dropout_rate,is_training=True,
                                                               reuse=False,key_length=self.seq_length,
                                                               query_length = tf.ones_like(short_term_intent4vallina[:, 0, 0], dtype=tf.int32),
                                                               t_querys = tf.expand_dims(self.target[2],1),
                                                               t_keys = self.time_list,
                                                               t_keys_length=max_len,t_querys_length=1 )

            #self.hybird_preference = hybird_preference


        with tf.variable_scope('OutputLayer'):
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.FLAGS.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.long_term_prefernce = long_term_prefernce
            self.short_term_intent = self.short_term_intent
            self.hybird_preference = hybird_preference

            self.z_concate = tf.concat([self.long_term_prefernce, self.short_term_intent, self.hybird_preference], 1)

            self.z = tf.layers.dense(inputs=self.z_concate, units=3,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.regulation_rate))
            self.z = tf.nn.softmax(self.z)




            if self.FLAGS.PISTRec_type == 'hard':
                if tf.argmax(self.z) == 0:
                    self.predict_behavior_emb = self.long_term_prefernce
                elif tf.argmax(self.z) == 1:
                    self.predict_behavior_emb = self.short_term_intent
                else:
                    self.predict_behavior_emb = self.hybird_preference

            elif self.FLAGS.PISTRec_type == 'soft':
                self.predict_behavior_emb = tf.multiply(tf.slice(self.z,[0,0],[-1,1]),self.long_term_prefernce)+\
                                            tf.multiply(tf.slice(self.z, [0, 1], [-1, 1]), self.short_term_intent)+\
                                            tf.multiply(tf.slice(self.z, [0, 2], [-1, 1]), self.hybird_preference)
            elif self.FLAGS.PISTRec_type == 'short':
                self.predict_behavior_emb = self.short_term_intent
            elif self.FLAGS.PISTRec_type == 'long':
                self.predict_behavior_emb = self.long_term_prefernce
            elif self.FLAGS.PISTRec_type == 'hybird':
                self.predict_behavior_emb = self.hybird_preference

            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)

            l2_norm = tf.add_n([
                tf.nn.l2_loss(self.item_list_emb),
                tf.nn.l2_loss(self.category_list_emb),
                tf.nn.l2_loss(self.position_list_emb)
            ])
            regulation_rate = self.FLAGS.regulation_rate

            item_lookup_table_T = tf.transpose(self.embedding.item_emb_lookup_table)
            logits = tf.matmul(self.predict_behavior_emb, item_lookup_table_T)
            log_probs = tf.nn.log_softmax(logits)
            label_ids = tf.reshape(self.target[0], [-1])
            one_hot_labels = tf.one_hot(
                label_ids, depth=self.embedding.item_count+3, dtype=tf.float32)
            self.loss_origin = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])

            pistloss = regulation_rate * l2_norm +tf.reduce_mean(self.loss_origin)


        with tf.name_scope('LearningtoRankLoss'):
            self.loss = pistloss
            tf.summary.scalar('l2_norm', l2_norm)
            tf.summary.scalar('Training Loss', self.loss)
            tf.summary.scalar('Learning_rate',self.learning_rate)










