import tensorflow as tf

from Model.Modules.multihead_attention import Attention
from Model.Modules.net_utils import gather_indexes,layer_norm
from Model.Modules.time_aware_attention import Time_Aware_Attention
from Model.base_model import base_model
class Attention_Baseline_Model(base_model):
    def __init__(self, FLAGS,Embeding,sess):

        super(Attention_Baseline_Model, self).__init__(FLAGS, Embeding)  # 此处修改了
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
        self.timelast_list, \
        self.timenow_list, \
        self.target, \
        self.seq_length = self.embedding.get_embedding(self.num_units)
        self.max_len = self.FLAGS.length_of_user_history
        self.mask_index = tf.reshape(self.seq_length - 1, [self.now_bacth_data_size, 1])

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)

class Self_Attention_Model(Attention_Baseline_Model):
    def build_model(self):
        attention = Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = attention.self_attention(enc=self.behavior_list_embedding_dense, num_units=self.num_units,
                                                    num_heads=self.num_heads, num_blocks=self.num_blocks,
                                                    dropout_rate=self.dropout_rate, is_training=True, reuse=tf.AUTO_REUSE,
                                                    key_length=self.seq_length, query_length=self.seq_length)
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()
class Time_Aware_Self_Attention_Model(Attention_Baseline_Model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.self_attention(self.behavior_list_embedding_dense,
                                                                       self.num_units,
                                                                       self.num_heads, self.num_blocks,
                                                                       self.dropout_rate, is_training=True,
                                                                       reuse=False, key_length=self.seq_length,
                                                                       query_length=self.seq_length,
                                                                       t_querys=self.time_list,
                                                                       t_keys=self.time_list,
                                                                       t_keys_length=self.max_len, t_querys_length=self.max_len)
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()
class Ti_Self_Attention_Model(Attention_Baseline_Model):
    def build_model(self):
        time_aware_attention = Time_Aware_Attention()
        with tf.variable_scope("UserHistoryEncoder"):
            user_history = time_aware_attention.Tiself_attention(self.behavior_list_embedding_dense,
                                                                       self.num_units,
                                                                       self.num_heads, self.num_blocks,
                                                                       self.dropout_rate, is_training=True,
                                                                       reuse=False, key_length=self.seq_length,
                                                                       query_length=self.seq_length,
                                                                       t_querys=self.time_list,
                                                                       t_keys=self.time_list,
                                                                       t_keys_length=self.max_len, t_querys_length=self.max_len)
            long_term_prefernce = gather_indexes(batch_size=self.now_bacth_data_size, seq_length=self.max_len,
                                                 width=self.FLAGS.num_units, sequence_tensor=user_history,
                                                 positions=self.mask_index)
            self.predict_behavior_emb = long_term_prefernce
            self.predict_behavior_emb = layer_norm(self.predict_behavior_emb)
        self.output()