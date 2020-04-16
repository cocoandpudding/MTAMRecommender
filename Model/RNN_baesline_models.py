from Model.Modules.gru import GRU
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.base_model import base_model
import tensorflow as tf

class RNN_Baseline_model(base_model):
    def __init__(self, FLAGS,Embeding,sess):

        super(RNN_Baseline_model, self).__init__(FLAGS, Embeding)  # 此处修改了
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

class T_SeqRec(RNN_Baseline_model):
    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):

            self.time_aware_gru_net_input = tf.concat([self.behavior_list_embedding_dense,
                                                  tf.expand_dims(self.timelast_list,2),
                                                  tf.expand_dims(self.timenow_list,2)],2)
            self.short_term_intent_temp = self.gru_net_ins.time_aware_gru_net(hidden_units=self.num_units,
                                                              input_data=self.time_aware_gru_net_input,
                                                              input_length=tf.add(self.seq_length,-1),
                                                              type = 'T-SeqRec')
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()

class Gru4Rec(RNN_Baseline_model):
    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                              input_data=self.behavior_list_embedding_dense,
                                                              input_length=tf.add(self.seq_length,-1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()

class Vallina_Gru4Rec(RNN_Baseline_model):
    def build_model(self):
        self.gru_net_ins = GRU()
        with tf.variable_scope('ShortTermIntentEncoder'):
            self.short_term_intent_temp = self.gru_net_ins.gru_net(hidden_units=self.num_units,
                                                              input_data=self.item_list_emb,
                                                              input_length=tf.add(self.seq_length,-1))
            self.short_term_intent = gather_indexes(batch_size=self.now_bacth_data_size,
                                                    seq_length=self.max_len,
                                                    width=self.num_units,
                                                    sequence_tensor=self.short_term_intent_temp,
                                                    positions=self.mask_index-1)
            self.short_term_intent = self.short_term_intent

            self.predict_behavior_emb = layer_norm(self.short_term_intent)
        self.output()