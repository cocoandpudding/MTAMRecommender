import random

import numpy

from Model.Modules.gru import GRU
from Model.Modules.net_utils import gather_indexes, layer_norm
from Model.base_model import base_model
import tensorflow as tf

class BPRMF(base_model):
    def __init__(self, FLAGS,Embeding,sess):

        super(BPRMF, self).__init__(FLAGS, Embeding)  # 此处修改了
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


        self.item_b_lookup_table   = self.embedding.init_embedding_lookup_table(name="item_b", total_count=self.embedding.item_count+3,
                                                                      embedding_dim=1, is_training=True)

        self.build_model()
        # self.cal_gradient(tf.trainable_variables())
        self.init_variables(sess, self.checkpoint_path_dir)

    def build_model(self):
        self.item_pos_target_id = self.target[0]
        self.item_neg_target_id = tf.random_uniform([1,],dtype='int32',minval=0,maxval=self.embedding.item_count,)
        u_embedding_result = self.user_embedding
        item_pos_embedding_result= tf.nn.embedding_lookup(self.embedding.item_emb_lookup_table, self.item_pos_target_id)
        item_neg_embedding_result = tf.nn.embedding_lookup(self.embedding.item_emb_lookup_table, self.item_neg_target_id)
        item_pos_b = tf.nn.embedding_lookup(self.item_b_lookup_table, self.item_pos_target_id)
        item_neg_b = tf.nn.embedding_lookup(self.item_b_lookup_table, self.item_neg_target_id)

        x = item_pos_b - item_neg_b + tf.reduce_sum(tf.multiply(u_embedding_result, (item_pos_embedding_result - item_neg_embedding_result)), 1)

        l2_norm = tf.add_n([
            tf.nn.l2_loss(u_embedding_result),
            tf.nn.l2_loss(item_pos_embedding_result),
            tf.nn.l2_loss(item_neg_embedding_result)
        ])
        regulation_rate = 5e-5
        self.predict_behavior_emb = u_embedding_result
        self.loss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
        tf.summary.scalar('Training Loss', self.loss)
        self.cal_gradient(tf.trainable_variables())


