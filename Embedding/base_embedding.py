from util.read_embedding_dic import embedding_csv_dic
import tensorflow as tf
from util.model_log import create_log
import numpy as np


class Base_embedding():
    '''
    Base class of embedding module for all models.
    '''
    def __init__(self,is_training = True,config_file = None):

        self.embedding_file_path = config_file
        # self.embedding_dic = embedding_csv_dic(self.embedding_file_path)
        self.is_training = is_training
        #log_ins = create_log(type = self.FLAGS.type, experiment_type = self.FLAGS.experiment_type,version=self.FLAGS.version)
        log_ins = create_log()
        self.logger = log_ins.logger

        self.init_placeholders()

    def padding(self, one_list, max_len):

        if len(one_list) < max_len:

            np.pad(one_list, max_len, 'constant')

            # padding_list = [0] * (max_len - len(one_list))
            # one_list = one_list + padding_list

        if len(one_list) > max_len:

            one_list = one_list[:max_len]

        return one_list



    def init_placeholders(self):
        pass
    def get_embedding(self):
        pass
    def make_feed_dic(self,batch_data):
        pass

    def init_embedding_lookup_table(self,name,total_count,embedding_dim,is_training=True):

        total_count = int(total_count)
        embedding_dim = int(embedding_dim)
        # embedding_name = "embedding_name_" + name
        # saver.restore(sess, model_file)
        with tf.variable_scope("embedding_layer"):
            # see http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
            r = tf.sqrt(tf.cast(6 / embedding_dim, dtype="float32"))  # => sqrt( 6 / embedding_dim )
            # get embedding
            lookup_table = tf.get_variable(name, shape=[total_count, embedding_dim],
                                           initializer=tf.random_uniform_initializer(
                                               minval=-r, maxval=r),trainable=is_training)

        return lookup_table





