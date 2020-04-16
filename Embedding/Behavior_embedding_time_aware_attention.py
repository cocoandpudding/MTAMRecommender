import tensorflow as tf
from util.read_embedding_dic import embedding_csv_dic
import numpy as np
import copy
from Embedding.base_embedding import Base_embedding


class Behavior_embedding_time_aware_attention(Base_embedding):

    def __init__(self,is_training = True, user_count =0 ,
                 item_count=0, category_count=0, max_length_seq=0):

        super(Behavior_embedding_time_aware_attention, self).__init__(is_training)  # 此处修改了
        self.user_count = user_count
        self.item_count = item_count
        self.category_count = category_count
        self.position_count = max_length_seq
        #self.init_placeholders()


    def init_placeholders(self):

        with tf.variable_scope("input_layer"):

            # [B] user id
            self.user_id = tf.placeholder(tf.int32, [None, ],name = "user")
            # [B] item list (user history)
            self.item_list = tf.placeholder(tf.int32, [None,None],name = "item_seq")
            # category list
            self.category_list = tf.placeholder(tf.int32, [None, None],name='category_list')
            # time_list
            self.time_list = tf.placeholder(tf.float32, [None,None], name='time_list')
            # time_last list (the interval between the current item and its last item)
            self.timelast_list = tf.placeholder(tf.float32, [None, None],name='timelast_list')
            # time_now_list (the interval between the current item and the target item)
            self.timenow_list = tf.placeholder(tf.float32, [None,None], name='timenow_list')
            # position list
            self.position_list = tf.placeholder(tf.int32, [None, None],name='position_list')
            # target item id
            self.target_item_id = tf.placeholder(tf.int32, [None], name='target_item_id')
            # target item id
            self.target_item_category = tf.placeholder(tf.int32, [None], name='target_item_category')
            # target item id
            self.target_item_time = tf.placeholder(tf.float32, [None], name='target_item_time')
            # length of item list
            self.seq_length = tf.placeholder(tf.int32, [None,],name = "seq_length")


            # for key in self.embedding_dic.keys():
            #     key_id = key + "_list"
            #     if key != "item":
            #         temp_placeholder = tf.placeholder(tf.int32, [None, None],name=key_id)
            #         setattr(self, key_id, temp_placeholder)
            #         #target embedding placeholder
            #         key_target_id = key + "_positive"
            #         temp_placeholder = tf.placeholder(tf.int32, [None,],name=key_target_id)
            #         setattr(self, key_target_id, temp_placeholder)
            #         key_target_id = key + "_negative"
            #         temp_placeholder = tf.placeholder(tf.int32, [None,None], name=key_target_id)
            #         setattr(self, key_target_id, temp_placeholder)

    def get_embedding(self,num_units):
        # user embedding
        self.user_emb_lookup_table = self.init_embedding_lookup_table(name="user", total_count=self.user_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('user_emb_lookup_table', self.user_emb_lookup_table)
        user_embedding = tf.nn.embedding_lookup(self.user_emb_lookup_table, self.user_id)

        # item embedding
        self.item_emb_lookup_table = self.init_embedding_lookup_table(name="item", total_count=self.item_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('item_emb_lookup_table', self.item_emb_lookup_table)
        item_list_embedding = tf.nn.embedding_lookup(self.item_emb_lookup_table, self.item_list)

        # category embedding
        self.category_emb_lookup_table = self.init_embedding_lookup_table(name="category", total_count=self.category_count+3,
                                                                      embedding_dim=num_units,
                                                                      is_training=self.is_training)
        #tf.summary.histogram('category_emb_lookup_table', self.category_emb_lookup_table)
        category_list_embedding = tf.nn.embedding_lookup(self.category_emb_lookup_table, self.category_list)

        # position embedding
        self.position_emb_lookup_table = self.init_embedding_lookup_table(name="position",
                                                                          total_count=self.position_count+3,
                                                                          embedding_dim=num_units,
                                                                          is_training=self.is_training)
        #tf.summary.histogram('position_emb_lookup_table', self.position_emb_lookup_table)
        position_list_embedding = tf.nn.embedding_lookup(self.position_emb_lookup_table,
                                                         self.position_list)

        with tf.variable_scope("position_embedding"):

            behavior_list_embedding = tf.concat([item_list_embedding, category_list_embedding],
                                                  axis=2,
                                                  name="seq_embedding_concat")
            behavior_list_embedding_dense = tf.layers.dense(behavior_list_embedding, num_units,
                                                              activation=tf.nn.relu, use_bias=False,
                                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5),
                                                              name='dense4emb')
            #behavior_list_embedding_dense = tf.layers.dropout(behavior_list_embedding_dense, rate=0.5, training=tf.convert_to_tensor(True))
            behavior_list_embedding_dense = behavior_list_embedding_dense+position_list_embedding
            #behavior_list_embedding_dense = item_list_embedding + position_list_embedding
        return  user_embedding, \
                behavior_list_embedding_dense,\
                item_list_embedding,\
                category_list_embedding, \
                position_list_embedding,\
                self.time_list,\
                self.timelast_list,\
                self.timenow_list,\
                [self.target_item_id, self.target_item_category,self.target_item_time], \
                self.seq_length



    def tranform_list_ndarray(self,deal_data,max_len,index):

        result = np.zeros([len(self.batch_data),max_len],np.float)

        k = 0
        for t in deal_data:
            for l in range(len(t[1])):
                result[k][l] = t[k][l]
            k += 1

        return result


    def concat_time_emb(self,item_seq_emb,):

        if self.config['concat_time_emb'] == True:
            t_emb = tf.one_hot(self.hist_t, 12, dtype=tf.float32)
            item_seq_emb = tf.concat([item_seq_emb, t_emb], -1)
            item_seq_emb = tf.layers.dense(item_seq_emb, self.config['hidden_units'])
        else:
            t_emb = tf.layers.dense(tf.expand_dims(self.hist_t, -1),
                                    self.config['hidden_units'],
                                    activation=tf.nn.tanh)
            item_seq_emb += t_emb

        return item_seq_emb


    def make_feed_dic_new(self,batch_data):
        user_id = []
        item_list = []
        category_list = []
        time_list = []
        timelast_list = []
        timenow_list = []
        position_list = []
        target_id =[]
        target_category = []
        target_time = []
        length = []
        feed_dict = {}
        def normalize_time(time):
            time = np.log(time+np.ones_like(time))
            #_range = np.max(time) - np.min(time)
            return time/(np.mean(time)+1)
            #return (time - np.min(time)) / _range


        for example in batch_data:
            padding_size = [0,int(self.position_count-example[8])]
            user_id.append(example[0])
            item_list.append(np.pad(example[1],padding_size,'constant'))
            category_list.append(np.pad(example[2],padding_size,'constant'))
            time_list.append(np.pad(example[3],padding_size,'constant'))
            timelast_list.append(np.pad(example[4],padding_size,'constant'))
            timenow_list.append(np.pad(example[5],padding_size,'constant'))
            position_list.append(np.pad(example[6],padding_size,'constant'))
            target_id.append(example[7][0])
            target_category.append(example[7][1])
            target_time.append(example[7][2])
            length.append(example[8])

        feed_dict[self.user_id] = user_id
        feed_dict[self.item_list] = item_list
        feed_dict[self.category_list] = category_list
        feed_dict[self.time_list] = time_list
        feed_dict[self.timelast_list] = timelast_list
        feed_dict[self.timenow_list] = timenow_list
        feed_dict[self.position_list] = position_list
        feed_dict[self.target_item_id] = target_id
        feed_dict[self.target_item_category] = target_category
        feed_dict[self.target_item_time] = target_time
        feed_dict[self.seq_length] = length

        return feed_dict


        #feed_dic[self.]








