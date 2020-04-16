# coding=utf-8
import numpy as np
import tensorflow as tf
from Embedding.behavior_embedding_nodec import Behavior_embedding_nodec
from util.model_log import create_log
import pandas as pd
from trash.get_train_test_data import Get_train_test
from DataHandle.get_origin_data import Get_origin_data
from config import model_parameter
import matplotlib.pyplot as plt
from Model.ISTSBP_model import ISTSBP_model
from sklearn.manifold import TSNE
import os
import pickle

class generate_pic_class:

    def __init__(self,init = False,user_h = None,short_term_intent = None,attention_result = None,
                 item_table = None,item_category_dic = None):

        if init == True:

            model_parameter_ins = model_parameter()
            experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
            self.FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS
            log_ins = create_log(type = self.FLAGS.type, experiment_type = self.FLAGS.experiment_type,
                                 version=self.FLAGS.version)
            self.logger = log_ins.logger
            # self.model.user_h
            # self.model.short_term_intent
            # self.model.attention_result
            get_origin_data_ins = Get_origin_data(type = self.FLAGS.type,
                                                  raw_data_path=self.FLAGS.raw_data_path,
                                                  raw_data_path_meta=self.FLAGS.raw_data_path_meta,
                                                  logger=self.logger)

            origin_data = get_origin_data_ins.origin_data
            get_train_test_ins = Get_train_test(FLAGS=self.FLAGS,origin_data=origin_data)
            self.item_category_dic = get_train_test_ins.item_category_dic
            self.train_set, self.test_set = get_train_test_ins.get_train_test(mask_rate=self.FLAGS.mask_rate)


            self.sess = tf.Session()
            self.emb = Behavior_embedding_nodec(self.FLAGS.is_training,self.FLAGS.embedding_config_file)
            self.model = ISTSBP_model(self.FLAGS, self.emb, self.sess)

            # Initiate TF session
            # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            input_dic = self.emb.make_feed_dic(batch_data=self.test_set)
            input_dic[self.model.now_bacth_data_size] = len(self.test_set)

            self.model.restore(self.sess,path="data/check_point/Amazon_istsbp_vanilla_lr_0.001")
            #置于0
            self.model.init_reserved_field(self.sess)

            #get pic data
            self.user_h,self.short_term_intent,self.attention_result,self.item_table \
                = self.sess.run([self.model.user_h, self.model.short_term_intent,
                                 self.model.attention_result,self.emb.item_emb_lookup_table],input_dic)

            with open("data/gen_pic/user.h", 'wb') as f:
                pickle.dump(self.user_h, f, pickle.HIGHEST_PROTOCOL)
            with open("data/gen_pic/item_table.h", 'wb') as f:
                pickle.dump(self.item_table, f, pickle.HIGHEST_PROTOCOL)
            with open("data/gen_pic/item_category_dic", 'wb') as f:
                pickle.dump(item_category_dic, f, pickle.HIGHEST_PROTOCOL)
        else:

            self.user_h = user_h
            self.short_term_intent =  short_term_intent
            self.attention_result = attention_result
            self.item_table = item_table
            self.item_category_dic = item_category_dic




    def draw_picure(self,type,experiment_type,version,global_step = 0):

        gen_path_dir = "data/gen_pic/" + type + "_" + experiment_type + "_" + version
        if not os.path.exists(gen_path_dir):
            os.makedirs(gen_path_dir)

        user_history_path = gen_path_dir + "/gloal" + str(global_step) +"_user_history_"
        self.generate_user_history_pic(user_history_path)

        user_behavior_clustering_path = gen_path_dir + "/gloal" + str(global_step) +"_user_behavior_clustering_"
        self.generate_user_behavior_clustering_pic(user_behavior_clustering_path)



    def generate_emb_pic(self,path):

        plt.figure(figsize=(8, 5), dpi=80)
        ax = plt.subplot(111)

        cat_item_dic = {}

        for key,value in self.item_category_dic.items():
            key = int(key)
            if value in cat_item_dic.keys():
                cat_item_dic[value].append(key)

            else:
                cat_item_dic[value] = [key]


        result_mask = []
        count = 0

        #RGB
        colors = [[1,0,0],[0,1,0],[0,0,1],[1,0.5,0],[1,1,0]]
        index = 0
        for key,value in cat_item_dic.items():

            if len(cat_item_dic[key]) >= 200:
                row_index = np.array(cat_item_dic[key][:200])
                now_emb = self.item_table[row_index]
                X_embedded = TSNE(n_components=2).fit_transform(now_emb)
                count = count + 200
                ax.scatter(X_embedded[:,0], X_embedded[:,1], marker='o', color=colors[index], label='1',s=20)
                index = index + 1

            if index == 5:
                break

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([]) # 去掉纵坐标值
        plt.savefig(path, dpi=100)
        plt.show()

    def generate_user_history_pic(self,path):
        for index in range(0, self.user_h.shape[0]):
            d = self.user_h[index]
            df = pd.DataFrame(d)
            # 去掉padding
            #df = df.ix[~(df == 0).all(axis=1), :]

            index = index + 1
            if index % 50 == 0:
                print(df.shape)
                # df = df.T
                df = df.iloc[0:100, :]
                print(df.shape)
                fig = plt.figure()
                ax = fig.add_subplot(111)


                cax = ax.imshow(df, interpolation='nearest', cmap='coolwarm')
                plt.xticks([])  # 去掉横坐标值
                plt.yticks([])  # 去掉纵坐标值
                plt.xlabel('Item Embedding Space',fontdict={'family' : 'Times New Roman', 'size'   : 20})
                plt.ylabel('User Behavior Sequence',fontdict={'family' : 'Times New Roman', 'size'   : 20})
                path = path + str(index) + ".png"
                plt.savefig(path, dpi=100,bbox_inches ='tight')
                plt.show()

    def generate_user_behavior_clustering_pic(self,path):
        """
        plot user behavior clustering figure
        :param path: the folder 
        :return: 
        """""

        # variables = ['A', 'B', 'C', 'X']
        # labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3']
        # , columns=variables, index=labels
        #self.user_h = self.load_data("/Users/wendy/Documents/code/AUM_BE_LSTP/figure/item_table.h")


        colors = ['m', 'r', '#da70d6', 'b', 'c', 'g', '#7cb5ec']
        # colors = ['Blues', 'BuGn', 'BuPu','GnBu', 'Greens', 'Greys']
        # colors = ['#7cb5ec','#434348', '#90ed7d', '#f7a35c', '#8085e9', '#f15c80', '#e4d354']
        i = 0
        figure = 0

        for index in range(0, self.user_h.shape[0]):
            d = self.user_h[index]
            df = pd.DataFrame(d)
            # 去掉padding
            df = df.ix[~(df == 0).all(axis=1), :]

            index = index + 1
            if index % 50 == 0:
                # df = df.T
                df = df.iloc[0:100, :]
                user = np.array(df)
                if (i == 0):
                    user_set = user
                else:
                    user_set = np.concatenate((user_set, user))
                i = i + 1

                if i == 7:

                    plt.figure(figsize=(8, 5), dpi=500)
                    ax = plt.subplot(111)
                    tsne = TSNE(n_components=2, init='pca', random_state=0)
                    X_embedded = tsne.fit_transform(user_set)

                    for ii in range(0, 7):
                        lower = ii * 100
                        upper = (1 + ii) * 100
                        print(colors[ii] + ' ' + str(lower) + ' ' + str(upper))
                        ax.scatter(X_embedded[lower:upper, 0], X_embedded[lower:upper, 1], marker='o', color=colors[ii],
                                   label='1', s=10)
                    index = 0
                    plt.xticks([])  # 去掉横坐标值
                    plt.yticks([])  # 去掉纵坐标值
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    path = path  + str(index) + ".png"
                    plt.savefig(path, dpi=100,
                                bbox_inches='tight')
                    plt.show()
                    figure = figure + 1
                    i = 0

    def get_vector_pic(self):

        d = self.short_term_intent[0]
        df = pd.DataFrame(d)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(df,interpolation = 'nearest', cmap='coolwarm')
        #
        fig.colorbar(cax)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        plt.show()





if __name__ == '__main__':
    generate_pic_ins = generate_pic_class(init=True)
    generate_pic_ins.draw_picure(type="test",experiment_type="test",version="test",global_step=30)

