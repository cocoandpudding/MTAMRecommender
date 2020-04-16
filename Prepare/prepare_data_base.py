import random
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
import os
import copy
import random
from util.model_log import create_log
from Prepare.mask_data_process import mask_data_process
np.random.seed(1234)

class prepare_data_base():
    """
    Get_train_test(type,origin_data,experiment_type)

    generate training set and testing set
    -----------
    :parameter
        type: "Tmall", "Amazon"
        origin_data: Get_origin_data.origin_data
        experiment_type: "BSBE", "lSTSBP", "Atrank"....
        gapnum: number of gaps, default = 6
        user_count_limit: the limit of the data set
        test_frac： train test radio
    """

    def __init__(self, FLAGS, origin_data):

        self.FLAGS = FLAGS
        self.length = []
        self.type = FLAGS.type
        self.user_count_limit = FLAGS.user_count_limit
        self.test_frac = FLAGS.test_frac
        self.experiment_type = FLAGS.experiment_type
        self.neg_sample_ratio = FLAGS.neg_sample_ratio
        self.origin_data = origin_data

        #give the data whether to use action
        if self.type == "Tianchi":
            self.use_action = False
        else:
            self.use_action = False

        self.data_type_error = 0
        self.data_too_short = 0

        # give the random  target value
        #self.target_random_value

        # make origin data dir
        self.dataset_path = 'data/training_testing_data/' + self.type + "_" + \
                                  self.FLAGS.pos_embedding + "_" +      \
                                  self.FLAGS.experiment_data_type+'_' + \
                                  self.FLAGS.causality

        if not os.path.exists(self.dataset_path):
            os.mkdir(self.dataset_path)


        self.dataset_class_pkl = os.path.join(self.dataset_path,'parameters.pkl')
        self.dataset_class_train = os.path.join(self.dataset_path,'train_data.txt')
        self.dataset_class_test = os.path.join(self.dataset_path,'test_data.txt')
        self.mask_rate = self.FLAGS.mask_rate

        log_ins = create_log()
        self.logger = log_ins.logger

        # init or load
        if FLAGS.init_train_data == True:
            self.origin_data = origin_data
            self.get_gap_list(FLAGS.gap_num)
            self.map_process()
            self.filter_repetition()

        # load data
        else:
            # load train data
            with open(self.dataset_class_train, 'r') as f:
                self.train_set = []
                L = f.readlines()
                for line in L:
                    line = eval(line)
                    self.train_set.append(line)

            # load test data
            with open(self.dataset_class_test, 'r') as f:
                self.test_set = []
                L = f.readlines()
                for line in L:
                    line = eval(line)
                    self.test_set.append(line)

                # dont't need too large data set
                #if len(self.test_set) > 10000:
                    #self.test_set = random.sample(self.test_set, 10000)


            with open(self.dataset_class_pkl, 'rb') as f:

                data_dic = pickle.load(f)
                self.item_count = data_dic["item_count"]
                self.user_count = data_dic["user_count"]
                self.category_count = data_dic["category_count"]
                self.gap = data_dic["gap"]
                self.item_category_dic = data_dic["item_category"]
                self.logger.info("load data finish")
                self.logger.info('Size of training set is ' + str(len(self.train_set)))
                self.logger.info('Size of testing set is ' + str(len(self.test_set)))
                del data_dic

        self.init_train_data = FLAGS.init_train_data

    #give the index of item and category
    def map_process(self):
        """
        Map origin_data to one-hot-coding except time.

        """
        item_le = preprocessing.LabelEncoder()
        user_le = preprocessing.LabelEncoder()
        cat_le = preprocessing.LabelEncoder()

        # get item id list
        item_id = item_le.fit_transform(self.origin_data["item_id"].tolist())
        self.item_count = len(set(item_id))

        # get user id list
        user_id = user_le.fit_transform(self.origin_data["user_id"].tolist())
        self.user_count = len(set(user_id))

        # get category id list
        cat_id = cat_le.fit_transform(self.origin_data["cat_id"].tolist())
        self.category_count = len(set(cat_id))

        self.item_category_dic = {}
        for i in range(0, len(item_id)):
            self.item_category_dic[item_id[i]] = cat_id[i]

        self.logger.warning("item Count :" + str(len(item_le.classes_)))
        self.logger.info("user count is " + str(len(user_le.classes_)))
        self.logger.info("category count is " + str(len(cat_le.classes_)))

        # _key:key的列表，_map:key的列表加编号
        self.origin_data['item_id'] = item_id
        self.origin_data['user_id'] = user_id
        self.origin_data['cat_id'] = cat_id

        # 根据reviewerID、unixReviewTime编号进行排序（sort_values：排序函数）
        self.origin_data = self.origin_data.sort_values(['user_id', 'time_stamp'])

        # 重新建立索引
        self.origin_data = self.origin_data.reset_index(drop=True)
        return self.user_count, self.item_count

    #choose one for the action which are too close
    def filter_repetition(self):
        pass


    def get_train_test(self):
        """
        Generate training set and testing set with the mask_rate.
        The training set will be stored in training_set.pkl.
        The testing set will be stored in testing_set.pkl.
        dataset_path: 'data/training_testing_data/'
        :param
            data_size: number of samples
        :returns
            train_set: (user_id, item_list, (factor1_list, factor2,..., factorn), masked_item, label）
            test_set: (user_id, item_list, (factor1, factor2,..., factorn), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list), masked_item, label）
            test_set: (user_id, item_list,(time_interval_list, category_list), (masked_item_positive,masked_item_negtive)）
            e.g. Amazon_bsbe
            train_set: (user_id, item_list, (time_interval_list, category_list, action_list), masked_item, label）
            test_set: (user_id, item_list, (time_interval_list, category_list, action_list), (masked_item_positive,masked_item_negtive)）

        """
        if self.init_train_data == False:
            return self.train_set, self.test_set

        self.data_set = []
        self.train_set = []
        self.test_set = []

        self.now_count = 0

        #data_handle_process为各子类都使用的函数
        self.origin_data.groupby(["user_id"]).filter(lambda x: self.data_handle_process(x))
        # self.format_train_test()

        random.shuffle(self.train_set)
        random.shuffle(self.test_set)
        if len(self.test_set) > 20000:
            self.test_set = random.sample(self.test_set, 20000)

        self.logger.info('Size of training set is ' + str(len(self.train_set)))
        self.logger.info('Size of testing set is ' + str(len(self.test_set)))
        self.logger.info('Data type error size  is ' + str(self.data_type_error))
        self.logger.info('Data too short size is ' + str(self.data_too_short))


        with open(self.dataset_class_pkl, 'wb') as f:
            data_dic = {}
            data_dic["item_count"] = self.item_count
            data_dic["user_count"] = self.user_count
            data_dic["category_count"] = self.category_count
            data_dic["gap"] = self.gap
            data_dic["item_category"] = self.item_category_dic
            pickle.dump(data_dic, f, pickle.HIGHEST_PROTOCOL)

        # train text 和 test text 使用文本
        self.save(self.train_set,self.dataset_class_train)
        self.save(self.test_set,self.dataset_class_test)

        return self.train_set, self.test_set

    def data_handle_process_base(self, x):
        behavior_seq = copy.deepcopy(x)
        if self.FLAGS.remove_duplicate == True:
            behavior_seq = behavior_seq.drop_duplicates(keep="last")

        behavior_seq = behavior_seq.sort_values(by=['time_stamp'], na_position='first')
        behavior_seq = behavior_seq.reset_index(drop=True)
        columns_value = behavior_seq.columns.values.tolist()
        if "user_id" not in columns_value:
            self.data_type_error = self.data_type_error + 1
            return

        pos_list = behavior_seq['item_id'].tolist()  # asin属性的值
        length = len(pos_list)

        #limit length
        #if length < 2:
            #self.data_too_short = self.data_too_short + 1
            #return None

        # if length > self.FLAGS.length_of_user_history:
        #     behavior_seq = behavior_seq.tail(self.FLAGS.length_of_user_history)

        # user limit
        if self.now_count > self.user_count_limit:
            return None

        self.now_count = self.now_count + 1
        # test
        behavior_seq = behavior_seq.reset_index(drop=True)
        return behavior_seq

    #给出基本操作
    def data_handle_process(self, x):
        #Sort User sequence by time and delete sequences whose lenghts are not in [20,150]
        behavior_seq = self.data_handle_process_base(x)
        if behavior_seq is None:
            return

        mask_data_process_ins = mask_data_process(behavior_seq = behavior_seq,
                                                  use_action = self.use_action,
                                                  mask_rate = self.mask_rate)

        mask_data_process_ins.get_mask_index_list_behaivor()
        #根据测试训练的比例 来划分

        for index in mask_data_process_ins.mask_index_list:

            #这里只取单项
            user_id, item_seq_temp, factor_list = \
                mask_data_process_ins.mask_process_unidirectional(self.FLAGS.causality,
                                                                  index=index,
                                                                  time_window=24 * 3600 * 35,
                                                                  lengeth_limit=self.FLAGS.length_of_user_history)


            cat_list = factor_list[0]

            #换算成小时
            time_list = [int(x / 3600) for x in factor_list[1]]
            target_time = int(mask_data_process_ins.time_stamp_seq[index] / 3600)


            #mask the target item value
            item_seq_temp.append(self.item_count + 1)
            #mask the target category value
            cat_list.append(self.category_count + 1)

            #update time
            timelast_list, timenow_list = mask_data_process_ins.pro_time_method(time_list,target_time)
            position_list = mask_data_process_ins.proc_pos_emb(time_list)

            #进行padding的填充,便于对齐
            time_list.append(target_time)
            timelast_list.append(0)
            timenow_list.append(0)
            if index > 49:
                position_list.append(49)
            else:
                position_list.append(index)
            target_id = mask_data_process_ins.item_seq[index]
            target_category = self.item_category_dic[mask_data_process_ins.item_seq[index]]

            #以小时为准
            if index == len(mask_data_process_ins.mask_index_list) :
                self.test_set.append((user_id, item_seq_temp, cat_list, time_list,
                                      timelast_list, timenow_list, position_list,
                                      [target_id, target_category, target_time],
                                      len(item_seq_temp)))

            else:

                self.train_set.append((user_id, item_seq_temp, cat_list, time_list,
                                       timelast_list, timenow_list, position_list,
                                       [target_id, target_category, target_time],
                                       len(item_seq_temp)))


    def format_train_test(self):
        pass


    def get_gap_list(self, gapnum):
        gap = []
        for i in range(1, gapnum):
            if i == 1:
                gap.append(60)
            elif i == 2:
                gap.append(60 * 60)
            else:
                gap.append(3600 * 24 * np.power(2, i - 3))

        self.gap = np.array(gap)

    #给出写入文件
    def save(self,data_list,file_path):
        fp = open(file_path, 'w+')
        for i in data_list:
            fp.write(str(i) + '\n')

        fp.close()


