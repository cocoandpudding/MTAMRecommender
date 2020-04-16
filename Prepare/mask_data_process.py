import random
import numpy as np
import pickle
import pandas as pd
from sklearn import preprocessing
import os
import copy
import random
from util.model_log import create_log


class mask_data_process():

    """
       Get_train_test(type,origin_data,experiment_type)

       generate training set and testing set
       -----------
       :parameter
           type: "Tmall", "Amazon"
           origin_data: Get_origin_data.origin_data
           experiment_type: "BSBE", "lSTSBP", "Atrank"....
           gapnum: number of gaps, default = 6
           data_set_limit: the limit of the data set
           test_frac： train test radio
       """

    def __init__(self, behavior_seq,use_action,mask_rate):

        # the list for masking item
        mask_index_list = []

        #origin data
        self.user_seq = behavior_seq['user_id'].tolist()
        self.item_seq = behavior_seq["item_id"].tolist()
        self.category_seq = behavior_seq["cat_id"].tolist()
        self.time_stamp_seq = behavior_seq["time_stamp"].tolist()
        self.use_action = use_action
        self.length = behavior_seq.shape[0]

        #offset and  random_value
        # self.offset = offset
        # self.target_random_value = self.offset - 2

        if self.use_action == True:
            action_seq = behavior_seq['action_type'].tolist()
            self.action_seq = [i + self.offset for i in action_seq]

        # count the number of purchases ('action_type'=2)
        # filter too few items if use action
        if self.use_action == True:
            self.puchases_df = behavior_seq.loc[behavior_seq['action_type'] == 2]
            self.puchases_action_num = self.puchases_df.shape[0]

        self.mask_rate = mask_rate


    # get the mask item list
    def get_mask_index_list_behaivor(self,only_last = False):

        if only_last == True:
            # get the last index
            mask_index_list = [self.length - 1]

        elif self.use_action == False:
            mask_index_list = [i for i in range(1, self.length)]

        # if use index use only action index
        else:
            mask_index_list = list(self.puchases_df.index)

        self.mask_index_list = mask_index_list


    def get_mask_index_list_bert(self):

        # Generate random mask index
        if self.use_action == False:
            num_to_predict = int(self.mask_rate * self.length)
            mask_index_list = np.random.randint(0, self.length - 1, size=num_to_predict)

        # use_action == True
        else:
            # get the index of purchases action
            purchases_index_list = list(self.puchases_df.index)
            num_to_predict = max(int(self.mask_rate * self.puchases_action_num), 2)
            mask_index_list = random.sample(purchases_index_list, num_to_predict)
            item_mask = [self.item_seq[i] for i in mask_index_list]
            mask_left_list = []
            for i in range(0, self.length):
                if i not in mask_index_list:
                    mask_left_list.append(i)

        self.mask_index_list = mask_index_list
        self.mask_left_list  = mask_left_list

    #根据bert 来进行mask
    def mask_process_bert(self):

        user_seq_temp = [self.user_seq[i] for i in self.mask_left_list]
        item_seq_temp = [self.item_seq[i] for i in self.mask_left_list]
        category_seq_temp = [self.category_seq[i] for i in self.mask_left_list]
        time_stamp_seq_temp = [self.time_stamp_seq[i] for i in self.mask_left_list]
        if self.use_action == True:
            action_seq_temp = [self.action_seq[i] for i in self.mask_left_list]


    #双向mask 加到尾部
    def mask_process_bidirectional_tail(self,index):

        user_seq_temp = copy.deepcopy(self.user_seq)
        user_seq_temp.pop(index)
        item_seq_temp = copy.deepcopy(self.item_seq)
        item_seq_temp.pop(index)
        category_seq_temp = copy.deepcopy(self.category_seq)
        category_seq_temp.pop(index)
        time_stamp_seq_temp = copy.deepcopy(self.time_stamp_seq)
        time_stamp_seq_temp.pop(index)

        if self.use_action == True:
            action_seq_temp = copy.deepcopy(self.action_seq)
            action_seq_temp.pop(index)

        # give the random value
        user_seq_temp.append(self.target_random_value)
        item_seq_temp.append(self.target_random_value)
        category_seq_temp.append(self.target_random_value)
        time_stamp_seq_temp.append(self.target_random_value)
        factor_list = [category_seq_temp, time_stamp_seq_temp]
        if self.use_action == True:
            action_seq_temp.append(2+self.offset)
            factor_list = [category_seq_temp, time_stamp_seq_temp, action_seq_temp]

        return user_seq_temp[0], item_seq_temp, factor_list


    #双向mask 只从中间部分mask掉
    def mask_process_bidirectional_middle(self,index):

        user_seq_temp = copy.deepcopy(self.user_seq)
        item_seq_temp = copy.deepcopy(self.item_seq)
        item_seq_temp[index] = self.target_random_value
        category_seq_temp = copy.deepcopy(self.category_seq)
        category_seq_temp[index] = self.target_random_value
        time_stamp_seq_temp = copy.deepcopy(self.time_stamp_seq)
        if self.use_action == True:
            action_seq_temp = copy.deepcopy(self.action_seq)
            action_seq_temp[index] = 2 + self.offset


    #for different type sample
    #for random ,time or normal
    def mask_process_unidirectional(self,type,index,
                                    time_window = 24 * 3600 * 35,
                                    lengeth_limit = 50):

        # only sample the data before the label
        if type == "unidirection":
            temp_index = index

        elif type == "random":
            purchases_index = self.mask_index_list.index(index)
            pre_purchases_index = purchases_index - 1
            if pre_purchases_index < 0:
                start = 0
            else:
                start = self.mask_index_list[pre_purchases_index]
            # sample random
            temp_index = random.randint(start + 1, index)

        elif type == "time_window":
            target_time = self.time_stamp_seq[index]
            # print("target time is " + str(target_time))
            for i in range(0,index + 1):
                # print(self.time_stamp_seq[i])
                if target_time - self.time_stamp_seq[i] <= time_window:
                    temp_index = i
                    break

        #如果长度超标，进行截取
        if temp_index - lengeth_limit + 1 > 0:
            start = temp_index  - lengeth_limit +1
        else:
            start = 0


        user_seq_temp = [self.user_seq[i] for i in range(start, self.length) if i < temp_index]
        item_seq_temp = [self.item_seq[i] for i in range(start, self.length) if i < temp_index]
        category_seq_temp = [self.category_seq[i] for i in range(start, self.length) if i < temp_index]
        time_stamp_seq_temp = [self.time_stamp_seq[i] for i in range(start, self.length) if i < temp_index]
        if self.use_action == True:
            action_seq_temp = [self.action_seq[i] for i in range(start, self.length) if i < temp_index]
            
        # 最后填充，填充最后一位
        # give the random value
        # user_seq_temp.append(self.target_random_value)
        # item_seq_temp.append(self.target_random_value)
        # category_seq_temp.append(self.target_random_value)
        #time_stamp_seq_temp.append(self.target_random_value)
        factor_list = [category_seq_temp, time_stamp_seq_temp]
        if self.use_action == True:
            action_seq_temp.append(2)
            factor_list = [category_seq_temp, time_stamp_seq_temp, action_seq_temp]

        return user_seq_temp[0], item_seq_temp, factor_list

    #get the random neg item
    def get_neg_item(self,item_count,number = 1):
        # generate neg sample not in mask_list
        item_mask = [self.item_seq[i] for i in self.mask_index_list]
        neg_list = []
        while (True):
            neg_item_id = np.random.randint(self.offset, item_count)
            if neg_item_id not in neg_list and neg_item_id not in item_mask:
                neg_list.append(neg_item_id)

            if len(neg_list) == number:
                break

        return neg_list

    def make_bpr_data(self, behavior_seq, use_action=False):

        user_id = behavior_seq["user_id"].tolist()[0]

        if use_action == False:
            for index, behavior in behavior_seq.iterrows():
                self.data_set.append((user_id,
                                      (behavior['item_id'],
                                       np.random.randint(self.offset + 1, self.item_count + self.offset))))
        else:
            for index, behavior in behavior_seq.iterrows():
                if behavior['action_type'] == 2:
                    self.data_set.append((user_id,
                                          (behavior['item_id'],
                                           np.random.randint(self.offset + 1, self.item_count + self.offset))))


    def proc_time_emb(self, time_stamp_seq, mask_time,gap):
        interval_list = [np.abs(i - mask_time) for i in time_stamp_seq]
        interval_list = [np.sum(i >= gap) for i in interval_list]
        return interval_list

    # only pos embeding
    def proc_pos_emb(self, time_stamp_seq):
        interval_list = [i for i in range(0, len(time_stamp_seq))]
        return interval_list


    def pro_time_method(self, time_stamp_seq, mask_time):
        timelast_list = [time_stamp_seq[i+1]-time_stamp_seq[i] for i in range(0,len(time_stamp_seq)-1,1)]
        timelast_list.insert(0,0)
        timenow_list = [mask_time-time_stamp_seq[i] for i in range(0,len(time_stamp_seq),1)]

        return timelast_list,timenow_list








