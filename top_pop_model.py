import math
import time

from DataHandle.get_origin_data_amazon_beauty import Get_amazon_data_movie_tv
from DataHandle.get_origin_data_amazon_elec import Get_amazon_data_elec
from DataHandle.get_origin_data_amazon_music import Get_amazon_data_music
from DataHandle.get_origin_data_movielen import Get_movie_data
from DataHandle.get_origin_data_taobao import Get_taobaoapp_data
from DataHandle.get_origin_data_tmall import Get_tmall_data
from DataHandle.get_origin_data_yoochoose import Get_yoochoose_data
from Prepare.mask_data_process import mask_data_process
from Prepare.prepare_data_base import prepare_data_base
from config.model_parameter import model_parameter
from util.model_log import create_log
from DataHandle.get_origin_data import Get_origin_data
import sys
import copy
class top_pop_model():

    def __init__(self,test_data,prepare_data_base):

        self.test_data = test_data
        self.origin_data = prepare_data_base.origin_data
        self.origin_data = self.origin_data.groupby(by=["item_id"], as_index=False)["user_id"].count()
        self.origin_data = self.origin_data.sort_values(["user_id"],ascending = False)
        self.origin_data = self.origin_data["item_id"].tolist()


    def cal_top_pop(self, topk = 10):

        length = len(self.test_data)
        recall_count = 0
        ndcg_value_sum = 0
        error = 0

        #result_dic_count = sorted(self.origin_data, key=lambda item: item, reverse=True)
        result_dic_count = self.origin_data[:topk]
        for onedata in self.test_data:
            label = onedata[7][0]


            if label in result_dic_count:
                recall_count = recall_count + 1


            for i in range(len(result_dic_count)):
                if result_dic_count[i] == label:
                    ndcg_value = math.log(2) / math.log(i + 2)
                    ndcg_value_sum = ndcg_value_sum + ndcg_value
                    break

        print("Top pop the recall rate is: " +str(topk)+" "+ str(recall_count/length))
        print("Top pop the ndcg value is: " +str(topk)+" "+ str(ndcg_value_sum/length))
        print('error:'+str(error))

        return

    def cal_p_pop(self, topk = 10):

        length = len(self.test_data)
        recall_count = 0
        ndcg_value_sum = 0
        error = 0
        for onedata in self.test_data:

            result_dic_count = {}

            one_item_list = copy.deepcopy(onedata[1])
            if len(one_item_list)==0:
                error +=1
                print(onedata)
                continue
            one_item_list.pop()
            for index in one_item_list:
                # one_item_list[index]
                if index not in result_dic_count.keys():
                    result_dic_count[index] = 1
                else:
                    result_dic_count[index] = result_dic_count[index] + 1
            label = onedata[7][0]

            result_dic_count = sorted(result_dic_count.items(), key=lambda item: item[1],reverse=True)
            result_dic_count = [x[0] for x in result_dic_count][:topk]
            if label in result_dic_count:
                recall_count = recall_count + 1


            for i in range(len(result_dic_count)):
                if result_dic_count[i] == label:
                    ndcg_value = math.log(2) / math.log(i + 2)
                    ndcg_value_sum = ndcg_value_sum + ndcg_value
                    break

        print("P Pop the recall rate is: " +str(topk)+" "+ str(recall_count/length))
        print("P Pop the ndcg value is: " +str(topk)+" "+ str(ndcg_value_sum/length))
        print('error:'+str(error))

        return


if __name__ == "__main__":

    start_time = time.time()
    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS
    FLAGS.type = sys.argv[1]

    log_ins = create_log(type=FLAGS.type, experiment_type=FLAGS.experiment_type,
                         version=FLAGS.version)

    logger = log_ins.logger
    logger.info("hello world the experiment begin")

    # logger.info("The model parameter is :" + str(self.FLAGS._parse_flags()))

    if FLAGS.type == "yoochoose":
        get_origin_data_ins = Get_yoochoose_data(FLAGS=FLAGS)
        get_origin_data_ins.getDataStatistics()


    elif FLAGS.type == "movielen":
        get_origin_data_ins = Get_movie_data(FLAGS=FLAGS)

    elif FLAGS.type == "tmall":
        get_origin_data_ins = Get_tmall_data(FLAGS=FLAGS)


    elif FLAGS.type == "beauty":
        get_origin_data_ins = Get_amazon_data_movie_tv(FLAGS=FLAGS)
        get_origin_data_ins.getDataStatistics()
    elif FLAGS.type == "music":
        get_origin_data_ins =  Get_amazon_data_music(FLAGS=FLAGS)
        get_origin_data_ins.getDataStatistics()

    elif FLAGS.type == "elec":
        get_origin_data_ins = Get_amazon_data_elec(FLAGS=FLAGS)
        get_origin_data_ins.getDataStatistics()

    elif FLAGS.type == 'taobaoapp':
        get_origin_data_ins = Get_taobaoapp_data(FLAGS=FLAGS)
        get_origin_data_ins.getDataStatistics()

    # get_train_test_ins = Get_train_test(FLAGS=self.FLAGS,origin_data=get_origin_data_ins.origin_data)
    prepare_data_behavior_ins = prepare_data_base(FLAGS, get_origin_data_ins.origin_data)
    prepare_data_behavior_ins.map_process()
    train_set, test_set = prepare_data_behavior_ins.get_train_test()

    # fetch part of test_data
    # if len(self.test_set) > 10000:
    # self.test_set = random.sample(self.test_set,10000)
    # self.test_set = self.test_set.sample(3500)

    logger.info('DataHandle Process.\tCost time: %.2fs' % (time.time() - start_time))
    start_time = time.time()

    top_pop_ins = top_pop_model(train_set,prepare_data_behavior_ins)
    top_pop_ins.cal_p_pop(1)
    top_pop_ins.cal_p_pop(5)
    top_pop_ins.cal_p_pop(10)
    top_pop_ins.cal_p_pop(30)
    top_pop_ins.cal_p_pop(50)

    top_pop_ins.cal_top_pop(1)
    top_pop_ins.cal_top_pop(5)
    top_pop_ins.cal_top_pop(10)
    top_pop_ins.cal_top_pop(30)
    top_pop_ins.cal_top_pop(50)

