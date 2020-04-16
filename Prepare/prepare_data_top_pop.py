from Prepare.prepare_data_base import prepare_data_base
import numpy as np
from Prepare.mask_data_process import mask_data_process
from config.model_parameter import model_parameter
from util.model_log import create_log
from DataHandle.get_origin_data import Get_origin_data

np.random.seed(1234)


class prepare_data_top_pop(prepare_data_base):

    def __init__(self, FLAGS, origin_data):

        #init prepare
        super(prepare_data_top_pop, self).__init__(FLAGS, origin_data)
        self.result_dic_label = {}
        self.origin_data = origin_data


    def get_top_pop_result(self,is_purchase = False):

        #give the result of top pop item of every user
        result_dic = {}
        def statistics_view(x):
            user_id = x["user_id"].tolist()[0]
            result = x.groupby(["item_id"],as_index=False)["user_id"].count()
            result.columns = ["item_id","count"]
            result= result.sort_values(by="count",ascending=False)
            result_dic[user_id] = result["item_id"].tolist()


        def statistics_purchase(x):
            x = x.loc[x["action_type"] == 2]
            user_id = x["user_id"].tolist()[0]
            result = x.groupby(["item_id"],as_index=False)["user_id"].count()
            result.columns = ["item_id","count"]
            result= result.sort_values(by="count",ascending=False)
            result_dic[user_id] = result["item_id"].tolist()


        if is_purchase == False:
            self.origin_data.groupby(["user_id"]).apply(lambda x: statistics_view(x))

        else:
            self.origin_data.groupby(["user_id"]).apply(lambda x: statistics_purchase(x))

        self.result_dic  = result_dic

    #only get index
    def data_handle_process(self,x):


        # 不同的数据处理方式
        behavior_seq = self.data_handle_process_base(x)
        if behavior_seq is None:
            return

        mask_data_process_ins = mask_data_process(behavior_seq=behavior_seq,
                                                  use_action=self.use_action,
                                                  mask_rate=self.mask_rate,
                                                  offset=self.offset)


        mask_data_process_ins.get_mask_index_list_behaivor()



    def format_train_test(self):

        self.train_set = []
        self.test_set = []
        for i in range(len(self.data_set)):
            if i % self.test_frac == 0:
                self.test_set.append(self.data_set[i])
            else:
                self.train_set.append(self.data_set[i])

if __name__ == "__main__":

    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    log_ins = create_log(type=FLAGS.type,
                         experiment_type = FLAGS.experiment_type,
                         version= FLAGS.version)

    logger = log_ins.logger
    logger.info("hello world the experiment begin")

    # logger.info("The model parameter is :" + str(self.FLAGS._parse_flags()))

    # init data and embeding
    get_origin_data_ins = Get_origin_data(type = FLAGS.type,
                                          raw_data_path= FLAGS.raw_data_path,
                                          raw_data_path_meta = FLAGS.raw_data_path_meta,
                                          logger= logger)

    # get_train_test_ins = Get_train_test(FLAGS=self.FLAGS,origin_data=get_origin_data_ins.origin_data)
    prepare_data_top_ins = prepare_data_top_pop(FLAGS, get_origin_data_ins.origin_data)
    prepare_data_top_ins.get_top_pop_result()




