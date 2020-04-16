from DataHandle.get_origin_data_base import Get_origin_data_base
import numpy as np
import pandas as pd
from config.model_parameter import model_parameter

np.random.seed(1234)


class Get_ml_data(Get_origin_data_base):

    def __init__(self, FLAGS):

        super(Get_ml_data, self).__init__(FLAGS = FLAGS)
        self.data_path = "data/orgin_data/movielens.csv"

        if FLAGS.init_origin_data == True:
            self.movie_data = pd.read_csv("data/raw_data/ml-1m/movies.dat",sep="::",header = None,names=['movieId','title','genres'],engine ='python')
            self.ratings_data = pd.read_csv("data/raw_data/ml-1m/ratings.dat",sep="::",header=None,names=['userId','movieId','rating','timestamp'],engine='python')
            self.get_movie_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)



    def get_movie_data(self):
        self.logger.info(self.ratings_data.shape)
        user_filter = self.ratings_data.groupby("userId").count()
        userfiltered = user_filter.sample(frac=0.8)
        self.ratings_data = self.ratings_data[self.ratings_data['userId'].isin(userfiltered.index)]
        self.logger.info(self.ratings_data.shape)

        #进行拼接，进行格式的规范化
        self.origin_data = pd.merge(self.ratings_data,self.movie_data,on="movieId")
        self.origin_data = self.origin_data[["userId","movieId","timestamp","genres"]]
        self.origin_data = self.origin_data.rename(columns={"userId": "user_id",
                                         "movieId": "item_id",
                                         "timestamp":"time_stamp",
                                         "genres":"cat_id",
                                         })


        self.filtered_data = self.filter(self.origin_data)
        self.filtered_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_data



if __name__ == "__main__":
    model_parameter_ins = model_parameter()
    experiment_name = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment_name).FLAGS

    ins = Get_ml_data(FLAGS=FLAGS)
    ins.getDataStatistics()










