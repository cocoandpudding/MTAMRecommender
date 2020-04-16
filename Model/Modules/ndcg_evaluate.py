import math
import pandas as pd

class NDCG(object):
    """
    normalized discount cumulative gain
    """

    def __init__(self, topk=20):
        self.topk = topk

    def error(self, comp_matrix, pred_matrix):
        """
        here pred_matrix represnt the predict rating or ranking
        """
        self.comp_matrix = comp_matrix.copy()
        self.pred_matrix = pred_matrix.copy()
        self.usernum, self.itemnum = self.comp_matrix.shape
        pred_usernum, pred_itemnum = self.pred_matrix.shape

        assert self.usernum == pred_usernum
        assert self.itemnum == pred_itemnum

        if self.pred_matrix.dtypes[0] == float:
            # predict matrix is rating
            ndcg_k = self.rating_ndcg(self.comp_matrix, self.pred_matrix)
        else:
            # predict matrix is ranking
            ndcg_k = self.ranking_ndcg(self.comp_matrix, self.pred_matrix)

        return ndcg_k


    def rating_ndcg(self, comp_matrix, pred_matrix):
        """
        change the ranking prediction to ranking in order to calculate the ndcg-k
        """
        for user in range(self.usernum):
            pred_matrix.ix[user] = pred_matrix.ix[user].order(ascending=False).index

        ndcg_k = self.ranking_ndcg(comp_matrix, pred_matrix)
        return ndcg_k


    def ranking_ndcg(self, comp_matrix, pred_matrix):
        """
        formula ndcg-k calculation of ranking
        """
        ndcg_vector = []
        for user in range(self.usernum):
            pred_item_index = pred_matrix.ix[user]
            comp_item_index = comp_matrix.ix[user].order(ascending=False).index

            ndcg_ki = 0.
            idcg_ki = 0.
            for ki in range(1, self.topk):
                log_2_i = math.log(ki, 2) if math.log(ki, 2) != 0. else 1.
                ndcg_ki += comp_matrix[pred_item_index[ki]][user] / log_2_i
                idcg_ki += comp_matrix[comp_item_index[ki]][user] / log_2_i

            ndcg_ki = ndcg_ki / idcg_ki
            ndcg_vector.append(ndcg_ki)



        ndcg_k = pd.Series(ndcg_vector).sum() / self.usernum
        return ndcg_k




if __name__ == "__main__":
    main()