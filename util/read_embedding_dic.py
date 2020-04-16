# coding=utf-8
import csv
import collections

def embedding_csv_dic(file_path):


    with open(file_path, encoding="utf-8") as csvfile:
        result_dic = collections.OrderedDict()
        csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
        birth_header = next(csv_reader)  # 读取第一行每一列的标题
        for row in csv_reader:  # 将csv 文件中的数据保存到birth_data中
            result_dic[row[0]] = [row[1],row[2],row[3],row[4]]

        return result_dic