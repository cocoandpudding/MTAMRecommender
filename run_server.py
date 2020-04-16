import  numpy as np
import os
import random
data_type_list = ["movielen"]
cuda_visible_devices_list = ["0","1","2","3","4","6"]
cuda_dict = {}
for one_cuda in cuda_visible_devices_list:
    cuda_dict[one_cuda] = 1

#给出了每个占GPU的比例
memory_fraction = 1
experiment_type_list = ["T_SeqRec","NARM+",
                        "NARM++","LSTUR",
                        "MTAM","MTAM_no_time_aware_rnn"]

#不同的模型用不同的batch
train_batch_size_dic = {}
train_batch_size_dic["rnn_base"] = 4096
train_batch_size_dic["time_aware_rnn"] = 4096
train_batch_size_dic["time_aware_rnn_baseline"] = 4096
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att"] = 2048
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att_via_switch_network_soft"] = 1024
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att_via_switch_network_hard"] = 2048
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att_via_switch_network_add"] = 2048
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att_long_via_rnn"] = 2048
train_batch_size_dic["Time_Aware_Hybird_model_no_self_att_via_switch_network_soft_personalized"] = 2048

train_batch_size_dic["MTAM_hybird"] = 2048
train_batch_size_dic["T_SeqRec"]    = 2048
train_batch_size_dic["NARM"]        = 2048
train_batch_size_dic["NARM+"]        = 2048
train_batch_size_dic["LSTUR"]       = 2048
train_batch_size_dic["MTAM"]        = 2048
train_batch_size_dic["STAMP"]       = 2048
train_batch_size_dic["Gru4Rec"]       = 2048
train_batch_size_dic["STAMP"]       = 2048
train_batch_size_dic["MTAM"]       = 2048
train_batch_size_dic["SASrec"]       = 1024
train_batch_size_dic["NARM++"]        = 2048
train_batch_size_dic["MTAM_no_time_aware_rnn"] = 2048



version = "Tom_six"

for data_type in data_type_list:

    for experiment_type in experiment_type_list:
        cmd = "nohup python3 train_process.py"
        data_type_str =  " --type "  + str(data_type)
        experiment_type_str = " --experiment_type " + str(experiment_type)
        version_str = " --version " +  str(version)
        train_batch_size_str = " --train_batch_size " + str(train_batch_size_dic[experiment_type])


        #进行一个GPU的粗浅分配
        used_up = True
        while(True):

            if len(cuda_dict.keys()) == 0:
                break

            cuda = random.sample(cuda_dict.keys(), 1)[0]
            now_frac = cuda_dict[cuda] - memory_fraction

            #如果是小于0 或者等于0
            if now_frac < 0:
                del cuda_dict[cuda]
                continue

            if now_frac == 0:
                used_up = False
                del cuda_dict[cuda]
                break

            if now_frac > 0:
                used_up = False
                cuda_dict[cuda] = cuda_dict[cuda] - memory_fraction
                break

        if used_up == True:
            print("GPU used up")
            break


        #选出可见的GPU
        cuda_visible_devices_str = " --cuda_visible_devices " + cuda
        #给出每个GPU的比例
        per_process_gpu_memory_fraction_str = " --per_process_gpu_memory_fraction " + str(memory_fraction)
        cmd = cmd + data_type_str
        cmd = cmd + experiment_type_str
        cmd = cmd + version_str
        cmd = cmd + cuda_visible_devices_str
        cmd = cmd + per_process_gpu_memory_fraction_str
        cmd = cmd + train_batch_size_str
        cmd = cmd + " &"

        #去智能分配的GPU，保证GPU的资源
        os.system(cmd)
        print("data type: " + str(data_type) + " experiment_type: " + str(experiment_type) + " done!!")


