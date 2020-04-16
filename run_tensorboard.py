import os
import sys

#自动化打开tensor的脚本
#先关闭20个端口程序
for i in range(0,20):
    cmd = "fuser -k -n tcp "
    cmd = cmd + str(9020 + i)
    os.system(cmd)

#获取当前绝对路径
root_path = os.getcwd()
print(root_path)
tensorboard_path = os.path.join(root_path,"TensorBoard")
print(tensorboard_path)
list_file = os.listdir(tensorboard_path)
start_port = 9020
for one_file in list_file:
    #看看是否包含版本关键字
    if sys.argv[1] in one_file:
        print(one_file)
        file_path = os.path.join(tensorboard_path, one_file)

        # 如果是tar先解压
        if "tar" in one_file:
            target_dir = one_file.replace(".tar","")
            target_dir = target_dir.replace(".xz", "")
            target_dir = os.path.join(tensorboard_path, target_dir)
            if os.path.exists(target_dir)==False:
                os.makedirs(target_dir)
            cmd = "tar -xvJf " + file_path + " -C " + target_dir
            os.system(cmd)
            # file_path = os.path.join(tensorboard_path, "TomSun")
            # file_path = os.path.join(file_path, "ijcai2020")
            # file_path = os.path.join(file_path, "data")
            #
            #
            # "TomSun" "ijcai2020" "data"  tensorboard_result / movielen_rnn_base_Tom_second_base_2020 - 01 - 0
            # 9 - -15: 26:40
            file_path = target_dir

        cmd = "nohup tensorboard --logdir "
        #给出具体路径
        file_path = os.path.join(file_path, "tensorboard_train")
        cmd = cmd + file_path + " --port " + str(start_port) + " &"
        os.system(cmd)
        start_port =  start_port + 1

print("Done!!!!!!!!")