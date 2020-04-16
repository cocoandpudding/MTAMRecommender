import os
import sys


#获取当前绝对路径
root_path = os.getcwd()
root_path = os.path.join(root_path,"data")
tensorboard_path = os.path.join(root_path,"tensorboard_result")
os.system("cd " + tensorboard_path)
compress_path = os.path.join(root_path,"tensorboard_compress")
list_file = os.listdir(tensorboard_path)

for one_file in list_file:
    print(one_file)
    #看看是否包含版本关键字
    if sys.argv[1] in one_file:

        cmd = "nohup tar -cvJf "
        #给出具体路径
        file_path  = os.path.join(tensorboard_path, one_file)
        compress_file_path = os.path.join(compress_path, one_file+".tar.xz ")
        cmd = cmd + compress_file_path + " -C " + file_path + " . &"
        os.system(cmd)

print("Done!!!!!!!!")