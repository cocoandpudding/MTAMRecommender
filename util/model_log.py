import logging.handlers
import threading
import time
#each experiment and data type has a text
class create_log():
    """create log type"""
    _instance_lock = threading.Lock()

    def __init__(self,type = None,experiment_type = None,version = None):

        if (type == None and experiment_type ==None and version ==None):
            pass
        else:
            self.logger = logging.getLogger("logger")

            timeArray = time.localtime(time.time())
            timeStr = time.strftime("%Y_%m_%d__%H_%M_%S", timeArray)
            self.filename = "data/log_data/"
            self.filename = self.filename + type + "_" + experiment_type + "_" + version + "_" + timeStr + "_log.txt"


            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(filename=self.filename)

            self.logger.setLevel(logging.DEBUG)
            handler1.setLevel(logging.INFO)
            handler2.setLevel(logging.INFO)

            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
            handler1.setFormatter(formatter)
            handler2.setFormatter(formatter)

            self.logger.addHandler(handler1)
            self.logger.addHandler(handler2)



    #singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(create_log, "_instance"):
            with create_log._instance_lock:
                if not hasattr(create_log, "_instance"):
                    create_log._instance = object.__new__(cls)
        return create_log._instance



    def get_logger(self):
        return self.logger

