import tensorflow as tf


class model_parameter:

    def __init__(self):
        # Network parameters
        self.flags = tf.flags
        self.flags.DEFINE_string('version', 'bpr', 'model version')
        self.flags.DEFINE_string('checkpoint_path_dir', 'data/check_point/bisIE_adam_blocks2_adam_dropout0.5_lr0.0001/','directory of save model')
        self.flags.DEFINE_integer('hidden_units', 128, 'Number of hidden units in each layer')
        self.flags.DEFINE_integer('num_blocks', 6, 'Number of blocks in each attention')
        self.flags.DEFINE_integer('num_heads', 8, 'Number of heads in each attention')
        self.flags.DEFINE_integer('num_units', 128, 'Number of units in each attention')

        self.flags.DEFINE_float('dropout', 0.5, 'Dropout probability(0.0: no dropout)')
        self.flags.DEFINE_float('regulation_rate', 0.00005, 'L2 regulation rate')
        self.flags.DEFINE_integer('itemid_embedding_size', 64, 'Item id embedding_beauty.csv size')
        self.flags.DEFINE_integer('cateid_embedding_size', 64, 'Cate id embedding_beauty.csv size')

        self.flags.DEFINE_boolean('concat_time_emb', True, 'Concat time-embedding_beauty.csv instead of Add')

        # 随机梯度下降sgd
        self.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop,sgd*)')
        self.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
        self.flags.DEFINE_float('decay_rate', 0.001, 'decay rate')
        # 最大梯度渐变到5
        self.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
        # 训练批次32
        self.flags.DEFINE_integer('train_batch_size', 256, 'Training Batch size')
        # 测试批次128
        self.flags.DEFINE_integer('test_batch_size', 100, 'Testing Batch size')
        # 最大迭代次数
        self.flags.DEFINE_integer('max_epochs', 200, 'Maximum # of training epochs')
        # 每100个批次的训练状态
        self.flags.DEFINE_integer('display_freq', 10, 'Display training status every this iteration')
        self.flags.DEFINE_integer('eval_freq', 200, 'Display training status every this iteration')
        self.flags.DEFINE_integer('max_len', 150, 'max len of attention')
        self.flags.DEFINE_integer('global_step', 100, 'global_step to summery AUC')

        # Runtime parameters
        self.flags.DEFINE_string('cuda_visible_devices', '2', 'Choice which GPU to use')
        self.flags.DEFINE_float('per_process_gpu_memory_fraction', 0.8,
                                'Gpu memory use fraction, 0.0 for allow_growth=True')
        # date process parameters
        self.flags.DEFINE_integer('gap_num', 6, 'sequence gap')
        self.flags.DEFINE_boolean('is_training', True, 'train of inference')
        self.flags.DEFINE_string('type', "yoochoose", 'raw date type')
        self.flags.DEFINE_string('experiment_type', "pistrec", 'experiment date type, e.g. istsbp, pistrec')
        self.flags.DEFINE_integer('length_of_user_history',50, 'the maximum length of user history')
        self.flags.DEFINE_integer('length_of_item_history', 50, 'the maximum length of item history')
        self.flags.DEFINE_integer('max_length_seq', 50, 'the length of the seq ')

        #parameters about origin_data
        self.flags.DEFINE_boolean('init_origin_data', False, 'whether to initialize the raw data')
        self.flags.DEFINE_boolean('init_train_data', False, 'whether to initialize the origin data')
        self.flags.DEFINE_integer('user_count_limit', 10000, "the limit of user")
        self.flags.DEFINE_string('causality', "unidirection", "the mask method")
        self.flags.DEFINE_string('pos_embedding', "time", "the method to embedding_beauty.csv pos")
        self.flags.DEFINE_integer('test_frac', 5, "train test radio")
        self.flags.DEFINE_float('mask_rate', 0.2, 'mask rate')
        self.flags.DEFINE_float('neg_sample_ratio', 20, 'negetive sample ratio')
        self.flags.DEFINE_boolean('remove_duplicate',True,'whether to remove duplicate entries')
        self.flags.DEFINE_string('experiment_data_type','item_based', 'item_based, dual')


        self.flags.DEFINE_string('fine_tune_load_path', None, 'the check point paht for the fine tune mode ')
        #parameters about model
        self.flags.DEFINE_string('load_type', "from_scratch", "the type of loading data")
        self.flags.DEFINE_boolean('draw_pic', False, "whether to draw picture")
        self.flags.DEFINE_integer('top_k', 20, "evaluate recall ndcg for k users")
        self.flags.DEFINE_string('experiment_name', "data_init", "the expeiment")


    def get_parameter(self,type):

        if type == "data_init":

            self.flags.FLAGS.type = "taobaoapp"
            self.flags.FLAGS.init_train_data  = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.user_count_limit = 80000
            self.flags.FLAGS.version = "tmall_init"
            self.flags.FLAGS.pos_embedding = "time"
            self.flags.FLAGS.test_frac = 100
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.remove_duplicate = False
            self.flags.FLAGS.gap_num = 15
            self.flags.FLAGS.length_of_user_history = 50


        if type == "statistics":

            self.flags.FLAGS.type = "beauty"
            self.flags.FLAGS.init_train_data  = True
            self.flags.FLAGS.init_origin_data = True
            self.flags.FLAGS.user_count_limit = 100000000
            self.flags.FLAGS.version = "beauty_statistics"
            self.flags.FLAGS.pos_embedding = "time"
            self.flags.FLAGS.test_frac = 100
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.gap_num = 15
            self.flags.FLAGS.length_of_item_history = 50
        if type == "Ti_Self_Attention_Modelb3_beauty":
            self.flags.FLAGS.type = "beauty"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 3
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "Ti_Self_Attention_Modelb3_beauty"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "Ti_Self_Attention_Model"
            self.flags.FLAGS.length_of_user_history = 50
        if type == "STAMP_beauty":
            self.flags.FLAGS.type = "beauty"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 6
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "STAMP_beauty"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "STAMP"
            self.flags.FLAGS.length_of_user_history = 50
        if type == "MTAM_via_rnnb6_beauty":
            self.flags.FLAGS.type = "beauty"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 6
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "MTAM_via_rnn6_beauty"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM"
            self.flags.FLAGS.length_of_user_history = 50

        if type == "Time_Aware_Self_Attention_Modelb3_yoochoose":
            self.flags.FLAGS.type = "yoochoose"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 3
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "Time_Aware_Self_Attention_Modelb3_yoochoose"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "Time_Aware_Self_Attention_Model"
            self.flags.FLAGS.length_of_user_history = 50

        if type == "MTAMb7_elec":
            self.flags.FLAGS.type = "elec"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 7
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "MTAMb7_elec"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM"
            self.flags.FLAGS.length_of_user_history = 50

        if type == "MTAMb8_elec":
            self.flags.FLAGS.type = "elec"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 8
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "MTAMb8_elec"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM"
            self.flags.FLAGS.length_of_user_history = 50
        if type == "MTAM_with_T_SeqRecb6_yoochoose":
            self.flags.FLAGS.type = "yoochoose"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 6
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "MTAM_with_T_SeqRecb6_yoochoose"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM_with_T_SeqRec"
            self.flags.FLAGS.length_of_user_history = 50
        if type == "MTAM_no_time_aware_attb7_music_256":
            self.flags.FLAGS.type = "music"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 7
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "MTAM_no_time_aware_attb7_music_256"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM_no_time_aware_att"
            self.flags.FLAGS.length_of_user_history = 50

        if type == "MTAM_with_T_SeqRecb7_music":
                self.flags.FLAGS.type = "music"
                self.flags.FLAGS.causality = "unidirection"
                self.flags.FLAGS.num_blocks = 7
                self.flags.FLAGS.num_heads = 1
                self.flags.FLAGS.learning_rate = 0.001
                self.flags.FLAGS.decay_rate = 0.995
                self.flags.FLAGS.regulation_rate = 0.00005
                self.flags.FLAGS.checkpoint_path_dir = None
                self.flags.FLAGS.user_count_limit = 1000000
                self.flags.FLAGS.init_train_data = False
                self.flags.FLAGS.init_origin_data = False
                self.flags.FLAGS.max_epochs = 200
                self.flags.FLAGS.load_type = "from_scratch"
                self.flags.FLAGS.train_batch_size = 256
                self.flags.FLAGS.test_batch_size = 2048
                self.flags.FLAGS.eval_freq = 500
                self.flags.FLAGS.version = "MTAM_with_T_SeqRecb7_music"
                self.flags.FLAGS.dropout = 0.5
                self.flags.FLAGS.cuda_visible_devices = "0"
                self.flags.FLAGS.experiment_type = "MTAM_with_T_SeqRec"
                self.flags.FLAGS.length_of_user_history = 50
        if type == "MTAM_via_rnnb7_music":
                self.flags.FLAGS.type = "music"
                self.flags.FLAGS.causality = "unidirection"
                self.flags.FLAGS.num_blocks = 7
                self.flags.FLAGS.num_heads = 1
                self.flags.FLAGS.learning_rate = 0.001
                self.flags.FLAGS.decay_rate = 0.995
                self.flags.FLAGS.regulation_rate = 0.00005
                self.flags.FLAGS.checkpoint_path_dir = None
                self.flags.FLAGS.user_count_limit = 1000000
                self.flags.FLAGS.init_train_data = False
                self.flags.FLAGS.init_origin_data = False
                self.flags.FLAGS.max_epochs = 200
                self.flags.FLAGS.load_type = "from_scratch"
                self.flags.FLAGS.train_batch_size = 256
                self.flags.FLAGS.test_batch_size = 1500
                self.flags.FLAGS.eval_freq = 500
                self.flags.FLAGS.version = "MTAM_via_rnnb7_music"
                self.flags.FLAGS.dropout = 0.5
                self.flags.FLAGS.cuda_visible_devices = "0"
                self.flags.FLAGS.experiment_type = "MTAM_via_rnn"
                self.flags.FLAGS.length_of_user_history = 50

        if type == "Time_Aware_Self_Attention_Modelb3_music":
            self.flags.FLAGS.type = "music"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 3
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "Time_Aware_Self_Attention_Modelb3_music"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "Time_Aware_Self_Attention_Model"
            self.flags.FLAGS.length_of_user_history = 50

        if type == "Time_Aware_Self_Attention_Modelb2_elec":
            self.flags.FLAGS.type = "elec"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 2
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995 
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "Time_Aware_Self_Attention_Modelb2_elec"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "Time_Aware_Self_Attention_Model"
            self.flags.FLAGS.length_of_user_history = 50
        if type == "Time_Aware_Self_Attention_Modelb1_elec":
            self.flags.FLAGS.type = "elec"
            self.flags.FLAGS.causality = "unidirection"
            self.flags.FLAGS.num_blocks = 1
            self.flags.FLAGS.num_heads = 1
            self.flags.FLAGS.learning_rate = 0.001
            self.flags.FLAGS.decay_rate = 0.995
            self.flags.FLAGS.regulation_rate = 0.00005
            self.flags.FLAGS.checkpoint_path_dir = None
            self.flags.FLAGS.user_count_limit = 1000000
            self.flags.FLAGS.init_train_data = False
            self.flags.FLAGS.init_origin_data = False
            self.flags.FLAGS.max_epochs = 200
            self.flags.FLAGS.load_type = "from_scratch"
            self.flags.FLAGS.train_batch_size = 256
            self.flags.FLAGS.test_batch_size = 2048
            self.flags.FLAGS.eval_freq = 500
            self.flags.FLAGS.version = "Time_Aware_Self_Attention_Modelb1_elec"
            self.flags.FLAGS.dropout = 0.5
            self.flags.FLAGS.cuda_visible_devices = "0"
            self.flags.FLAGS.experiment_type = "MTAM_with_T_SeqRec"
            self.flags.FLAGS.length_of_user_history = 50
        return  self.flags



