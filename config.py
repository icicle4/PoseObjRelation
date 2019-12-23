import time
import os


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name):
        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = False
        self.device_list = "0"  # id list of gpus used for training

        self.data_path = '/home/icicle/Documents/Datasets/hico_20160224_det'

        self.keypoint_num = 17

        self.train_random_seed = 0
        self.train_learning_rate = 1e-3  # initial learning rate
        self.lr_plan = {41: 5e-4, 81: 1e-4, 121: 5e-5}  # change learning rate in these epochs
        self.train_dropout_prob = 0.3  # dropout probability
        self.weight_decay = 0  # l2 weight decay

        self.max_epoch = 150  # max training epoch
        self.test_interval_epoch = 2

        self.hidden_dim = 1024

        self.test_before_train = False
        self.exp_note = 'PoseBallRelation'
        self.exp_name = None

        self.num_feature = 3
        self.n_filters = 1024
        self.n_blocks = 4
        self.n_classes = 2

        self.act = 'relu'
        self.norm = 'batch'
        self.bias = True
        self.epsilon = 0.2
        self.stochastic = True
        self.dropout = 0.3

    def init_config(self, need_new_folder=True):
        if self.exp_name is None:
            time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.exp_name = '[%s_stage]<%s>' % (self.exp_note, time_str)

        self.result_path = 'result/%s' % self.exp_name
        self.log_path = 'result/%s/log.txt' % self.exp_name

        if need_new_folder:
            os.mkdir(self.result_path)
