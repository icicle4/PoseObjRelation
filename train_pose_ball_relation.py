import sys
sys.path.append(".")
from train import *

cfg = Config('PoseBallRelation')

cfg.device_list = "0"
cfg.batch_size = 64
cfg.test_batch_size = 64
cfg.train_learning_rate = 0.0001
cfg.max_epoch = 100

cfg.data_path = '/home/icicle/Documents/Datasets/hico_20160224_det'

cfg.exp_note = 'pose_ball_relation_0'
train_net(cfg)

