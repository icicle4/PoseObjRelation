import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn.functional as F
from visdom import Visdom
from torch_geometric.data import DataLoader
from SKVecNet import DeepResGCN
from config import *
from PoseBallRelationDataset import return_dataset

from util_tools.util_tool import *
from loss import SmoothCrossEntropy


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


def train_net(cfg):
    """
    training gcn net
    """
    cfg.init_config()

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    training_set, validation_set = return_dataset(cfg)
    training_loader = DataLoader(training_set, batch_size=cfg.batch_size, num_workers=6)
    validation_loader = DataLoader(validation_set, batch_size=cfg.test_batch_size, num_workers=6)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = DeepResGCN(cfg)

    model = model.to(device=device)

    model.train()

    criteration = SmoothCrossEntropy(True, 0.2).to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train_learning_rate,
                           weight_decay=cfg.weight_decay)

    train = train_sk_vec
    test = test_sk_vec

    # Training iteration
    best_result = {'epoch': 0, 'acc': 0}
    start_epoch = 1

    plotter = VisdomLinePlotter(env_name='pose_ball_relation')

    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        # One epoch of forward and backward
        train_info = train(training_loader, model, criteration, device, optimizer, epoch, cfg)
        plotter.plot('loss', 'train', 'Loss', epoch, train_info['loss'])
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model, criteration, device, epoch, cfg)
            plotter.plot('loss', 'val', 'Loss', epoch, test_info['loss'])
            plotter.plot('acc', 'val', 'Accuracy', epoch, test_info['acc'])

            show_epoch_info('Test', cfg.log_path, test_info)

            if test_info['acc'] > best_result['acc']:
                best_result = test_info
            print_log(cfg.log_path,
                      'Best accuracy: %.2f%% at epoch #%d.' % (
                      best_result['acc'], best_result['epoch']))

            # Save model
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                # 'amp': amp.state_dict()
            }
            filepath = cfg.result_path + '/epoch%d_%.2f%%.pth' % (epoch, test_info['acc'])
            torch.save(state, filepath)
            print('model saved to:', filepath)


def train_sk_vec(data_loader, model, criteration, device, optimizer, epoch, cfg):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()

    for batch_data in data_loader:
        model.train()
        batch_data = batch_data.to(device)
        y = batch_data['y']

        optimizer.zero_grad()
        out = model(batch_data)
        loss = criteration(out, y)

        # Optim
        loss.backward()
        optimizer.step()

        # Predict actions
        labels = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(labels.int(), y.int()).float())

        # Get accuracy
        accuracy = correct.item() / out.shape[0]

        acc_meter.update(accuracy, out.shape[0])

        loss_meter.update(loss.item(), batch_data.num_graphs)

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acc': acc_meter.avg * 100
    }

    return train_info


def test_sk_vec(data_loader, model, criteration, device, epoch, cfg):
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    epoch_timer = Timer()

    for batch_data in data_loader:
        model.eval()
        batch_data = batch_data.to(device)
        y = batch_data['y']
        out = model(batch_data)

        loss = criteration(out, y)

        # Predict actions
        labels = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(labels.int(), y.int()).float())

        # Get accuracy
        accuracy = correct.item() / out.shape[0]

        acc_meter.update(accuracy, out.shape[0])
        loss_meter.update(loss.item(), batch_data.num_graphs)

    test_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        'loss': loss_meter.avg,
        'acc': acc_meter.avg * 100
    }
    return test_info


























