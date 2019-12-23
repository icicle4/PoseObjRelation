import time


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        print_log(cfg.log_path, k, ': ', v)
    print_log(cfg.log_path, '======================End=======================')


def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    if phase == 'Test':
        print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    else:
        print_log(log_path, '%s at epoch #%d' % (phase, info['epoch']))

    print_log(log_path, 'Accuracy: %.2f%%, Loss: %.5f, Using %.1f seconds' % (
        info['acc'], info['loss'], info['time']))


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time
