import os
from os import path as osp
import yaml
import time
import shutil
import logging
import torch

def move_dict_to_device(dict, device, tensor2float=False):
    for k,v in dict.items():
        if isinstance(v, torch.Tensor):
            if tensor2float:
                dict[k] = v.float().to(device)
            else:
                dict[k] = v.to(device)
                
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)
        
def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    os.makedirs(logdir, exist_ok=True)
    shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg

def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)

    log_file = osp.join(logdir, f'{phase}_log.txt')

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file,
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def get_optimizer(model, optim_type, lr, weight_decay, momentum):
    if optim_type in ['sgd', 'SGD']:
        opt = torch.optim.SGD(lr=lr, params=model.parameters(), momentum=momentum)
    elif optim_type in ['Adam', 'adam', 'ADAM']:
        opt = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    else:
        raise ModuleNotFoundError
    return opt


