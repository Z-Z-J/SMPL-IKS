# -*- coding: utf-8 -*-
import argparse
from yacs.config import CfgNode as CN

# CONSTANTS
# You may modify them at will
ROOT_PATH = ' '

SMPLX_DB_DIR = ROOT_PATH + 'IKS/data/smplx/smplxiks_db'

SMPLX_MODEL_PATH = ROOT_PATH +     'IKS/data/smplx/smplxiks_data/SMPLX_NEUTRAL.npz'
SMPLX_KID_MODEL_PATH = ROOT_PATH + 'IKS/data/smplx/smplxiks_data/smplx_kid_template.npy'

SMPLX_SI_DATA_PATH = ROOT_PATH + 'IKS/data/smplx/smplxiks_data/skeleton_2_beta_smplx.npz'

PRETRAINED_MODEL_PATH = ROOT_PATH + 'IKS/data/smplx/pretrained_model/model_best.pth.tar'


# Configuration variables
cfg = CN()

cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.LOGDIR = ''
cfg.DEVICE='cuda'
cfg.NUM_WORKERS = 8
cfg.SEED_VALUE = -1

cfg.CUDNN = CN()
cfg.CUDNN.BENCHMARK = False
cfg.CUDNN.DETERMINISTIC = True
cfg.CUDNN.ENABLED = True

cfg.TRAIN = CN()
cfg.TRAIN.DATASETS_3D = ['AMASS']
cfg.TRAIN.DATASET_EVAL = 'AMASS'
cfg.TRAIN.BATCH_SIZE = 1024
cfg.TRAIN.START_EPOCH = 0
cfg.TRAIN.END_EPOCH = 3
cfg.TRAIN.LR_PATIENCE = 2

cfg.TRAIN.PRETRAINED_REGRESSOR = ''

cfg.TRAIN.BODY_PART = ''
cfg.TRAIN.JTS_INDEX = []
cfg.TRAIN.TWIST_INDEX = []

# <====== generator optimizer
cfg.TRAIN.GEN_OPTIM = 'Adam'
cfg.TRAIN.GEN_LR = 1e-4
cfg.TRAIN.GEN_WD = 0.1
cfg.TRAIN.GEN_MOMENTUM = 0.9


cfg.DATASET = CN()
cfg.DATASET.SEQLEN = 1
cfg.DATASET.OVERLAP = 0.0

cfg.LOSS = CN()
cfg.LOSS.KP_W = 2.
cfg.LOSS.THETA_W = 2.
cfg.LOSS.VERT_W = 4.
cfg.LOSS.PHI_W = 1.


cfg.MODEL = CN()

cfg.MODEL.TEMPORAL_TYPE = 'mlp'

# MLP model hyperparams
cfg.MODEL.MLP = CN()
cfg.MODEL.MLP.EMBED_DIM = 1024
cfg.MODEL.MLP.STAGE_NUM = 3


def get_cfg_defaults():
   """Get a yacs CfgNode object with default values for my_project."""
   # Return a clone so that the defaults will not be altered
   # This is for the "local variable" use pattern
   return cfg.clone()


def update_cfg(cfg_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file


