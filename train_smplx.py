# -*- coding: utf-8 -*-
import os
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.smplx.config import SMPLX_MODEL_PATH,  SMPLX_KID_MODEL_PATH

from lib.core.smplx.loss import SMPLXIKSLoss
from lib.core.smplx.trainer import Trainer
from lib.core.smplx.config import parse_args
from lib.utils.utils import prepare_output_dir
from lib.models.smplx.body_models import SMPLXLayer
from lib.models.smplx.model import SMPLX_HybrIK
from lib.dataset.smplx.loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)
    
    logger = create_logger(cfg.LOGDIR, phase='train')
    
    #logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    #logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)
    
    # ======== SMPL ======== #
    SMPLX = SMPLXLayer(
         model_path=SMPLX_MODEL_PATH,
         num_betas=10,
         use_pca=False,
         age='adult',
         kid_template_path=SMPLX_KID_MODEL_PATH,
    ).to(cfg.DEVICE)
    
    # ======== Dataloaders ======== #
    data_loaders = get_data_loaders(cfg)
    
    # ======== Compile Loss ======== #
    loss = SMPLXIKSLoss(
        smplx=SMPLX,
        e_phi_loss_weight=cfg.LOSS.PHI_W,
        e_theta_loss_weight=cfg.LOSS.THETA_W,
        e_vert_loss_weight=cfg.LOSS.VERT_W,
        )
    
    # ======== Initialize networks, optimizers and lr_schedulers ======== #
    generator = SMPLX_HybrIK(
        num_hidden=cfg.MODEL.MLP.EMBED_DIM,
        ).to(cfg.DEVICE)
    
    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
        )
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       gen_optimizer,
       mode='min',
       factor=0.1,
       patience=cfg.TRAIN.LR_PATIENCE,
       verbose=True,
       )


    # ======== Start Training ======== #
    Trainer(
        data_loaders=data_loaders,
        generator=generator,
        optimizer=gen_optimizer,
        criterion=loss,
        smplx=SMPLX,
      
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        lr_scheduler=lr_scheduler,
        device=cfg.DEVICE,
        writer=writer,
        logdir=cfg.LOGDIR,
        performance_type='min',
        ).fit()
    
if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)
    main(cfg)
    
