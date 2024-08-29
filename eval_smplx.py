# -*- coding: utf-8 -*-
import os
import torch

from lib.core.smplx.evaluator import Evaluator
from lib.core.smplx.config import parse_args

from lib.models.smplx.body_models import SMPLXLayer
from lib.models.smplx.model import SMPLX_HybrIK
from lib.dataset.smplx.loaders import get_data_loaders

from lib.core.smplx.config import SMPLX_MODEL_PATH, SMPLX_KID_MODEL_PATH, PRETRAINED_MODEL_PATH

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(cfg):
    print('...Evaluating on test set...')
    
    # ======== SMPL ======== #
    SMPLX = SMPLXLayer(
         model_path=SMPLX_MODEL_PATH,
         num_betas=10,
         use_pca=False,
         age='adult',
         kid_template_path=SMPLX_KID_MODEL_PATH,
    ).to(cfg.DEVICE)
    
    # ======== Dataloaders ======== #
    _, test_loaders = get_data_loaders(cfg)
    
    # ======== networks ======== #
    generator = SMPLX_HybrIK(
        num_hidden=cfg.MODEL.MLP.EMBED_DIM,
        ).to(cfg.DEVICE)
    
    # ======== Load pretrained model ======== #
    checkpoint = torch.load(PRETRAINED_MODEL_PATH)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    
    # ======== Start Evaluating ======== #
    Evaluator(
        test_loaders=test_loaders,
        generator=generator,
        smplx=SMPLX,
        device=cfg.DEVICE,
        ).run()
    
if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    main(cfg)
    
