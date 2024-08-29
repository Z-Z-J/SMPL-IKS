import os
import torch

from lib.core.smpl.config import SMPL_MODEL_PATH, SMPL_KID_MODEL_PATH, PRETRAINED_MODEL_PATH

from lib.core.smpl.evaluator import Evaluator
from lib.core.smpl.config import parse_args

from lib.models.smpl.smpl import SMPLLayer
from lib.models.smpl.model import SMPL_HybrIK
from lib.dataset.smpl.loaders import get_data_loaders

os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main(cfg):
    print('...Evaluating on test set...')
    
    # ======== SMPL ======== #
    SMPL = SMPLLayer(
         model_path=SMPL_MODEL_PATH,
         kid_template_path=SMPL_KID_MODEL_PATH,
         dtype=torch.float32,
         age='adult',
    ).to(cfg.DEVICE)
    
    # ======== Dataloaders ======== #
    _, test_loaders = get_data_loaders(cfg)
    
    # ======== Initialize networks, optimizers and lr_schedulers ======== #
    generator = SMPL_HybrIK(
        num_hidden=cfg.MODEL.MLP.EMBED_DIM,
        ).to(cfg.DEVICE)
    
    # ======== Load pretrained model ======== #
    checkpoint = torch.load(PRETRAINED_MODEL_PATH)
    generator.load_state_dict(checkpoint['gen_state_dict'])
    
    # ======== Start Evaluating ======== #
    Evaluator(
        test_loaders=test_loaders,
        generator=generator,
        smpl=SMPL,
        device=cfg.DEVICE,
        ).run()
    
if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    main(cfg)
    