from torch.utils.data import ConcatDataset, DataLoader

from lib.dataset.smpl.amass import AMASS
from lib.dataset.smpl.threedpw import ThreeDPW
from lib.dataset.smpl.agora import AGORA

def get_data_loaders(cfg):
    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(set='train', seqlen=cfg.DATASET.SEQLEN, overlap=cfg.DATASET.OVERLAP)
            datasets.append(db)
        return ConcatDataset(datasets)
    
    # ==== 3D keypoint datasets ====
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)
    
    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        )
    
    # ==== Evaluation dataset ====
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(set='test', seqlen=cfg.DATASET.SEQLEN, overlap=cfg.DATASET.OVERLAP)
    
    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        )
    
    return train_3d_loader, valid_loader