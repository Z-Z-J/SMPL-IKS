# -*- coding: utf-8 -*-
import torch
import logging
import os.path as osp
import joblib
from torch.utils.data import Dataset

from lib.core.smplx.config import SMPLX_DB_DIR
from lib.utils.img_utils import split_into_chunks

logger = logging.getLogger(__name__)

class Dataset3D(Dataset):
    def __init__(self, set, seqlen, overlap=0., dataset_name=None):

        self.set = set
        self.dataset_name = dataset_name
        self.seqlen = seqlen
        self.stride = int(seqlen * (1-overlap))
        self.db = self.load_db()
        self.vid_indices = split_into_chunks(self.db['vid_name'], self.seqlen, self.stride)
 
    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def load_db(self):
        db_file = osp.join(SMPLX_DB_DIR, f'{self.dataset_name}_{self.set}_db.pt')

        if osp.isfile(db_file):
            db = joblib.load(db_file)
        else:
            raise ValueError(f'{db_file} do not exists')

        print(f'Loaded {self.dataset_name} dataset from {db_file}')
        return db

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]
        # pose
        betas_tensor = self.db['betas'][start_index:end_index+1]
        global_orient_tensor = self.db['global_orient'][start_index:end_index+1]
        body_pose_tensor = self.db['body_pose'][start_index:end_index+1]
        lhand_pose_tensor = self.db['lhand_pose'][start_index:end_index+1]
        rhand_pose_tensor = self.db['rhand_pose'][start_index:end_index+1]
        twist_angle_tensor = self.db['twist_angle'][start_index:end_index+1]
        
        target = {
            'betas': torch.from_numpy(betas_tensor).float(),
            'global_orient': torch.from_numpy(global_orient_tensor).float(),
            'body_pose': torch.from_numpy(body_pose_tensor).float(),
            'lhand_pose': torch.from_numpy(lhand_pose_tensor).float(), 
            'rhand_pose': torch.from_numpy(rhand_pose_tensor).float(), 
            'twist_angle': torch.from_numpy(twist_angle_tensor).float(),
            
        }
        return target
