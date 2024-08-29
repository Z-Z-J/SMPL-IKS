# -*- coding: utf-8 -*-
from lib.dataset.smplx.dataset_3d import Dataset3D

class MOTIONX(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75):
        db_name = 'motionx'
        
        # during testing we don't need data augmentation
        print('MOTIONX Dataset overlap ratio:', overlap)
        super(MOTIONX, self).__init__(
            set=set,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
