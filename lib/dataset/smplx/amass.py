# -*- coding: utf-8 -*-
from lib.dataset.smplx.dataset_3d import Dataset3D

class AMASS(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75):
        db_name = 'amass'
        
        # during testing we don't need data augmentation
        print('AMASS Dataset overlap ratio:', overlap)
        super(AMASS, self).__init__(
            set=set,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
