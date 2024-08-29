from lib.dataset.smpl.dataset_3d import Dataset3D

class AGORA(Dataset3D):
    def __init__(self, set, seqlen, overlap=0.75):
        db_name = 'agora'
        
        # during testing we don't need data augmentation
        print('AGORA Dataset overlap ratio:', overlap)
        super(AGORA, self).__init__(
            set=set,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            )
        print(f'{db_name} - number of dataset objects {self.__len__()}')
        

