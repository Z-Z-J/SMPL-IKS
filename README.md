# SMPL-IKS: An Inverse Kinematic Solver for 3D Human Body Recovery

This repository is the official implementation of [SMPL-IKS: An Inverse Kinematic Solver for 3D Human Body Recovery]. 

## Requirements

To install requirements:

```setup
#1. Create a conda virtual environment.
conda create -n smpliks python=3.6.10
conda activate smpliks

#2. Install requirements.
pip install -r requirements.txt
```

## Preparing Data and Rre-trained model
1. Download train/test datasets and pretrained model.
   * Download the SMPL model

2. You need to follow directory structure of the `data` as below.
```
|-- data
`-- |-- smplik_db
    `-- |-- amass_train_db.pt
        `-- amss_test_db.pt
        `-- 3dpw_test_db.pt
        `-- agora_test_db.pt
`-- |-- smplik_data
    `-- |-- SMPL_NEUTRAL.pkl
        `-- smpl_kid_template.npy
        `-- skeleton_2_beta.npz
`-- |-- pretrained_model
    `-- |-- spine
        `-- |-- model_best.pth.tar
    `-- |-- leg
        `-- |-- model_best.pth.tar
    `-- |-- arm
        `-- |-- model_best.pth.tar
```
3. You need to modify the ROOT_PATH.
```setup
#1. cd SMPLIKS
#2. vim lib/core/config.py
#3. you should modify the ROOT_PATH = <YOU PATH>
```

## Training

To train our Part-aware Network in the paper, run this command:

```train
python train_pan.py --cfg configs/config_leg.yaml
python train_pan.py --cfg configs/config_arm.yaml
python train_pan.py --cfg configs/config_spine.yaml
```

## Evaluation

To evaluate our SI+APR+AnalyIK or SI+APR+HybrIK, run:

```eval
python eval_hybrik.py --cfg configs/config_eval.yaml
python eval_analyik.py --cfg config/config_eval.yaml
```

## Results

Our model achieves the following performance:

| Methods            |MPBE(AMASS)|MPJPE(AMASS)|MPVE(AMASS)|MPBE(3DPW)|MPJPE(3DPW)|MPVE(3DPW)|MPBE(AGORA)|MPJPE(AGORA)|MPVE(AGORA)|
| -------------------|-----------|------------|-----------|----------|-----------|----------|-----------|------------|-----------|
| SI+APR+AnalyIK     |   0.2mm   |     0.3mm  |    10.9mm |   0.0mm  |    0.2mm  |   14.2mm |    0.1mm  |     0.2mm  |   23.4mm  |            
| SI+APR+HybrIK      |   0.2mm   |     1.0mm  |    6.6mm  |   0.0mm  |    0.3mm  |   10.5mm |    0.1mm  |     0.7mm  |   19.2mm  |  

## License
By downloading and using this code you agree to the terms in the [LICENSE](LICENSE). Third-party datasets and software are subject to their respective licenses.

