# -*- coding: utf-8 -*-
import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
        )

    def forward(self, x):
        out = self.block(x) + x
        return out


class MLPBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, p_dropout=0.2):
        super().__init__()

        self.pos_emd = nn.Sequential(
           nn.Linear(input_dim, hidden_dim-64),
           nn.BatchNorm1d(hidden_dim-64),
           nn.PReLU(),
           nn.Dropout(p_dropout),
        )
       
        self.beta_emd = nn.Sequential(
           nn.Linear(10, hidden_dim-192),
           nn.BatchNorm1d(hidden_dim-192),
           nn.PReLU(),
           nn.Dropout(p_dropout),
        )
       
        self.block = nn.Sequential(
           BasicBlock(hidden_dim, hidden_dim, hidden_dim),
           BasicBlock(hidden_dim, hidden_dim, hidden_dim),
           BasicBlock(hidden_dim, hidden_dim, hidden_dim),
           nn.Linear(hidden_dim, output_dim),
        )
       
    def forward(self, p3d, beta):
        x = self.pos_emd(p3d)
        y = self.beta_emd(beta)
        
        inp = torch.cat([x, y], dim=-1)
        
        out = self.block(inp)
        return out

class SMPLX_PART_RF(nn.Module):
    def __init__(
            self,
            num_hidden,
            jts_index,
            theta_index,
        ):
        super(SMPLX_PART_RF, self).__init__()
        
        self.num_jts = len(jts_index) # Arm: 12 Spine: 10 Leg: 9
        self.num_theta = len(theta_index) # Arm: 8 Spine: 3 Leg: 6
       
        self.num_hidden = num_hidden
        self.jts_index = jts_index
        self.theta_index = theta_index
      
        self.upose_upd_net = MLPBlock(self.num_jts*3, self.num_theta*2, self.num_hidden)
        
    def forward(self, p3d, beta):
        # p3d: [b,24,3] init_theta: [b,24,3]     
        b,_,_ = p3d.shape
     
        up_p3d = p3d[:,self.jts_index,:].contiguous().view(-1,self.num_jts*3)
        
        up_pred_phi = self.upose_upd_net(up_p3d, beta).view(-1,self.num_theta,2)         
        
        out = up_pred_phi
        return out


class SMPLX_HybrIK(nn.Module):
    def __init__(self, num_hidden):
        super(SMPLX_HybrIK, self).__init__()
        
        self.leg_jts_index = [0,1,2,3,4,5,7,8,10,11] 
        self.leg_twist_index = [0,3,6,1,4,7]
        
        self.spine_jts_index = [0,1,2,3,6,9,12,13,14,15]
        self.spine_twist_index = [2,5,11]
        
        self.arm_jts_index = [9,12,13,14,16,17,18,19, 20,37,25,28,34,31,
                                                      21,52,40,43,49,46]
        self.arm_twist_index = [12,15,17,13,16,18]
        
        self.lhand_jts_index = [20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70]
        self.lhand_twist_index = [36,37,38,24,25,26,27,28,29,33,34,35,30,31,32]
        
        self.rhand_jts_index = [21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]
        self.rhand_twist_index = [51,52,53,39,40,41,42,43,44,48,49,50,45,46,47]
            
        self.leg_rf_net = SMPLX_PART_RF(num_hidden, self.leg_jts_index, self.leg_twist_index)
        self.spine_rf_net = SMPLX_PART_RF(num_hidden, self.spine_jts_index, self.spine_twist_index)
        self.arm_rf_net = SMPLX_PART_RF(num_hidden, self.arm_jts_index, self.arm_twist_index)        
        self.lhand_rf_net = SMPLX_PART_RF(num_hidden, self.lhand_jts_index, self.lhand_twist_index)    
        self.rhand_rf_net = SMPLX_PART_RF(num_hidden, self.rhand_jts_index, self.rhand_twist_index)
    
    def forward(self, p3d, beta):
        
        leg_twist_phi = self.leg_rf_net(p3d, beta)
        spine_twist_phi = self.spine_rf_net(p3d, beta)
        arm_twist_phi = self.arm_rf_net(p3d, beta)
        lhand_twist_phi = self.lhand_rf_net(p3d, beta)
        rhand_twist_phi = self.rhand_rf_net(p3d, beta)
        
        arm_hand_twist_phi = torch.cat([arm_twist_phi, lhand_twist_phi, rhand_twist_phi], dim=1)
        
        return leg_twist_phi, spine_twist_phi, arm_hand_twist_phi
    

        
        
        
        
        




