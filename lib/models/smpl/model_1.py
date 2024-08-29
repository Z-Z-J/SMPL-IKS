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

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Dropout(p_dropout),
            BasicBlock(hidden_dim, hidden_dim, hidden_dim),
            BasicBlock(hidden_dim, hidden_dim, hidden_dim),
            BasicBlock(hidden_dim, hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.block(x)
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
      
        self.upose_upd_net = MLPBlock(self.num_jts*3, self.num_theta*2 + 10, self.num_hidden)
        
    def forward(self, p3d):
        # p3d: [b,24,3] init_theta: [b,24,3]     
        b,_,_ = p3d.shape
     
        up_p3d = p3d[:,self.jts_index,:].contiguous().view(-1,self.num_jts*3)
   
        up_pred_beta_phi = self.upose_upd_net(up_p3d)
        
        up_pred_beta = up_pred_beta_phi[:,:10]
        up_pred_phi = up_pred_beta_phi[:,10:].contiguous().view(-1,self.num_theta,2)         
        
        return up_pred_beta, up_pred_phi


class SMPLX_HybrIK(nn.Module):
    def __init__(self, num_hidden):
        super(SMPLX_HybrIK, self).__init__()
        
        self.jts_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                           20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 
                           21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]
    
        self.twist_index = [0,3,6,1,4,7,
                            2,5,11,
                            12,15,17,13,16,18,
                            36,37,38,24,25,26,27,28,29,33,34,35,30,31,32,
                            51,52,53,39,40,41,42,43,44,48,49,50,45,46,47,
                           ]
        
        self.rf_net = SMPLX_PART_RF(num_hidden, self.jts_index, self.twist_index)
    
    def forward(self, p3d):
        
        beta, twist_phi = self.rf_net(p3d)
            
        leg_twist_phi = twist_phi[:,0:6]
        spine_twist_phi = twist_phi[:,6:9]
        arm_twist_phi = twist_phi[:,9:15]
        lhand_twist_phi = twist_phi[:,15:30]
        rhand_twist_phi = twist_phi[:,30:45]
        
        arm_hand_twist_phi = torch.cat([arm_twist_phi, lhand_twist_phi, rhand_twist_phi], dim=1)
        
        return beta, leg_twist_phi, spine_twist_phi, arm_hand_twist_phi
    


        
        
        
        




