# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

class SMPLXIKSLoss(nn.Module):
    def __init__(
            self,
            smplx,
            e_phi_loss_weight,
            e_theta_loss_weight,
            e_vert_loss_weight,
            device='cuda',
    ):
        super(SMPLXIKSLoss, self).__init__()
    
          
        self.e_phi_loss_weight = e_phi_loss_weight
        self.e_theta_loss_weight = e_theta_loss_weight
        self.e_vert_loss_weight = e_vert_loss_weight
        
        self.device = device
        
        self.criterion_smpl = nn.MSELoss().to(device)
        self.criterion_pos = mpjpe_cal
        self.criterion_vert = mpjpe_cal
    
    
    def forward(
            self,
            gt_p_pos,
            gt_verts,
            analyik_p_pos,
            analyik_verts,
            theory_p_pos,
            theory_verts,
            hybrik_p_pos,
            hybrik_verts,
            
            gt_leg_twist_phi,
            gt_spine_twist_phi,
            gt_arm_hand_twist_phi,
            pred_leg_twist_phi,
            pred_spine_twist_phi,
            pred_arm_hand_twist_phi, 
            
            gt_body_pose,
            gt_lhand_pose,
            gt_rhand_pose,
            hybrik_body_pose,
            hybrik_lhand_pose,
            hybrik_rhand_pose,
    ):
        
        # <======= Generator Loss
        loss_leg_twist_phi = self.criterion_smpl(pred_leg_twist_phi, gt_leg_twist_phi) 
        loss_spine_twist_phi = self.criterion_smpl(pred_spine_twist_phi, gt_spine_twist_phi)
        loss_arm_hand_twist_phi = self.criterion_smpl(pred_arm_hand_twist_phi, gt_arm_hand_twist_phi)
        
        loss_body_theta = self.criterion_smpl(hybrik_body_pose, gt_body_pose) 
        loss_lhand_theta = self.criterion_smpl(hybrik_lhand_pose, gt_lhand_pose)
        loss_rhand_theta = self.criterion_smpl(hybrik_lhand_pose, gt_rhand_pose)

        loss_analyik_vert = self.criterion_vert(analyik_verts, gt_verts)
        loss_analyik_p3d = self.criterion_pos(analyik_p_pos, gt_p_pos)
       
        loss_theory_vert = self.criterion_vert(theory_verts, gt_verts)
        loss_theory_p3d = self.criterion_pos(theory_p_pos, gt_p_pos)     
       
        loss_hybrik_vert = self.criterion_vert(hybrik_verts, gt_verts)
        loss_hybrik_p3d = self.criterion_pos(hybrik_p_pos, gt_p_pos)
       
        loss_phi = (loss_leg_twist_phi + loss_spine_twist_phi + loss_arm_hand_twist_phi) * self.e_phi_loss_weight
        loss_theta = (loss_body_theta + loss_lhand_theta + loss_rhand_theta) * self.e_theta_loss_weight
        loss_vert = loss_hybrik_vert * self.e_vert_loss_weight
        
        loss_dict = {
            'loss_phi': loss_phi,
            'loss_theta': loss_theta,
            'loss_a_p': loss_analyik_p3d,
            'loss_a_v': loss_analyik_vert,
            'loss_t_p': loss_theory_p3d,
            'loss_t_v': loss_theory_vert,
            'loss_h_p': loss_hybrik_p3d,
            'loss_h_v': loss_hybrik_vert,

            }    
        
        gen_loss = loss_phi + loss_theta + loss_vert
        
        return gen_loss, loss_dict
        
        
 
    