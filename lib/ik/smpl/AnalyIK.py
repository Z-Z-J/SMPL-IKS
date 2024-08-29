# -*- coding: utf-8 -*-

import torch
from lib.utils.ik_utils import batch_get_pelvis_orient, batch_get_neck_orient, vectors2rotmat, vectors2rotmat_bk, \
        batch_get_orient, get_twist_rotmat, rotation_matrix_to_angle_axis


def SMPL_AnalyIK_V3(t_pos, p_pos, parent, children):
    """
    Functions: Get SMPL pose(thetas) parameters.
    Arguments:
        t_pos: [b,24,3,1]
        p_pos: [b,24,3,1]
    """
    batch_size = t_pos.shape[0]
    device = t_pos.device
    
    t_pos = t_pos - t_pos[:,0:1]
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    p_pos = p_pos.unsqueeze(-1)
    
    ## 
    root_rotmat = batch_get_pelvis_orient(p_pos, t_pos, parent[1:24], children, torch.float32).unsqueeze(1)
    pt_pos = torch.matmul(root_rotmat.transpose(2,3), p_pos)

    ## 
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    vec_pt = pt_pos[:, 1:] - pt_pos[:, parent[1:]]
 

    ## left leg-------------------------------------------------------------------------
    ### 1
    local_rotmat_1 =  vectors2rotmat(vec_t[:,3], vec_pt[:,3], torch.float32)
    local_swing_rotmat_1, local_twist_rotmat_1 = get_twist_rotmat(local_rotmat_1, vec_t[:,3], torch.float32)
    accumulate_rotmat = local_swing_rotmat_1.clone()

    ### 4
    vec_74 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_pt[:,6])
    local_rotmat_4 = vectors2rotmat(vec_t[:,6], vec_74, torch.float32)
    local_swing_rotmat_4, local_twist_rotmat_4 = get_twist_rotmat(local_rotmat_4, vec_t[:,6], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_4)

    ### 7
    vec_107 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_pt[:,9])
    local_rotmat_7 = vectors2rotmat(vec_t[:,9], vec_107, torch.float32)
    local_swing_rotmat_7, local_twist_rotmat_7 = get_twist_rotmat(local_rotmat_7, vec_t[:,9], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_7)
    
    ## right leg-------------------------------------------------------------------------
    ### 2
    local_rotmat_2 =  vectors2rotmat(vec_t[:,4], vec_pt[:,4], torch.float32)
    local_swing_rotmat_2, local_twist_rotmat_2 = get_twist_rotmat(local_rotmat_2, vec_t[:,4], torch.float32)
    accumulate_rotmat = local_swing_rotmat_2.clone()

    ### 5
    vec_85 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_pt[:,7])
    local_rotmat_5 = vectors2rotmat(vec_t[:,7], vec_85, torch.float32)
    local_swing_rotmat_5, local_twist_rotmat_5 = get_twist_rotmat(local_rotmat_5, vec_t[:,7], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_5)

    ### 8
    vec_118 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_pt[:,10])
    local_rotmat_8 = vectors2rotmat(vec_t[:,10], vec_118, torch.float32)
    local_swing_rotmat_8, local_twist_rotmat_8 = get_twist_rotmat(local_rotmat_8, vec_t[:,10], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_8)
   
    
    ### spine------------------------------------------------------------------------
    ### 3
    local_rotmat_3 = vectors2rotmat(vec_t[:,5], vec_pt[:,5], torch.float32)
    local_swing_rotmat_3, local_twist_rotmat_3 = get_twist_rotmat(local_rotmat_3, vec_t[:,5], torch.float32)
    accumulate_rotmat = local_swing_rotmat_3.clone()

    ### 6
    vec_96 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_pt[:,8])
    local_rotmat_6 = vectors2rotmat(vec_t[:,8], vec_96, torch.float32)
    local_swing_rotmat_6, local_twist_rotmat_6 = get_twist_rotmat(local_rotmat_6, vec_t[:,8], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_6).unsqueeze(1)
    
    #------------------------------------------------------------------------------
    vec_qt = vec_pt.clone()
    vec_qt[:,8:9] = torch.matmul(accumulate_rotmat.transpose(2,3), vec_qt[:,8:9])
    vec_qt[:,11:] = torch.matmul(accumulate_rotmat.transpose(2,3), vec_qt[:,11:])
    spine3_rotmat = batch_get_neck_orient(vec_qt, vec_t, parent[1:24], children, torch.float32)
     
    ### 9
    local_rotmat_9 = spine3_rotmat
    accumulate_rotmat = torch.matmul(accumulate_rotmat.squeeze(1), local_rotmat_9).unsqueeze(1)

    #------------------------------------------------------------------------------
    vec_ppt = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,11:])
    vec_1512 = vec_ppt[:,3]
    vec_1613 = vec_ppt[:,4]
    vec_1714 = vec_ppt[:,5]
    vec_1816 = vec_ppt[:,6]
    vec_1917 = vec_ppt[:,7]
    vec_2018 = vec_ppt[:,8]
    vec_2119 = vec_ppt[:,9]
    vec_2220 = vec_ppt[:,10]
    vec_2321 = vec_ppt[:,11]

    
    ### 12
    local_rotmat_12 = vectors2rotmat(vec_t[:,14], vec_1512, torch.float32)
    local_swing_rotmat_12, local_twist_rotmat_12 = get_twist_rotmat(local_rotmat_12, vec_t[:,14], torch.float32)
  
    ### left arm-------------------------------------------------------------------------
    ### 13
    local_rotmat_13 = vectors2rotmat(vec_t[:,15], vec_1613, torch.float32)
    local_swing_rotmat_13, local_twist_rotmat_13 = get_twist_rotmat(local_rotmat_13, vec_t[:,15], torch.float32)
    accumulate_rotmat = local_swing_rotmat_13.clone()

    ### 16
    vec_1816 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_1816)
    local_rotmat_16 = vectors2rotmat(vec_t[:,17], vec_1816, torch.float32)
    local_swing_rotmat_16, local_twist_rotmat_16 = get_twist_rotmat(local_rotmat_16, vec_t[:,17], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_16)    
    
    ### 18
    vec_2018 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_2018)
    local_rotmat_18 = vectors2rotmat(vec_t[:,19], vec_2018, torch.float32)
    local_swing_rotmat_18, local_twist_rotmat_18 = get_twist_rotmat(local_rotmat_18, vec_t[:,19], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_18)    
        
    ### 20
    vec_2220 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_2220)
    local_rotmat_20 = vectors2rotmat(vec_t[:,21], vec_2220, torch.float32)
    local_swing_rotmat_20, local_twist_rotmat_20 = get_twist_rotmat(local_rotmat_20, vec_t[:,21], torch.float32)
    
    
    ### right arm-------------------------------------------------------------------------
    ### 14
    local_rotmat_14 = vectors2rotmat(vec_t[:,16], vec_1714, torch.float32)
    local_swing_rotmat_14, local_twist_rotmat_14 = get_twist_rotmat(local_rotmat_14, vec_t[:,16], torch.float32)
    accumulate_rotmat = local_swing_rotmat_14.clone()
    
    ### 17
    vec_1917 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_1917)
    local_rotmat_17 = vectors2rotmat(vec_t[:,18], vec_1917, torch.float32)
    local_swing_rotmat_17, local_twist_rotmat_17 = get_twist_rotmat(local_rotmat_17, vec_t[:,18], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_17)    

    ### 19
    vec_2119 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_2119)
    local_rotmat_19 = vectors2rotmat(vec_t[:,20], vec_2119, torch.float32)
    local_swing_rotmat_19, local_twist_rotmat_19 = get_twist_rotmat(local_rotmat_19, vec_t[:,20], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_19)   

    ### 21
    vec_2321 = torch.matmul(accumulate_rotmat.transpose(1, 2), vec_2321)
    local_rotmat_21 = vectors2rotmat(vec_t[:,22], vec_2321, torch.float32)
    local_swing_rotmat_21, local_twist_rotmat_21 = get_twist_rotmat(local_rotmat_21, vec_t[:,22], torch.float32)
    

    ## Local_rotmat
    iden = torch.eye(3).unsqueeze(0).to(device)
    local_rotmat = iden.unsqueeze(1).repeat(batch_size,24,1,1)
    local_rotmat[:,0:1] = root_rotmat
    
    local_rotmat[:,1] = local_swing_rotmat_1
    local_rotmat[:,4] = local_swing_rotmat_4
    local_rotmat[:,7] = local_swing_rotmat_7
    
    local_rotmat[:,2] = local_swing_rotmat_2
    local_rotmat[:,5] = local_swing_rotmat_5
    local_rotmat[:,8] = local_swing_rotmat_8
    
    local_rotmat[:,3] = local_swing_rotmat_3
    local_rotmat[:,6] = local_swing_rotmat_6
    local_rotmat[:,9] = local_rotmat_9
    
    local_rotmat[:,12] = local_swing_rotmat_12
 
    local_rotmat[:,13] = local_swing_rotmat_13
    local_rotmat[:,16] = local_swing_rotmat_16
    local_rotmat[:,18] = local_swing_rotmat_18
    local_rotmat[:,20] = local_swing_rotmat_20
   
    local_rotmat[:,14] = local_swing_rotmat_14
    local_rotmat[:,17] = local_swing_rotmat_17
    local_rotmat[:,19] = local_swing_rotmat_19
    local_rotmat[:,21] = local_swing_rotmat_21
 
    theta = rotation_matrix_to_angle_axis(local_rotmat.view(-1,3,3)).view(-1,24,3).contiguous()
   
    return theta