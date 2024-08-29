# -*- coding: utf-8 -*-
import torch
from lib.utils.ik_utils import vectors2rotmat, batch_rodrigues, rotation_matrix_to_angle_axis, batch_get_neck_orient, batch_get_wrist_orient

def get_twist_rotmat_from_cossin(phis, child_vec_t):
 
    batch_size = phis.shape[0]
    dtype=phis.dtype
    device=phis.device
    
    child_rest_loc = child_vec_t.clone()
    child_rest_norm = torch.norm(child_rest_loc, dim=1, keepdim=True)
    # Convert spin to rot_mat
    # (B, 3, 1)
    spin_axis = child_rest_loc / child_rest_norm
    # (B, 1, 1)
    rx, ry, rz = torch.split(spin_axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
                .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)

    # (B, 1, 1)
    cos, sin = torch.split(phis, 1, dim=1)
    cos = torch.unsqueeze(cos, dim=2)
    sin = torch.unsqueeze(sin, dim=2)

    rot_mat_spin = ident + sin * K + (1-cos) * torch.bmm(K,K)

    return rot_mat_spin
  
def refine_arm_rotmat(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat, phis):
    
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    
    t_pos = t_pos - t_pos[:,0:1]
    p_pos = p_pos - p_pos[:,0:1]
    
    ## First
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    body_t_pos = body_t_pos.unsqueeze(-1)
    body_p_pos = body_p_pos.unsqueeze(-1)
    
    ## Second
    body_vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    body_vec_p = body_p_pos[:, 1:] - body_p_pos[:, body_parent[1:]]
 
    ## Third
    body_accumulate_rotmat = analyik_global_orient_rotmat[:,0].clone()
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, analyik_body_pose_rotmat[:,3-1])
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, analyik_body_pose_rotmat[:,6-1])
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, analyik_body_pose_rotmat[:,9-1]).unsqueeze(1)
    
    hand_accumulate_rotmat = body_accumulate_rotmat.clone().squeeze(1)
    
    ## Four
    body_vec_pt = body_vec_p.clone()
    body_vec_pt[:,11:] = torch.matmul(body_accumulate_rotmat.transpose(2,3), body_vec_pt[:,11:])
     

    #-----------------------------l-arm----------------------------------------
    # 13
    body_local_twist_rotmat_13 = get_twist_rotmat_from_cossin(phis[:,0], body_vec_t[:,15])
    body_local_swing_rotmat_13 = vectors2rotmat(body_vec_t[:,15], body_vec_pt[:,15], torch.float32)
    body_local_rotmat_13 = torch.matmul(body_local_swing_rotmat_13, body_local_twist_rotmat_13)
    body_accumulate_rotmat = body_local_rotmat_13.clone()
    lhand_accumulate_rotmat = torch.matmul(hand_accumulate_rotmat, body_local_rotmat_13) 
    
    # 16
    body_local_twist_rotmat_16 = get_twist_rotmat_from_cossin(phis[:,1], body_vec_t[:,17])
    body_vec_pt_1816 = torch.matmul(body_accumulate_rotmat.transpose(1,2), body_vec_pt[:,17])  
    body_local_swing_rotmat_16 = vectors2rotmat(body_vec_t[:,17], body_vec_pt_1816, torch.float32)
    body_local_rotmat_16 = torch.matmul(body_local_swing_rotmat_16, body_local_twist_rotmat_16)
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, body_local_rotmat_16)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, body_local_rotmat_16) 

    # 18
    body_local_twist_rotmat_18 = get_twist_rotmat_from_cossin(phis[:,2], body_vec_t[:,19])
    body_vec_pt_2018 = torch.matmul(body_accumulate_rotmat.transpose(1,2), body_vec_pt[:,19]) 
    body_local_swing_rotmat_18 = vectors2rotmat(body_vec_t[:,19], body_vec_pt_2018, torch.float32)
    body_local_rotmat_18 = torch.matmul(body_local_swing_rotmat_18, body_local_twist_rotmat_18)
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, body_local_rotmat_18)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, body_local_rotmat_18).unsqueeze(1)  

    #-----------------------------r-arm----------------------------------------
    # 14
    body_local_twist_rotmat_14 = get_twist_rotmat_from_cossin(phis[:,3], body_vec_t[:,16])
    body_local_swing_rotmat_14 = vectors2rotmat(body_vec_t[:,16], body_vec_pt[:,16], torch.float32)
    body_local_rotmat_14 = torch.matmul(body_local_swing_rotmat_14, body_local_twist_rotmat_14)
    body_accumulate_rotmat = body_local_rotmat_14.clone()
    rhand_accumulate_rotmat = torch.matmul(hand_accumulate_rotmat, body_local_rotmat_14) 

    # 17
    body_local_twist_rotmat_17 = get_twist_rotmat_from_cossin(phis[:,4], body_vec_t[:,18])
    body_vec_pt_1917 = torch.matmul(body_accumulate_rotmat.transpose(1,2), body_vec_pt[:,18])  
    body_local_swing_rotmat_17 = vectors2rotmat(body_vec_t[:,18], body_vec_pt_1917, torch.float32)
    body_local_rotmat_17 = torch.matmul(body_local_swing_rotmat_17, body_local_twist_rotmat_17)
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, body_local_rotmat_17)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, body_local_rotmat_17) 

    # 19
    body_local_twist_rotmat_19 = get_twist_rotmat_from_cossin(phis[:,5], body_vec_t[:,20])
    body_vec_pt_2119 = torch.matmul(body_accumulate_rotmat.transpose(1,2), body_vec_pt[:,20]) 
    body_local_swing_rotmat_19 = vectors2rotmat(body_vec_t[:,20], body_vec_pt_2119, torch.float32)
    body_local_rotmat_19 = torch.matmul(body_local_swing_rotmat_19, body_local_twist_rotmat_19)
    body_accumulate_rotmat = torch.matmul(body_accumulate_rotmat, body_local_rotmat_19)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, body_local_rotmat_19).unsqueeze(1)  

    body_rf_rotmat = analyik_body_pose_rotmat.clone()
    body_rf_rotmat[:,13-1] = body_local_rotmat_13
    body_rf_rotmat[:,16-1] = body_local_rotmat_16
    body_rf_rotmat[:,18-1] = body_local_rotmat_18
    
    body_rf_rotmat[:,14-1] = body_local_rotmat_14
    body_rf_rotmat[:,17-1] = body_local_rotmat_17
    body_rf_rotmat[:,19-1] = body_local_rotmat_19
    
    # -------------------------Lhand-IK-----------------------------------------
    lhand_index = torch.tensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70])
    lhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    lhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    lhand_t_pos = t_pos[:,lhand_index]
    lhand_p_pos = p_pos[:,lhand_index]

    lhand_t_pos = lhand_t_pos.unsqueeze(-1)    
    lhand_p_pos = lhand_p_pos.unsqueeze(-1)

    lhand_vec_t = lhand_t_pos[:, 1:] - lhand_t_pos[:, lhand_parent[1:]]   # [b,n,3,1]
    lhand_vec_p = lhand_p_pos[:, 1:] - lhand_p_pos[:, lhand_parent[1:]]
    
    
    lhand_vec_pt = torch.matmul(lhand_accumulate_rotmat.transpose(2,3), lhand_vec_p)
    
    ## lWrist-20
    lhand_wrist_rotmat = batch_get_wrist_orient(lhand_vec_pt, lhand_vec_t, lhand_parent, lhand_children, torch.float32).unsqueeze(1)
    lhand_vec_pt = torch.matmul(lhand_wrist_rotmat.transpose(2,3), lhand_vec_pt)
    
    ## -------------------------l-thumb----------------------------------------
    ### 37
    lhand_local_twist_rotmat_1 = get_twist_rotmat_from_cossin(phis[:,6], lhand_vec_t[:,1])
    lhand_local_swing_rotmat_1 = vectors2rotmat(lhand_vec_t[:,1], lhand_vec_pt[:,1], torch.float32)
    lhand_local_rotmat_1 = torch.matmul(lhand_local_swing_rotmat_1, lhand_local_twist_rotmat_1)
    lhand_accumulate_rotmat = lhand_local_rotmat_1.clone()
    
    ### 38
    lhand_local_twist_rotmat_2 = get_twist_rotmat_from_cossin(phis[:,7], lhand_vec_t[:,2])
    lhand_vec_pt_32 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,2])  
    lhand_local_swing_rotmat_2 = vectors2rotmat(lhand_vec_t[:,2], lhand_vec_pt_32, torch.float32)
    lhand_local_rotmat_2 = torch.matmul(lhand_local_swing_rotmat_2, lhand_local_twist_rotmat_2)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_2)
    
    ### 39
    lhand_local_twist_rotmat_3 = get_twist_rotmat_from_cossin(phis[:,8], lhand_vec_t[:,3])
    lhand_vec_pt_43 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,3])  
    lhand_local_swing_rotmat_3 = vectors2rotmat(lhand_vec_t[:,3], lhand_vec_pt_43, torch.float32)
    lhand_local_rotmat_3 = torch.matmul(lhand_local_swing_rotmat_3, lhand_local_twist_rotmat_3)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_3)
    
    
    ## -------------------------l-index----------------------------------------
    ### 25
    lhand_local_twist_rotmat_5 = get_twist_rotmat_from_cossin(phis[:,9], lhand_vec_t[:,5])
    lhand_local_swing_rotmat_5 = vectors2rotmat(lhand_vec_t[:,5], lhand_vec_pt[:,5], torch.float32)
    lhand_local_rotmat_5 = torch.matmul(lhand_local_swing_rotmat_5, lhand_local_twist_rotmat_5)
    lhand_accumulate_rotmat = lhand_local_rotmat_5.clone()
    
    ### 26
    lhand_local_twist_rotmat_6 = get_twist_rotmat_from_cossin(phis[:,10], lhand_vec_t[:,6])
    lhand_vec_pt_76 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,6])  
    lhand_local_swing_rotmat_6 = vectors2rotmat(lhand_vec_t[:,6], lhand_vec_pt_76, torch.float32)
    lhand_local_rotmat_6 = torch.matmul(lhand_local_swing_rotmat_6, lhand_local_twist_rotmat_6)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_6)
    
    ### 27
    lhand_local_twist_rotmat_7 = get_twist_rotmat_from_cossin(phis[:,11], lhand_vec_t[:,7])
    lhand_vec_pt_87 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,7])  
    lhand_local_swing_rotmat_7 = vectors2rotmat(lhand_vec_t[:,7], lhand_vec_pt_87, torch.float32)
    lhand_local_rotmat_7 = torch.matmul(lhand_local_swing_rotmat_7, lhand_local_twist_rotmat_7)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_7)
    
    
    ## -------------------------l-middle---------------------------------------
    ### 28
    lhand_local_twist_rotmat_9 = get_twist_rotmat_from_cossin(phis[:,12], lhand_vec_t[:,9])
    lhand_local_swing_rotmat_9 = vectors2rotmat(lhand_vec_t[:,9], lhand_vec_pt[:,9], torch.float32)
    lhand_local_rotmat_9 = torch.matmul(lhand_local_swing_rotmat_9, lhand_local_twist_rotmat_9)
    lhand_accumulate_rotmat = lhand_local_rotmat_9.clone()
    
    ### 29
    lhand_local_twist_rotmat_10 = get_twist_rotmat_from_cossin(phis[:,13], lhand_vec_t[:,10])
    lhand_vec_pt_1110 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,10])  
    lhand_local_swing_rotmat_10 = vectors2rotmat(lhand_vec_t[:,10], lhand_vec_pt_1110, torch.float32)
    lhand_local_rotmat_10 = torch.matmul(lhand_local_swing_rotmat_10, lhand_local_twist_rotmat_10)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_10)
    
    ### 30
    lhand_local_twist_rotmat_11 = get_twist_rotmat_from_cossin(phis[:,14], lhand_vec_t[:,11])
    lhand_vec_pt_1211 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,11])  
    lhand_local_swing_rotmat_11 = vectors2rotmat(lhand_vec_t[:,11], lhand_vec_pt_1211, torch.float32)
    lhand_local_rotmat_11 = torch.matmul(lhand_local_swing_rotmat_11, lhand_local_twist_rotmat_11)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_11)
    
    
    ## -------------------------l-ring-----------------------------------------
    ### 34
    lhand_local_twist_rotmat_13 = get_twist_rotmat_from_cossin(phis[:,15], lhand_vec_t[:,13])
    lhand_local_swing_rotmat_13 = vectors2rotmat(lhand_vec_t[:,13], lhand_vec_pt[:,13], torch.float32)
    lhand_local_rotmat_13 = torch.matmul(lhand_local_swing_rotmat_13, lhand_local_twist_rotmat_13)
    lhand_accumulate_rotmat = lhand_local_rotmat_13.clone()
    
    ### 35
    lhand_local_twist_rotmat_14 = get_twist_rotmat_from_cossin(phis[:,16], lhand_vec_t[:,14])
    lhand_vec_pt_1514 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,14])  
    lhand_local_swing_rotmat_14 = vectors2rotmat(lhand_vec_t[:,14], lhand_vec_pt_1514, torch.float32)
    lhand_local_rotmat_14 = torch.matmul(lhand_local_swing_rotmat_14, lhand_local_twist_rotmat_14)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_14)
    
    ### 36
    lhand_local_twist_rotmat_15 = get_twist_rotmat_from_cossin(phis[:,17], lhand_vec_t[:,15])
    lhand_vec_pt_1615 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,15])  
    lhand_local_swing_rotmat_15 = vectors2rotmat(lhand_vec_t[:,15], lhand_vec_pt_1615, torch.float32)
    lhand_local_rotmat_15 = torch.matmul(lhand_local_swing_rotmat_15, lhand_local_twist_rotmat_15)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_15)
    
    
    ## -------------------------l-prinky-----------------------------------------
    ### 31
    lhand_local_twist_rotmat_17 = get_twist_rotmat_from_cossin(phis[:,18], lhand_vec_t[:,17])
    lhand_local_swing_rotmat_17 = vectors2rotmat(lhand_vec_t[:,17], lhand_vec_pt[:,17], torch.float32)
    lhand_local_rotmat_17 = torch.matmul(lhand_local_swing_rotmat_17, lhand_local_twist_rotmat_17)
    lhand_accumulate_rotmat = lhand_local_rotmat_17.clone()
    
    ### 32
    lhand_local_twist_rotmat_18 = get_twist_rotmat_from_cossin(phis[:,19], lhand_vec_t[:,18])
    lhand_vec_pt_1918 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,18])  
    lhand_local_swing_rotmat_18 = vectors2rotmat(lhand_vec_t[:,18], lhand_vec_pt_1918, torch.float32)
    lhand_local_rotmat_18 = torch.matmul(lhand_local_swing_rotmat_18, lhand_local_twist_rotmat_18)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_18)
    
    ### 33
    lhand_local_twist_rotmat_19 = get_twist_rotmat_from_cossin(phis[:,20], lhand_vec_t[:,19])
    lhand_vec_pt_2019 = torch.matmul(lhand_accumulate_rotmat.transpose(1,2), lhand_vec_pt[:,19])  
    lhand_local_swing_rotmat_19 = vectors2rotmat(lhand_vec_t[:,19], lhand_vec_pt_2019, torch.float32)
    lhand_local_rotmat_19 = torch.matmul(lhand_local_swing_rotmat_19, lhand_local_twist_rotmat_19)
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, lhand_local_rotmat_19)
    
    
    # -------------------------Rhand-IK-----------------------------------------
    rhand_index = torch.tensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
    rhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    rhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    rhand_t_pos = t_pos[:,rhand_index]
    rhand_p_pos = p_pos[:,rhand_index]
    
    rhand_t_pos = rhand_t_pos.unsqueeze(-1)    
    rhand_p_pos = rhand_p_pos.unsqueeze(-1)
    
    rhand_vec_t = rhand_t_pos[:, 1:] - rhand_t_pos[:, rhand_parent[1:]]   # [b,n,3,1]
    rhand_vec_p = rhand_p_pos[:, 1:] - rhand_p_pos[:, rhand_parent[1:]]

    rhand_vec_pt = torch.matmul(rhand_accumulate_rotmat.transpose(2,3), rhand_vec_p)
    
    ## rWrist-21
    rhand_wrist_rotmat = batch_get_wrist_orient(rhand_vec_pt, rhand_vec_t, rhand_parent, rhand_children, torch.float32).unsqueeze(1)
    rhand_vec_pt = torch.matmul(rhand_wrist_rotmat.transpose(2,3), rhand_vec_pt)
    
    ## -------------------------r-thumb----------------------------------------
    ### 52
    rhand_local_twist_rotmat_1 = get_twist_rotmat_from_cossin(phis[:,21], rhand_vec_t[:,1])
    rhand_local_swing_rotmat_1 = vectors2rotmat(rhand_vec_t[:,1], rhand_vec_pt[:,1], torch.float32)
    rhand_local_rotmat_1 = torch.matmul(rhand_local_swing_rotmat_1, rhand_local_twist_rotmat_1)
    rhand_accumulate_rotmat = rhand_local_rotmat_1.clone()
    
    ### 53
    rhand_local_twist_rotmat_2 = get_twist_rotmat_from_cossin(phis[:,22], rhand_vec_t[:,2])
    rhand_vec_pt_32 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,2])  
    rhand_local_swing_rotmat_2 = vectors2rotmat(rhand_vec_t[:,2], rhand_vec_pt_32, torch.float32)
    rhand_local_rotmat_2 = torch.matmul(rhand_local_swing_rotmat_2, rhand_local_twist_rotmat_2)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_2)
    
    ### 54
    rhand_local_twist_rotmat_3 = get_twist_rotmat_from_cossin(phis[:,23], rhand_vec_t[:,3])
    rhand_vec_pt_43 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,3])  
    rhand_local_swing_rotmat_3 = vectors2rotmat(rhand_vec_t[:,3], rhand_vec_pt_43, torch.float32)
    rhand_local_rotmat_3 = torch.matmul(rhand_local_swing_rotmat_3, rhand_local_twist_rotmat_3)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_3)
    
    
    ## -------------------------r-index----------------------------------------
    ### 40
    rhand_local_twist_rotmat_5 = get_twist_rotmat_from_cossin(phis[:,24], rhand_vec_t[:,5])
    rhand_local_swing_rotmat_5 = vectors2rotmat(rhand_vec_t[:,5], rhand_vec_pt[:,5], torch.float32)
    rhand_local_rotmat_5 = torch.matmul(rhand_local_swing_rotmat_5, rhand_local_twist_rotmat_5)
    rhand_accumulate_rotmat = rhand_local_rotmat_5.clone()
    
    ### 41
    rhand_local_twist_rotmat_6 = get_twist_rotmat_from_cossin(phis[:,25], rhand_vec_t[:,6])
    rhand_vec_pt_76 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,6])  
    rhand_local_swing_rotmat_6 = vectors2rotmat(rhand_vec_t[:,6], rhand_vec_pt_76, torch.float32)
    rhand_local_rotmat_6 = torch.matmul(rhand_local_swing_rotmat_6, rhand_local_twist_rotmat_6)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_6)
    
    ### 42
    rhand_local_twist_rotmat_7 = get_twist_rotmat_from_cossin(phis[:,26], rhand_vec_t[:,7])
    rhand_vec_pt_87 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,7])  
    rhand_local_swing_rotmat_7 = vectors2rotmat(rhand_vec_t[:,7], rhand_vec_pt_87, torch.float32)
    rhand_local_rotmat_7 = torch.matmul(rhand_local_swing_rotmat_7, rhand_local_twist_rotmat_7)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_7)
    
    
    ## -------------------------r-middle---------------------------------------
    ### 43
    rhand_local_twist_rotmat_9 = get_twist_rotmat_from_cossin(phis[:,27], rhand_vec_t[:,9])
    rhand_local_swing_rotmat_9 = vectors2rotmat(rhand_vec_t[:,9], rhand_vec_pt[:,9], torch.float32)
    rhand_local_rotmat_9 = torch.matmul(rhand_local_swing_rotmat_9, rhand_local_twist_rotmat_9)
    rhand_accumulate_rotmat = rhand_local_rotmat_9.clone()
    
    ### 44
    rhand_local_twist_rotmat_10 = get_twist_rotmat_from_cossin(phis[:,28], rhand_vec_t[:,10])
    rhand_vec_pt_1110 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,10])  
    rhand_local_swing_rotmat_10 = vectors2rotmat(rhand_vec_t[:,10], rhand_vec_pt_1110, torch.float32)
    rhand_local_rotmat_10 = torch.matmul(rhand_local_swing_rotmat_10, rhand_local_twist_rotmat_10)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_10)
    
    ### 45
    rhand_local_twist_rotmat_11 = get_twist_rotmat_from_cossin(phis[:,29], rhand_vec_t[:,11])
    rhand_vec_pt_1211 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,11])  
    rhand_local_swing_rotmat_11 = vectors2rotmat(rhand_vec_t[:,11], rhand_vec_pt_1211, torch.float32)
    rhand_local_rotmat_11 = torch.matmul(rhand_local_swing_rotmat_11, rhand_local_twist_rotmat_11)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_11)
    
    
    ## -------------------------r-ring-----------------------------------------
    ### 49
    rhand_local_twist_rotmat_13 = get_twist_rotmat_from_cossin(phis[:,30], rhand_vec_t[:,13])
    rhand_local_swing_rotmat_13 = vectors2rotmat(rhand_vec_t[:,13], rhand_vec_pt[:,13], torch.float32)
    rhand_local_rotmat_13 = torch.matmul(rhand_local_swing_rotmat_13, rhand_local_twist_rotmat_13)
    rhand_accumulate_rotmat = rhand_local_rotmat_13.clone()
    
    ### 50
    rhand_local_twist_rotmat_14 = get_twist_rotmat_from_cossin(phis[:,31], rhand_vec_t[:,14])
    rhand_vec_pt_1514 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,14])  
    rhand_local_swing_rotmat_14 = vectors2rotmat(rhand_vec_t[:,14], rhand_vec_pt_1514, torch.float32)
    rhand_local_rotmat_14 = torch.matmul(rhand_local_swing_rotmat_14, rhand_local_twist_rotmat_14)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_14)
    
    ### 51
    rhand_local_twist_rotmat_15 = get_twist_rotmat_from_cossin(phis[:,32], rhand_vec_t[:,15])
    rhand_vec_pt_1615 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,15])  
    rhand_local_swing_rotmat_15 = vectors2rotmat(rhand_vec_t[:,15], rhand_vec_pt_1615, torch.float32)
    rhand_local_rotmat_15 = torch.matmul(rhand_local_swing_rotmat_15, rhand_local_twist_rotmat_15)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_15)
    
    
    ## -------------------------r-prinky-----------------------------------------
    ### 46
    rhand_local_twist_rotmat_17 = get_twist_rotmat_from_cossin(phis[:,33], rhand_vec_t[:,17])
    rhand_local_swing_rotmat_17 = vectors2rotmat(rhand_vec_t[:,17], rhand_vec_pt[:,17], torch.float32)
    rhand_local_rotmat_17 = torch.matmul(rhand_local_swing_rotmat_17, rhand_local_twist_rotmat_17)
    rhand_accumulate_rotmat = rhand_local_rotmat_17.clone()
    
    ### 47
    rhand_local_twist_rotmat_18 = get_twist_rotmat_from_cossin(phis[:,34], rhand_vec_t[:,18])
    rhand_vec_pt_1918 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,18])  
    rhand_local_swing_rotmat_18 = vectors2rotmat(rhand_vec_t[:,18], rhand_vec_pt_1918, torch.float32)
    rhand_local_rotmat_18 = torch.matmul(rhand_local_swing_rotmat_18, rhand_local_twist_rotmat_18)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_18)
    
    ### 48
    rhand_local_twist_rotmat_19 = get_twist_rotmat_from_cossin(phis[:,35], rhand_vec_t[:,19])
    rhand_vec_pt_2019 = torch.matmul(rhand_accumulate_rotmat.transpose(1,2), rhand_vec_pt[:,19])  
    rhand_local_swing_rotmat_19 = vectors2rotmat(rhand_vec_t[:,19], rhand_vec_pt_2019, torch.float32)
    rhand_local_rotmat_19 = torch.matmul(rhand_local_swing_rotmat_19, rhand_local_twist_rotmat_19)
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, rhand_local_rotmat_19)
    
    
    body_rf_rotmat[:,20-1] = lhand_wrist_rotmat.squeeze(1)
    body_rf_rotmat[:,21-1] = rhand_wrist_rotmat.squeeze(1)
    
    lhand_rf_rotmat = analyik_lhand_pose_rotmat.clone()
    lhand_rf_rotmat[:,0] = lhand_local_rotmat_5
    lhand_rf_rotmat[:,1] = lhand_local_rotmat_6
    lhand_rf_rotmat[:,2] = lhand_local_rotmat_7
    lhand_rf_rotmat[:,3] = lhand_local_rotmat_9
    lhand_rf_rotmat[:,4] = lhand_local_rotmat_10
    lhand_rf_rotmat[:,5] = lhand_local_rotmat_11
    lhand_rf_rotmat[:,6] = lhand_local_rotmat_17
    lhand_rf_rotmat[:,7] = lhand_local_rotmat_18
    lhand_rf_rotmat[:,8] = lhand_local_rotmat_19
    lhand_rf_rotmat[:,9] = lhand_local_rotmat_13
    lhand_rf_rotmat[:,10] = lhand_local_rotmat_14
    lhand_rf_rotmat[:,11] = lhand_local_rotmat_15
    lhand_rf_rotmat[:,12] = lhand_local_rotmat_1
    lhand_rf_rotmat[:,13] = lhand_local_rotmat_2
    lhand_rf_rotmat[:,14] = lhand_local_rotmat_3
    
    rhand_rf_rotmat = analyik_rhand_pose_rotmat.clone()
    rhand_rf_rotmat[:,0] = rhand_local_rotmat_5
    rhand_rf_rotmat[:,1] = rhand_local_rotmat_6
    rhand_rf_rotmat[:,2] = rhand_local_rotmat_7
    rhand_rf_rotmat[:,3] = rhand_local_rotmat_9
    rhand_rf_rotmat[:,4] = rhand_local_rotmat_10
    rhand_rf_rotmat[:,5] = rhand_local_rotmat_11
    rhand_rf_rotmat[:,6] = rhand_local_rotmat_17
    rhand_rf_rotmat[:,7] = rhand_local_rotmat_18
    rhand_rf_rotmat[:,8] = rhand_local_rotmat_19
    rhand_rf_rotmat[:,9] = rhand_local_rotmat_13
    rhand_rf_rotmat[:,10] = rhand_local_rotmat_14
    rhand_rf_rotmat[:,11] = rhand_local_rotmat_15
    rhand_rf_rotmat[:,12] = rhand_local_rotmat_1
    rhand_rf_rotmat[:,13] = rhand_local_rotmat_2
    rhand_rf_rotmat[:,14] = rhand_local_rotmat_3

    return  body_rf_rotmat, lhand_rf_rotmat, rhand_rf_rotmat

def refine_leg_rotmat(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, phis):
    
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    ## First
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    body_t_pos = body_t_pos - body_t_pos[:,0:1]
    body_t_pos = body_t_pos.unsqueeze(-1)
    body_p_pos = body_p_pos - body_p_pos[:,0:1]
    body_p_pos = body_p_pos.unsqueeze(-1)
    
    ## Second
    vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    vec_p = body_p_pos[:, 1:] - body_p_pos[:, body_parent[1:]]
    
    ## Third
    root_rotmat = analyik_global_orient_rotmat.clone()
    
    ## Four
    vec_pt = vec_p.clone()
    vec_pt = torch.matmul(root_rotmat.transpose(2,3), vec_pt)
    
    ## Five
    swing_rotmat_list = []
 
    # 1
    local_twist_rotmat_1 = get_twist_rotmat_from_cossin(phis[:,0], vec_t[:,3])
    local_swing_rotmat_1 = vectors2rotmat(vec_t[:,3], vec_pt[:,3], torch.float32)
    local_rotmat_1 = torch.matmul(local_swing_rotmat_1, local_twist_rotmat_1)
    accumulate_rotmat = local_rotmat_1.clone()

    swing_rotmat_list.append(local_swing_rotmat_1)

    # 4
    local_twist_rotmat_4 = get_twist_rotmat_from_cossin(phis[:,1], vec_t[:,6])
    vec_pt_74 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,6])  
    local_swing_rotmat_4 = vectors2rotmat(vec_t[:,6], vec_pt_74, torch.float32)
    local_rotmat_4 = torch.matmul(local_swing_rotmat_4, local_twist_rotmat_4)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_4)

    swing_rotmat_list.append(local_swing_rotmat_4)

    # 7
    local_twist_rotmat_7 = get_twist_rotmat_from_cossin(phis[:,2], vec_t[:,9])
    vec_pt_107 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,9]) 
    local_swing_rotmat_7 = vectors2rotmat(vec_t[:,9], vec_pt_107, torch.float32)
    local_rotmat_7 = torch.matmul(local_swing_rotmat_7, local_twist_rotmat_7)

    swing_rotmat_list.append(local_swing_rotmat_7)
    
    # 2
    local_twist_rotmat_2 = get_twist_rotmat_from_cossin(phis[:,3], vec_t[:,4])
    local_swing_rotmat_2 = vectors2rotmat(vec_t[:,4], vec_pt[:,4], torch.float32)
    local_rotmat_2 = torch.matmul(local_swing_rotmat_2, local_twist_rotmat_2)
    accumulate_rotmat = local_rotmat_2.clone()

    swing_rotmat_list.append(local_swing_rotmat_2)
    
    # 5
    local_twist_rotmat_5 = get_twist_rotmat_from_cossin(phis[:,4], vec_t[:,7])
    vec_pt_85 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,7])  
    local_swing_rotmat_5 = vectors2rotmat(vec_t[:,7], vec_pt_85, torch.float32)
    local_rotmat_5 = torch.matmul(local_swing_rotmat_5, local_twist_rotmat_5)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_5)

    swing_rotmat_list.append(local_swing_rotmat_5)

    # 8
    local_twist_rotmat_8 = get_twist_rotmat_from_cossin(phis[:,5], vec_t[:,10])
    vec_pt_118 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,10]) 
    local_swing_rotmat_8 = vectors2rotmat(vec_t[:,10], vec_pt_118, torch.float32)
    local_rotmat_8 = torch.matmul(local_swing_rotmat_8, local_twist_rotmat_8)
    
    swing_rotmat_list.append(local_swing_rotmat_8)
        
    rf_rotmat = analyik_body_pose_rotmat.clone()

    rf_rotmat[:,1-1] = local_rotmat_1
    rf_rotmat[:,4-1] = local_rotmat_4
    rf_rotmat[:,7-1] = local_rotmat_7
    rf_rotmat[:,2-1] = local_rotmat_2
    rf_rotmat[:,5-1] = local_rotmat_5
    rf_rotmat[:,8-1] = local_rotmat_8
    
    
    return rf_rotmat

def refine_spine_rotmat(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, phis):
    
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    
    ## First
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    body_t_pos = body_t_pos - body_t_pos[:,0:1]
    body_t_pos = body_t_pos.unsqueeze(-1)
    body_p_pos = body_p_pos - body_p_pos[:,0:1]
    body_p_pos = body_p_pos.unsqueeze(-1)
    
    ## Second
    vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    vec_p = body_p_pos[:, 1:] - body_p_pos[:, body_parent[1:]]
    
    ## Third
    root_rotmat = analyik_global_orient_rotmat.clone()
    
    ## Four
    vec_pt = vec_p.clone()
    vec_pt = torch.matmul(root_rotmat.transpose(2,3), vec_pt)
    
    ## Five
    swing_rotmat_list = []
    
    # 3
    local_twist_rotmat_3 = get_twist_rotmat_from_cossin(phis[:,0], vec_t[:,5])
    local_swing_rotmat_3 = vectors2rotmat(vec_t[:,5], vec_pt[:,5], torch.float32)
    local_rotmat_3 = torch.matmul(local_swing_rotmat_3, local_twist_rotmat_3)
    accumulate_rotmat = local_rotmat_3.clone()

    swing_rotmat_list.append(local_swing_rotmat_3)    

    # 6
    local_twist_rotmat_6 = get_twist_rotmat_from_cossin(phis[:,1], vec_t[:,8])
    vec_pt_96 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,8])  
    local_swing_rotmat_6 = vectors2rotmat(vec_t[:,8], vec_pt_96, torch.float32)
    local_rotmat_6 = torch.matmul(local_swing_rotmat_6, local_twist_rotmat_6)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_6)
  
    swing_rotmat_list.append(local_swing_rotmat_6)

    # 9
    vec_qt = vec_pt.clone()
    vec_qt[:,8] = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,8])
    vec_qt[:,11:] = torch.matmul(accumulate_rotmat.unsqueeze(1).transpose(2,3), vec_pt[:,11:])
    spine3_rotmat = batch_get_neck_orient(vec_qt, vec_t, body_parent[1:22], body_children, torch.float32)
   
    accumulate_rotmat = torch.matmul(accumulate_rotmat, spine3_rotmat)
    
    # 12
    local_twist_rotmat_12 = get_twist_rotmat_from_cossin(phis[:,2], vec_t[:,14])
    vec_pt_1512 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,14]) 
    local_swing_rotmat_12 = vectors2rotmat(vec_t[:,14], vec_pt_1512, torch.float32)
    local_rotmat_12 = torch.matmul(local_swing_rotmat_12, local_twist_rotmat_12)

    swing_rotmat_list.append(local_swing_rotmat_12)
 
    rf_rotmat = analyik_body_pose_rotmat.clone()
    rf_rotmat[:,3-1] = local_rotmat_3
    rf_rotmat[:,6-1] = local_rotmat_6
    rf_rotmat[:,9-1] = spine3_rotmat
    rf_rotmat[:,12-1] = local_rotmat_12
    

    return rf_rotmat

def get_body_part_func(body_part):
    if body_part == 'ARM':
        return refine_arm_rotmat
    elif body_part == 'LEG':
        return refine_leg_rotmat
    elif body_part == 'SPINE':
        return refine_spine_rotmat
    else:
        assert False, 'body part invalid'
    