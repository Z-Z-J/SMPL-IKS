# -*- coding: utf-8 -*-
import torch
from lib.utils.ik_utils import vectors2rotmat, batch_rodrigues, rotation_matrix_to_angle_axis, batch_get_neck_orient

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
  
def refine_arm_rotmat(t_pos, p_pos, analyik_thetas, phis):
    
    parent = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21]) 
 
    ## First
    t_pos = t_pos - t_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    p_pos = p_pos - p_pos[:,0:1]
    p_pos = p_pos.unsqueeze(-1)
    
    ## Second
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    vec_p = p_pos[:, 1:] - p_pos[:, parent[1:]]
        
    ## Third
    analyik_rotmat = batch_rodrigues(analyik_thetas.view(-1,3)).view(-1,24,3,3)
    accumulate_rotmat = analyik_rotmat[:,0]
    accumulate_rotmat = torch.matmul(accumulate_rotmat, analyik_rotmat[:,3])
    accumulate_rotmat = torch.matmul(accumulate_rotmat, analyik_rotmat[:,6])
    accumulate_rotmat = torch.matmul(accumulate_rotmat, analyik_rotmat[:,9]).unsqueeze(1)
    
    ## Four
    vec_pt = vec_p.clone()
    vec_pt[:,11:] = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,11:])
    
    
    ## Five
    swing_rotmat_list = []
        
    # 13
    local_twist_rotmat_13 = get_twist_rotmat_from_cossin(phis[:,0], vec_t[:,15])
    local_swing_rotmat_13 = vectors2rotmat(vec_t[:,15], vec_pt[:,15], torch.float32)
    local_rotmat_13 = torch.matmul(local_swing_rotmat_13, local_twist_rotmat_13)
    accumulate_rotmat = local_rotmat_13.clone()
    
    swing_rotmat_list.append(local_swing_rotmat_13)
    
    # 16
    local_twist_rotmat_16 = get_twist_rotmat_from_cossin(phis[:,1], vec_t[:,17])
    vec_pt_1816 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,17])  
    local_swing_rotmat_16 = vectors2rotmat(vec_t[:,17], vec_pt_1816, torch.float32)
    local_rotmat_16 = torch.matmul(local_swing_rotmat_16, local_twist_rotmat_16)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_16)

    swing_rotmat_list.append(local_swing_rotmat_16)

    # 18
    local_twist_rotmat_18 = get_twist_rotmat_from_cossin(phis[:,2], vec_t[:,19])
    vec_pt_2018 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,19]) 
    local_swing_rotmat_18 = vectors2rotmat(vec_t[:,19], vec_pt_2018, torch.float32)
    local_rotmat_18 = torch.matmul(local_swing_rotmat_18, local_twist_rotmat_18)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_18)

    swing_rotmat_list.append(local_swing_rotmat_18)    

    # 20
    local_twist_rotmat_20 = get_twist_rotmat_from_cossin(phis[:,3], vec_t[:,21])
    vec_pt_2220 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,21]) 
    local_swing_rotmat_20 = vectors2rotmat(vec_t[:,21], vec_pt_2220, torch.float32)
    local_rotmat_20 = torch.matmul(local_swing_rotmat_20, local_twist_rotmat_20)

    swing_rotmat_list.append(local_swing_rotmat_20)    

    # 14
    local_twist_rotmat_14 = get_twist_rotmat_from_cossin(phis[:,4], vec_t[:,16])
    local_swing_rotmat_14 = vectors2rotmat(vec_t[:,16], vec_pt[:,16], torch.float32)
    local_rotmat_14 = torch.matmul(local_swing_rotmat_14, local_twist_rotmat_14)
    accumulate_rotmat = local_rotmat_14.clone()

    swing_rotmat_list.append(local_swing_rotmat_14)

    # 17
    local_twist_rotmat_17 = get_twist_rotmat_from_cossin(phis[:,5], vec_t[:,18])
    vec_pt_1917 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,18])  
    local_swing_rotmat_17 = vectors2rotmat(vec_t[:,18], vec_pt_1917, torch.float32)
    local_rotmat_17 = torch.matmul(local_swing_rotmat_17, local_twist_rotmat_17)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_17)

    swing_rotmat_list.append(local_swing_rotmat_17)

    # 19
    local_twist_rotmat_19 = get_twist_rotmat_from_cossin(phis[:,6], vec_t[:,20])
    vec_pt_2119 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,20]) 
    local_swing_rotmat_19 = vectors2rotmat(vec_t[:,20], vec_pt_2119, torch.float32)
    local_rotmat_19 = torch.matmul(local_swing_rotmat_19, local_twist_rotmat_19)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_rotmat_19)

    swing_rotmat_list.append(local_swing_rotmat_19)

    # 21
    local_twist_rotmat_21 = get_twist_rotmat_from_cossin(phis[:,7], vec_t[:,22])
    vec_pt_2321 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,22]) 
    local_swing_rotmat_21 = vectors2rotmat(vec_t[:,22], vec_pt_2321, torch.float32)
    local_rotmat_21 = torch.matmul(local_swing_rotmat_21, local_twist_rotmat_21)

    swing_rotmat_list.append(local_swing_rotmat_21)
        
    rf_rotmat = analyik_rotmat.clone()
    rf_rotmat[:,13] = local_rotmat_13
    rf_rotmat[:,16] = local_rotmat_16
    rf_rotmat[:,18] = local_rotmat_18
    rf_rotmat[:,20] = local_rotmat_20
    rf_rotmat[:,14] = local_rotmat_14
    rf_rotmat[:,17] = local_rotmat_17
    rf_rotmat[:,19] = local_rotmat_19
    rf_rotmat[:,21] = local_rotmat_21
    
    rf_theta = rotation_matrix_to_angle_axis(rf_rotmat.view(-1,3,3)).view(-1,24,3).contiguous()
    swing_rotmat = torch.stack(swing_rotmat_list, dim=1)

    return rf_theta, swing_rotmat

def refine_leg_rotmat(t_pos, p_pos, analyik_thetas, phis):
    
    parent = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21]) 
    
    ## First
    t_pos = t_pos - t_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    p_pos = p_pos - p_pos[:,0:1]
    p_pos = p_pos.unsqueeze(-1)
    
    ## Second
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    vec_p = p_pos[:, 1:] - p_pos[:, parent[1:]]
    
    ## Third
    analyik_rotmat = batch_rodrigues(analyik_thetas.view(-1,3)).view(-1,24,3,3)
    root_rotmat = analyik_rotmat[:,0:1]
    
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
        
    rf_rotmat = analyik_rotmat.clone()

    rf_rotmat[:,1] = local_rotmat_1
    rf_rotmat[:,4] = local_rotmat_4
    rf_rotmat[:,7] = local_rotmat_7
    rf_rotmat[:,2] = local_rotmat_2
    rf_rotmat[:,5] = local_rotmat_5
    rf_rotmat[:,8] = local_rotmat_8
    
    rf_theta = rotation_matrix_to_angle_axis(rf_rotmat.view(-1,3,3)).view(-1,24,3).contiguous()
    swing_rotmat = torch.stack(swing_rotmat_list, dim=1)
    
    return rf_theta, swing_rotmat

def refine_spine_rotmat(t_pos, p_pos, analyik_thetas, phis):
    
    parent = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
            16, 17, 18, 19, 20, 21]) 
    
    children = torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 27, 28, 15, 16, 17, 24, 18, 19,
           20, 21, 22, 23, 25, 26, -1, -1, -1, -1, -1])
    
    ## First
    t_pos = t_pos - t_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    p_pos = p_pos - p_pos[:,0:1]
    p_pos = p_pos.unsqueeze(-1)
    
    ## Second
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    vec_p = p_pos[:, 1:] - p_pos[:, parent[1:]]
    
    ## Third
    analyik_rotmat = batch_rodrigues(analyik_thetas.view(-1,3)).view(-1,24,3,3)
    root_rotmat = analyik_rotmat[:,0:1]
    
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
    spine3_rotmat = batch_get_neck_orient(vec_qt, vec_t, parent[1:24], children, torch.float32)
   
    accumulate_rotmat = torch.matmul(accumulate_rotmat, spine3_rotmat)
    
    # 12
    local_twist_rotmat_12 = get_twist_rotmat_from_cossin(phis[:,2], vec_t[:,14])
    vec_pt_1512 = torch.matmul(accumulate_rotmat.transpose(1,2), vec_pt[:,14]) 
    local_swing_rotmat_12 = vectors2rotmat(vec_t[:,14], vec_pt_1512, torch.float32)
    local_rotmat_12 = torch.matmul(local_swing_rotmat_12, local_twist_rotmat_12)

    swing_rotmat_list.append(local_swing_rotmat_12)
 
    rf_rotmat = analyik_rotmat.clone()
    rf_rotmat[:,3] = local_rotmat_3
    rf_rotmat[:,6] = local_rotmat_6
    rf_rotmat[:,9] = spine3_rotmat
    rf_rotmat[:,12] = local_rotmat_12
    
    rf_theta = rotation_matrix_to_angle_axis(rf_rotmat.view(-1,3,3)).view(-1,24,3).contiguous()
    swing_rotmat = torch.stack(swing_rotmat_list, dim=1)

    return rf_theta, swing_rotmat

def get_body_part_func(body_part):
    if body_part == 'ARM':
        return refine_arm_rotmat
    elif body_part == 'LEG':
        return refine_leg_rotmat
    elif body_part == 'SPINE':
        return refine_spine_rotmat
    else:
        assert False, 'body part invalid'
    