# -*- coding: utf-8 -*-

import torch

from lib.utils.ik_utils import rotation_matrix_to_angle_axis, batch_get_wrist_orient, vectors2rotmat_bk



def MANO_AnalyIK(t_pos, p_pos, parent, children):
    """
    Functions: Get SMPL pose(thetas) parameters.
    Arguments:
        t_pos: [b,21,3,1]
        p_pos: [b,21,3,1]
    """
    batch_size = t_pos.shape[0]
    device = t_pos.device
    
    t_pos = t_pos - t_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    
    p_pos = p_pos - p_pos[:,0:1]
    p_pos = p_pos.unsqueeze(-1)
    
    
    ## root
    wrist_rotmat = batch_get_wrist_orient(p_pos[:,1:], t_pos[:,1:], parent, children, torch.float32).unsqueeze(1)
    pt_pos = torch.matmul(wrist_rotmat.transpose(2,3), p_pos)
    
    ## 
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]   # [b,n,3,1]
    vec_pt = pt_pos[:, 1:] - pt_pos[:, parent[1:]]
    

    # Thumb 
    ## 1
    global_rotmat_10 = vectors2rotmat_bk(vec_t[:,0:1], vec_pt[:,0:1], torch.float32)  # [b,1,3,3]
    global_rotmat_21 = vectors2rotmat_bk(vec_t[:,1:2], vec_pt[:,1:2], torch.float32)
    local_swing_rotmat_1 = torch.matmul(global_rotmat_10.transpose(2, 3), global_rotmat_21)
    accumulate_rotmat = local_swing_rotmat_1.clone()  # [b, 1, 3, 3]
   
    ## 2
    vec_ppt_21 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,1:2])  # [b,1,3,3]
    vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,2:3])
    
    global_rotmat_21 = vectors2rotmat_bk(vec_t[:,1:2], vec_ppt_21, torch.float32)
    global_rotmat_32 = vectors2rotmat_bk(vec_t[:,2:3], vec_ppt_32, torch.float32)
    local_swing_rotmat_2 = torch.matmul(global_rotmat_21.transpose(2, 3), global_rotmat_32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_2)
    
    ## 3
    vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,2:3])  # [b,1,3,3]
    vec_ppt_43 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,3:4])
    
    global_rotmat_32 = vectors2rotmat_bk(vec_t[:,2:3], vec_ppt_32, torch.float32)
    global_rotmat_43 = vectors2rotmat_bk(vec_t[:,3:4], vec_ppt_43, torch.float32)
    local_swing_rotmat_3 = torch.matmul(global_rotmat_32.transpose(2, 3), global_rotmat_43)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_3)
    
    # Index
    ## 5
    global_rotmat_50 = vectors2rotmat_bk(vec_t[:,4:5], vec_pt[:,4:5], torch.float32)  # [b,1,3,3]
    global_rotmat_65 = vectors2rotmat_bk(vec_t[:,5:6], vec_pt[:,5:6], torch.float32)
    local_swing_rotmat_5 = torch.matmul(global_rotmat_50.transpose(2, 3), global_rotmat_65)
    accumulate_rotmat = local_swing_rotmat_5.clone()  # [b, 1, 3, 3]
   
    ## 6
    vec_ppt_65 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,5:6])  # [b,1,3,3]
    vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,6:7])
    
    global_rotmat_65 = vectors2rotmat_bk(vec_t[:,5:6], vec_ppt_65, torch.float32)
    global_rotmat_76 = vectors2rotmat_bk(vec_t[:,6:7], vec_ppt_76, torch.float32)
    local_swing_rotmat_6 = torch.matmul(global_rotmat_65.transpose(2, 3), global_rotmat_76)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_6)
    
    ## 7
    vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,6:7])  # [b,1,3,3]
    vec_ppt_87 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,7:8])
    
    global_rotmat_76 = vectors2rotmat_bk(vec_t[:,6:7], vec_ppt_76, torch.float32)
    global_rotmat_87 = vectors2rotmat_bk(vec_t[:,7:8], vec_ppt_87, torch.float32)
    local_swing_rotmat_7 = torch.matmul(global_rotmat_76.transpose(2, 3), global_rotmat_87)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_7)
    
    # Middle
    ## 9
    global_rotmat_90 = vectors2rotmat_bk(vec_t[:,8:9], vec_pt[:,8:9], torch.float32)  # [b,1,3,3]
    global_rotmat_109 = vectors2rotmat_bk(vec_t[:,9:10], vec_pt[:,9:10], torch.float32)
    local_swing_rotmat_9 = torch.matmul(global_rotmat_90.transpose(2, 3), global_rotmat_109)
    accumulate_rotmat = local_swing_rotmat_9.clone()  # [b, 1, 3, 3]
   
    ## 10
    vec_ppt_109 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,9:10])  # [b,1,3,3]
    vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,10:11])
    
    global_rotmat_109 = vectors2rotmat_bk(vec_t[:,9:10], vec_ppt_109, torch.float32)
    global_rotmat_1110 = vectors2rotmat_bk(vec_t[:,10:11], vec_ppt_1110, torch.float32)
    local_swing_rotmat_10 = torch.matmul(global_rotmat_109.transpose(2, 3), global_rotmat_1110)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_10)
    
    ## 11
    vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,10:11])  # [b,1,3,3]
    vec_ppt_1211 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,11:12])
    
    global_rotmat_1110 = vectors2rotmat_bk(vec_t[:,10:11], vec_ppt_1110, torch.float32)
    global_rotmat_1211 = vectors2rotmat_bk(vec_t[:,11:12], vec_ppt_1211, torch.float32)
    local_swing_rotmat_11 = torch.matmul(global_rotmat_1110.transpose(2, 3), global_rotmat_1211)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_11)
    
    # Ring
    ## 13
    global_rotmat_130 = vectors2rotmat_bk(vec_t[:,12:13], vec_pt[:,12:13], torch.float32)  # [b,1,3,3]
    global_rotmat_1413 = vectors2rotmat_bk(vec_t[:,13:14], vec_pt[:,13:14], torch.float32)
    local_swing_rotmat_13 = torch.matmul(global_rotmat_130.transpose(2, 3), global_rotmat_1413)
    accumulate_rotmat = local_swing_rotmat_13.clone()  # [b, 1, 3, 3]
   
    ## 14
    vec_ppt_1413 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,13:14])  # [b,1,3,3]
    vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,14:15])
    
    global_rotmat_1413 = vectors2rotmat_bk(vec_t[:,13:14], vec_ppt_1413, torch.float32)
    global_rotmat_1514 = vectors2rotmat_bk(vec_t[:,14:15], vec_ppt_1514, torch.float32)
    local_swing_rotmat_14 = torch.matmul(global_rotmat_1413.transpose(2, 3), global_rotmat_1514)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_14)
    
    ## 15
    vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,14:15])  # [b,1,3,3]
    vec_ppt_1615 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,15:16])
    
    global_rotmat_1514 = vectors2rotmat_bk(vec_t[:,14:15], vec_ppt_1514, torch.float32)
    global_rotmat_1615 = vectors2rotmat_bk(vec_t[:,15:16], vec_ppt_1615, torch.float32)
    local_swing_rotmat_15 = torch.matmul(global_rotmat_1514.transpose(2, 3), global_rotmat_1615)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_15)
    
    # Pinky
    ## 17
    global_rotmat_170 = vectors2rotmat_bk(vec_t[:,16:17], vec_pt[:,16:17], torch.float32)  # [b,1,3,3]
    global_rotmat_1817 = vectors2rotmat_bk(vec_t[:,17:18], vec_pt[:,17:18], torch.float32)
    local_swing_rotmat_17 = torch.matmul(global_rotmat_170.transpose(2, 3), global_rotmat_1817)
    accumulate_rotmat = local_swing_rotmat_17.clone()  # [b, 1, 3, 3]
   
    ## 18
    vec_ppt_1817 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,17:18])  # [b,1,3,3]
    vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,18:19])
    
    global_rotmat_1817 = vectors2rotmat_bk(vec_t[:,17:18], vec_ppt_1817, torch.float32)
    global_rotmat_1918 = vectors2rotmat_bk(vec_t[:,18:19], vec_ppt_1918, torch.float32)
    local_swing_rotmat_18 = torch.matmul(global_rotmat_1817.transpose(2, 3), global_rotmat_1918)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_18)
    
    ## 19
    vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,18:19])  # [b,1,3,3]
    vec_ppt_2019 = torch.matmul(accumulate_rotmat.transpose(2,3), vec_pt[:,19:20])
    
    global_rotmat_1918 = vectors2rotmat_bk(vec_t[:,18:19], vec_ppt_1918, torch.float32)
    global_rotmat_2019 = vectors2rotmat_bk(vec_t[:,19:20], vec_ppt_2019, torch.float32)
    local_swing_rotmat_19 = torch.matmul(global_rotmat_1918.transpose(2, 3), global_rotmat_2019)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_19)
    
    iden = torch.eye(3).unsqueeze(0).to(device)
    local_rotmat = iden.unsqueeze(1).repeat(batch_size,16,1,1)
    
    local_rotmat[:,0:1] = wrist_rotmat
    
    local_rotmat[:,13:14] = local_swing_rotmat_1
    local_rotmat[:,14:15] = local_swing_rotmat_2
    local_rotmat[:,15:16] = local_swing_rotmat_3
   
    local_rotmat[:,1:2] = local_swing_rotmat_5
    local_rotmat[:,2:3] = local_swing_rotmat_6
    local_rotmat[:,3:4] = local_swing_rotmat_7 
    
    local_rotmat[:,4:5] = local_swing_rotmat_9
    local_rotmat[:,5:6] = local_swing_rotmat_10
    local_rotmat[:,6:7] = local_swing_rotmat_11 
    
    local_rotmat[:,10:11] = local_swing_rotmat_13
    local_rotmat[:,11:12] = local_swing_rotmat_14
    local_rotmat[:,12:13] = local_swing_rotmat_15 
    
    local_rotmat[:,7:8] = local_swing_rotmat_17
    local_rotmat[:,8:9] = local_swing_rotmat_18
    local_rotmat[:,9:10] = local_swing_rotmat_19 
   
    theta = rotation_matrix_to_angle_axis(local_rotmat.view(-1,3,3)).view(-1,16,3).contiguous()
    theta = theta.view(-1,48)
    return theta