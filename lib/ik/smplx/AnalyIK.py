# -*- coding: utf-8 -*-

import torch

from lib.utils.ik_utils import batch_get_pelvis_orient, batch_get_neck_orient, batch_get_spine3_twist, batch_get_wrist_twist, batch_get_wrist_orient, vectors2rotmat, vectors2rotmat_bk, \
        batch_get_orient, get_twist_rotmat

#------------------------------------------------------------------------------
def SMPLX_AnalyIK_V1(t_pos, p_pos):
    """
    Functions: Get SMPL pose(thetas) parameters.
    Arguments:
        t_pos: [b,76,3]
        p_pos: [b,76,3]
    """
    batch_size = t_pos.shape[0]
    device = t_pos.device
    
    t_pos = t_pos - t_pos[:,0:1]
    t_pos = t_pos.unsqueeze(-1)
    
    p_pos = p_pos - p_pos[:,0:1]
    p_pos = p_pos.unsqueeze(-1)
    
    
    # -------------------------Body-IK-----------------------------------------
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    ## root
    root_rotmat = batch_get_pelvis_orient(body_p_pos, body_t_pos, body_parent[1:21], body_children, torch.float32).unsqueeze(1)
    body_pt_pos = torch.matmul(root_rotmat.transpose(2,3), body_p_pos)
    
    body_vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    body_vec_pt = body_pt_pos[:, 1:] - body_pt_pos[:, body_parent[1:]]
    
    ## ------------------------left leg----------------------------------------
    ### 1
    local_rotmat_1 =  vectors2rotmat(body_vec_t[:,3], body_vec_pt[:,3], torch.float32)
    local_swing_rotmat_1, local_twist_rotmat_1 = get_twist_rotmat(local_rotmat_1, body_vec_t[:,3], torch.float32)
    accumulate_rotmat = local_swing_rotmat_1.clone()
    
    ### 4
    body_vec_74 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_pt[:,6])
    local_rotmat_4 = vectors2rotmat(body_vec_t[:,6], body_vec_74, torch.float32)
    local_swing_rotmat_4, local_twist_rotmat_4 = get_twist_rotmat(local_rotmat_4, body_vec_t[:,6], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_4)
    
    ### 7
    body_vec_107 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_pt[:,9])
    local_rotmat_7 = vectors2rotmat(body_vec_t[:,9], body_vec_107, torch.float32)
    local_swing_rotmat_7, local_twist_rotmat_7 = get_twist_rotmat(local_rotmat_7, body_vec_t[:,9], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_7)
    
    ## ------------------------right leg----------------------------------------
    ### 2
    local_rotmat_2 =  vectors2rotmat(body_vec_t[:,4], body_vec_pt[:,4], torch.float32)
    local_swing_rotmat_2, local_twist_rotmat_2 = get_twist_rotmat(local_rotmat_2, body_vec_t[:,4], torch.float32)
    accumulate_rotmat = local_swing_rotmat_2.clone()

    ### 5
    body_vec_85 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_pt[:,7])
    local_rotmat_5 = vectors2rotmat(body_vec_t[:,7], body_vec_85, torch.float32)
    local_swing_rotmat_5, local_twist_rotmat_5 = get_twist_rotmat(local_rotmat_5, body_vec_t[:,7], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_5)

    ### 8
    body_vec_118 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_pt[:,10])
    local_rotmat_8 = vectors2rotmat(body_vec_t[:,10], body_vec_118, torch.float32)
    local_swing_rotmat_8, local_twist_rotmat_8 = get_twist_rotmat(local_rotmat_8, body_vec_t[:,10], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_8)
    
    ##--------------------------spine------------------------------------------
    ### 3
    local_rotmat_3 = vectors2rotmat(body_vec_t[:,5], body_vec_pt[:,5], torch.float32)
    local_swing_rotmat_3, local_twist_rotmat_3 = get_twist_rotmat(local_rotmat_3, body_vec_t[:,5], torch.float32)
    accumulate_rotmat = local_swing_rotmat_3.clone()
    hand_accumulate_rotmat = torch.matmul(root_rotmat.squeeze(1), local_swing_rotmat_3)
   
    ### 6
    body_vec_96 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_pt[:,8])
    local_rotmat_6 = vectors2rotmat(body_vec_t[:,8], body_vec_96, torch.float32)
    local_swing_rotmat_6, local_twist_rotmat_6 = get_twist_rotmat(local_rotmat_6, body_vec_t[:,8], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_6).unsqueeze(1)
    hand_accumulate_rotmat = torch.matmul(hand_accumulate_rotmat, local_swing_rotmat_6)
    
    ### 9
    body_vec_qt = body_vec_pt.clone()
    body_vec_qt[:,8:9] = torch.matmul(accumulate_rotmat.transpose(2,3), body_vec_qt[:,8:9])
    body_vec_qt[:,11:] = torch.matmul(accumulate_rotmat.transpose(2,3), body_vec_qt[:,11:])
    spine3_rotmat = batch_get_neck_orient(body_vec_qt, body_vec_t, body_parent[1:21], body_children, torch.float32)
    
    local_rotmat_9 = spine3_rotmat
    accumulate_rotmat = torch.matmul(accumulate_rotmat.squeeze(1), local_rotmat_9).unsqueeze(1)
    
    lhand_accumulate_rotmat = torch.matmul(hand_accumulate_rotmat, local_rotmat_9)
    rhand_accumulate_rotmat = torch.matmul(hand_accumulate_rotmat, local_rotmat_9)
    
    ### 12
    body_vec_ppt = torch.matmul(accumulate_rotmat.transpose(2,3), body_vec_pt[:,11:])
    body_vec_1512 = body_vec_ppt[:,3]
    body_vec_1613 = body_vec_ppt[:,4]
    body_vec_1714 = body_vec_ppt[:,5]
    body_vec_1816 = body_vec_ppt[:,6]
    body_vec_1917 = body_vec_ppt[:,7]
    body_vec_2018 = body_vec_ppt[:,8]
    body_vec_2119 = body_vec_ppt[:,9]
    
    local_rotmat_12 = vectors2rotmat(body_vec_t[:,14], body_vec_1512, torch.float32)
    local_swing_rotmat_12, local_twist_rotmat_12 = get_twist_rotmat(local_rotmat_12, body_vec_t[:,14], torch.float32)
    
    ## --------------------------- left arm------------------------------------
    ### 13
    local_rotmat_13 = vectors2rotmat(body_vec_t[:,15], body_vec_1613, torch.float32)
    local_swing_rotmat_13, local_twist_rotmat_13 = get_twist_rotmat(local_rotmat_13, body_vec_t[:,15], torch.float32)
    accumulate_rotmat = local_swing_rotmat_13.clone()
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, local_swing_rotmat_13)

    ### 16
    body_vec_1816 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_1816)
    local_rotmat_16 = vectors2rotmat(body_vec_t[:,17], body_vec_1816, torch.float32)
    local_swing_rotmat_16, local_twist_rotmat_16 = get_twist_rotmat(local_rotmat_16, body_vec_t[:,17], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_16)    
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, local_swing_rotmat_16)
    
    ### 18
    body_vec_2018 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_2018)
    local_rotmat_18 = vectors2rotmat(body_vec_t[:,19], body_vec_2018, torch.float32)
    local_swing_rotmat_18, local_twist_rotmat_18 = get_twist_rotmat(local_rotmat_18, body_vec_t[:,19], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_18)    
    lhand_accumulate_rotmat = torch.matmul(lhand_accumulate_rotmat, local_swing_rotmat_18).unsqueeze(1)    

    ## --------------------------right arm-------------------------------------
    ### 14
    local_rotmat_14 = vectors2rotmat(body_vec_t[:,16], body_vec_1714, torch.float32)
    local_swing_rotmat_14, local_twist_rotmat_14 = get_twist_rotmat(local_rotmat_14, body_vec_t[:,16], torch.float32)
    accumulate_rotmat = local_swing_rotmat_14.clone()
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, local_swing_rotmat_14)
    
    ### 17
    body_vec_1917 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_1917)
    local_rotmat_17 = vectors2rotmat(body_vec_t[:,18], body_vec_1917, torch.float32)
    local_swing_rotmat_17, local_twist_rotmat_17 = get_twist_rotmat(local_rotmat_17, body_vec_t[:,18], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_17)    
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, local_swing_rotmat_17)
    
    ### 19
    body_vec_2119 = torch.matmul(accumulate_rotmat.transpose(1, 2), body_vec_2119)
    local_rotmat_19 = vectors2rotmat(body_vec_t[:,20], body_vec_2119, torch.float32)
    local_swing_rotmat_19, local_twist_rotmat_19 = get_twist_rotmat(local_rotmat_19, body_vec_t[:,20], torch.float32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_19)   
    rhand_accumulate_rotmat = torch.matmul(rhand_accumulate_rotmat, local_swing_rotmat_19).unsqueeze(1)  
    
    # -------------------------Lhand-IK-----------------------------------------
    lhand_index = torch.tensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70])
    lhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    lhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    lhand_t_pos = t_pos[:,lhand_index]
    lhand_p_pos = p_pos[:,lhand_index]

    #lhand_t_pos = lhand_t_pos - lhand_t_pos[:,0:1]
    #lhand_p_pos = lhand_p_pos - lhand_p_pos[:,0:1]

    lhand_vec_t = lhand_t_pos[:, 1:] - lhand_t_pos[:, lhand_parent[1:]]   # [b,n,3,1]
    lhand_vec_p = lhand_p_pos[:, 1:] - lhand_p_pos[:, lhand_parent[1:]]

    lhand_vec_pt = torch.matmul(lhand_accumulate_rotmat.transpose(2,3), lhand_vec_p)
    
    ## lWrist-20
    lhand_wrist_rotmat = batch_get_wrist_orient(lhand_vec_pt, lhand_vec_t, lhand_parent, lhand_children, torch.float32).unsqueeze(1)
    lhand_vec_pt = torch.matmul(lhand_wrist_rotmat.transpose(2,3), lhand_vec_pt)
    
    ## -------------------------l-thumb----------------------------------------
    ### 37
    global_rotmat_10 = vectors2rotmat_bk(lhand_vec_t[:,0:1], lhand_vec_pt[:,0:1], torch.float32)  # [b,1,3,3]
    global_rotmat_21 = vectors2rotmat_bk(lhand_vec_t[:,1:2], lhand_vec_pt[:,1:2], torch.float32)
    local_swing_rotmat_37 = torch.matmul(global_rotmat_10.transpose(2, 3), global_rotmat_21)
    accumulate_rotmat = local_swing_rotmat_37.clone()  # [b, 1, 3, 3]
    
    ### 38
    lhand_vec_ppt_21 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,1:2])  # [b,1,3,3]
    lhand_vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,2:3])
    
    global_rotmat_21 = vectors2rotmat_bk(lhand_vec_t[:,1:2], lhand_vec_ppt_21, torch.float32)
    global_rotmat_32 = vectors2rotmat_bk(lhand_vec_t[:,2:3], lhand_vec_ppt_32, torch.float32)
    local_swing_rotmat_38 = torch.matmul(global_rotmat_21.transpose(2, 3), global_rotmat_32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_38)
    
    ### 39
    lhand_vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,2:3])  # [b,1,3,3]
    lhand_vec_ppt_43 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,3:4])
    
    global_rotmat_32 = vectors2rotmat_bk(lhand_vec_t[:,2:3], lhand_vec_ppt_32, torch.float32)
    global_rotmat_43 = vectors2rotmat_bk(lhand_vec_t[:,3:4], lhand_vec_ppt_43, torch.float32)
    local_swing_rotmat_39 = torch.matmul(global_rotmat_32.transpose(2, 3), global_rotmat_43)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_39)
    
    ## -------------------------l-index----------------------------------------
    ## 25
    global_rotmat_50 = vectors2rotmat_bk(lhand_vec_t[:,4:5], lhand_vec_pt[:,4:5], torch.float32)  # [b,1,3,3]
    global_rotmat_65 = vectors2rotmat_bk(lhand_vec_t[:,5:6], lhand_vec_pt[:,5:6], torch.float32)
    local_swing_rotmat_25 = torch.matmul(global_rotmat_50.transpose(2, 3), global_rotmat_65)
    accumulate_rotmat = local_swing_rotmat_25.clone()  # [b, 1, 3, 3]
   
    ## 26
    lhand_vec_ppt_65 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,5:6])  # [b,1,3,3]
    lhand_vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,6:7])
    
    global_rotmat_65 = vectors2rotmat_bk(lhand_vec_t[:,5:6], lhand_vec_ppt_65, torch.float32)
    global_rotmat_76 = vectors2rotmat_bk(lhand_vec_t[:,6:7], lhand_vec_ppt_76, torch.float32)
    local_swing_rotmat_26 = torch.matmul(global_rotmat_65.transpose(2, 3), global_rotmat_76)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_26)
    
    ## 27
    lhand_vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,6:7])  # [b,1,3,3]
    lhand_vec_ppt_87 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,7:8])
    
    global_rotmat_76 = vectors2rotmat_bk(lhand_vec_t[:,6:7], lhand_vec_ppt_76, torch.float32)
    global_rotmat_87 = vectors2rotmat_bk(lhand_vec_t[:,7:8], lhand_vec_ppt_87, torch.float32)
    local_swing_rotmat_27 = torch.matmul(global_rotmat_76.transpose(2, 3), global_rotmat_87)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_27)
    
    ## -------------------------l-middle----------------------------------------
    ## 28
    global_rotmat_90 = vectors2rotmat_bk(lhand_vec_t[:,8:9], lhand_vec_pt[:,8:9], torch.float32)  # [b,1,3,3]
    global_rotmat_109 = vectors2rotmat_bk(lhand_vec_t[:,9:10], lhand_vec_pt[:,9:10], torch.float32)
    local_swing_rotmat_28 = torch.matmul(global_rotmat_90.transpose(2, 3), global_rotmat_109)
    accumulate_rotmat = local_swing_rotmat_28.clone()  # [b, 1, 3, 3]
   
    ## 29
    lhand_vec_ppt_109 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,9:10])  # [b,1,3,3]
    lhand_vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,10:11])
    
    global_rotmat_109 = vectors2rotmat_bk(lhand_vec_t[:,9:10], lhand_vec_ppt_109, torch.float32)
    global_rotmat_1110 = vectors2rotmat_bk(lhand_vec_t[:,10:11], lhand_vec_ppt_1110, torch.float32)
    local_swing_rotmat_29 = torch.matmul(global_rotmat_109.transpose(2, 3), global_rotmat_1110)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_29)
    
    ## 30
    lhand_vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,10:11])  # [b,1,3,3]
    lhand_vec_ppt_1211 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,11:12])
    
    global_rotmat_1110 = vectors2rotmat_bk(lhand_vec_t[:,10:11], lhand_vec_ppt_1110, torch.float32)
    global_rotmat_1211 = vectors2rotmat_bk(lhand_vec_t[:,11:12], lhand_vec_ppt_1211, torch.float32)
    local_swing_rotmat_30 = torch.matmul(global_rotmat_1110.transpose(2, 3), global_rotmat_1211)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_30)
    
    ## -------------------------l-ring-----------------------------------------
    ## 34
    global_rotmat_130 = vectors2rotmat_bk(lhand_vec_t[:,12:13], lhand_vec_pt[:,12:13], torch.float32)  # [b,1,3,3]
    global_rotmat_1413 = vectors2rotmat_bk(lhand_vec_t[:,13:14], lhand_vec_pt[:,13:14], torch.float32)
    local_swing_rotmat_34 = torch.matmul(global_rotmat_130.transpose(2, 3), global_rotmat_1413)
    accumulate_rotmat = local_swing_rotmat_34.clone()  # [b, 1, 3, 3]
   
    ## 35
    lhand_vec_ppt_1413 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,13:14])  # [b,1,3,3]
    lhand_vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,14:15])
    
    global_rotmat_1413 = vectors2rotmat_bk(lhand_vec_t[:,13:14], lhand_vec_ppt_1413, torch.float32)
    global_rotmat_1514 = vectors2rotmat_bk(lhand_vec_t[:,14:15], lhand_vec_ppt_1514, torch.float32)
    local_swing_rotmat_35 = torch.matmul(global_rotmat_1413.transpose(2, 3), global_rotmat_1514)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_35)
    
    ## 36
    lhand_vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,14:15])  # [b,1,3,3]
    lhand_vec_ppt_1615 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,15:16])
    
    global_rotmat_1514 = vectors2rotmat_bk(lhand_vec_t[:,14:15], lhand_vec_ppt_1514, torch.float32)
    global_rotmat_1615 = vectors2rotmat_bk(lhand_vec_t[:,15:16], lhand_vec_ppt_1615, torch.float32)
    local_swing_rotmat_36 = torch.matmul(global_rotmat_1514.transpose(2, 3), global_rotmat_1615)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_36)
    
    ## -------------------------l-Pinky----------------------------------------
    ## 31
    global_rotmat_170 = vectors2rotmat_bk(lhand_vec_t[:,16:17], lhand_vec_pt[:,16:17], torch.float32)  # [b,1,3,3]
    global_rotmat_1817 = vectors2rotmat_bk(lhand_vec_t[:,17:18], lhand_vec_pt[:,17:18], torch.float32)
    local_swing_rotmat_31 = torch.matmul(global_rotmat_170.transpose(2, 3), global_rotmat_1817)
    accumulate_rotmat = local_swing_rotmat_31.clone()  # [b, 1, 3, 3]
   
    ## 32
    lhand_vec_ppt_1817 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,17:18])  # [b,1,3,3]
    lhand_vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,18:19])
    
    global_rotmat_1817 = vectors2rotmat_bk(lhand_vec_t[:,17:18], lhand_vec_ppt_1817, torch.float32)
    global_rotmat_1918 = vectors2rotmat_bk(lhand_vec_t[:,18:19], lhand_vec_ppt_1918, torch.float32)
    local_swing_rotmat_32 = torch.matmul(global_rotmat_1817.transpose(2, 3), global_rotmat_1918)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_32)
    
    ## 33
    lhand_vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,18:19])  # [b,1,3,3]
    lhand_vec_ppt_2019 = torch.matmul(accumulate_rotmat.transpose(2,3), lhand_vec_pt[:,19:20])
    
    global_rotmat_1918 = vectors2rotmat_bk(lhand_vec_t[:,18:19], lhand_vec_ppt_1918, torch.float32)
    global_rotmat_2019 = vectors2rotmat_bk(lhand_vec_t[:,19:20], lhand_vec_ppt_2019, torch.float32)
    local_swing_rotmat_33 = torch.matmul(global_rotmat_1918.transpose(2, 3), global_rotmat_2019)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_33)
    
    
    # -------------------------Rhand-IK-----------------------------------------
    rhand_index = torch.tensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
    rhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    rhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    rhand_t_pos = t_pos[:,rhand_index]
    rhand_p_pos = p_pos[:,rhand_index]

    rhand_t_pos = rhand_t_pos - rhand_t_pos[:,0:1]
    rhand_p_pos = rhand_p_pos - rhand_p_pos[:,0:1]

    rhand_vec_t = rhand_t_pos[:, 1:] - rhand_t_pos[:, rhand_parent[1:]]   # [b,n,3,1]
    rhand_vec_p = rhand_p_pos[:, 1:] - rhand_p_pos[:, rhand_parent[1:]]

    rhand_vec_pt = torch.matmul(rhand_accumulate_rotmat.transpose(2,3), rhand_vec_p)
    
    ## rWrist-20
    rhand_wrist_rotmat = batch_get_wrist_orient(rhand_vec_pt, rhand_vec_t, rhand_parent, rhand_children, torch.float32).unsqueeze(1)
    rhand_vec_pt = torch.matmul(rhand_wrist_rotmat.transpose(2,3), rhand_vec_pt)
    
    ## -------------------------r-thumb----------------------------------------
    ### 52
    global_rotmat_10 = vectors2rotmat_bk(rhand_vec_t[:,0:1], rhand_vec_pt[:,0:1], torch.float32)  # [b,1,3,3]
    global_rotmat_21 = vectors2rotmat_bk(rhand_vec_t[:,1:2], rhand_vec_pt[:,1:2], torch.float32)
    local_swing_rotmat_52 = torch.matmul(global_rotmat_10.transpose(2, 3), global_rotmat_21)
    accumulate_rotmat = local_swing_rotmat_52.clone()  # [b, 1, 3, 3]
    
    ### 53
    rhand_vec_ppt_21 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,1:2])  # [b,1,3,3]
    rhand_vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,2:3])
    
    global_rotmat_21 = vectors2rotmat_bk(rhand_vec_t[:,1:2], rhand_vec_ppt_21, torch.float32)
    global_rotmat_32 = vectors2rotmat_bk(rhand_vec_t[:,2:3], rhand_vec_ppt_32, torch.float32)
    local_swing_rotmat_53 = torch.matmul(global_rotmat_21.transpose(2, 3), global_rotmat_32)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_53)
    
    ### 54
    rhand_vec_ppt_32 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,2:3])  # [b,1,3,3]
    rhand_vec_ppt_43 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,3:4])
    
    global_rotmat_32 = vectors2rotmat_bk(rhand_vec_t[:,2:3], rhand_vec_ppt_32, torch.float32)
    global_rotmat_43 = vectors2rotmat_bk(rhand_vec_t[:,3:4], rhand_vec_ppt_43, torch.float32)
    local_swing_rotmat_54 = torch.matmul(global_rotmat_32.transpose(2, 3), global_rotmat_43)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_54)
    
    ## -------------------------r-index----------------------------------------
    ## 40
    global_rotmat_50 = vectors2rotmat_bk(rhand_vec_t[:,4:5], rhand_vec_pt[:,4:5], torch.float32)  # [b,1,3,3]
    global_rotmat_65 = vectors2rotmat_bk(rhand_vec_t[:,5:6], rhand_vec_pt[:,5:6], torch.float32)
    local_swing_rotmat_40 = torch.matmul(global_rotmat_50.transpose(2, 3), global_rotmat_65)
    accumulate_rotmat = local_swing_rotmat_40.clone()  # [b, 1, 3, 3]
   
    ## 41
    rhand_vec_ppt_65 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,5:6])  # [b,1,3,3]
    rhand_vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,6:7])
    
    global_rotmat_65 = vectors2rotmat_bk(rhand_vec_t[:,5:6], rhand_vec_ppt_65, torch.float32)
    global_rotmat_76 = vectors2rotmat_bk(rhand_vec_t[:,6:7], rhand_vec_ppt_76, torch.float32)
    local_swing_rotmat_41 = torch.matmul(global_rotmat_65.transpose(2, 3), global_rotmat_76)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_41)
    
    ## 42
    rhand_vec_ppt_76 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,6:7])  # [b,1,3,3]
    rhand_vec_ppt_87 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,7:8])
    
    global_rotmat_76 = vectors2rotmat_bk(rhand_vec_t[:,6:7], rhand_vec_ppt_76, torch.float32)
    global_rotmat_87 = vectors2rotmat_bk(rhand_vec_t[:,7:8], rhand_vec_ppt_87, torch.float32)
    local_swing_rotmat_42 = torch.matmul(global_rotmat_76.transpose(2, 3), global_rotmat_87)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_42)
    
    ## -------------------------r-middle----------------------------------------
    ## 43
    global_rotmat_90 = vectors2rotmat_bk(rhand_vec_t[:,8:9], rhand_vec_pt[:,8:9], torch.float32)  # [b,1,3,3]
    global_rotmat_109 = vectors2rotmat_bk(rhand_vec_t[:,9:10], rhand_vec_pt[:,9:10], torch.float32)
    local_swing_rotmat_43 = torch.matmul(global_rotmat_90.transpose(2, 3), global_rotmat_109)
    accumulate_rotmat = local_swing_rotmat_43.clone()  # [b, 1, 3, 3]
   
    ## 44
    rhand_vec_ppt_109 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,9:10])  # [b,1,3,3]
    rhand_vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,10:11])
    
    global_rotmat_109 = vectors2rotmat_bk(rhand_vec_t[:,9:10], rhand_vec_ppt_109, torch.float32)
    global_rotmat_1110 = vectors2rotmat_bk(rhand_vec_t[:,10:11], rhand_vec_ppt_1110, torch.float32)
    local_swing_rotmat_44 = torch.matmul(global_rotmat_109.transpose(2, 3), global_rotmat_1110)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_44)
    
    ## 45
    rhand_vec_ppt_1110 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,10:11])  # [b,1,3,3]
    rhand_vec_ppt_1211 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,11:12])
    
    global_rotmat_1110 = vectors2rotmat_bk(rhand_vec_t[:,10:11], rhand_vec_ppt_1110, torch.float32)
    global_rotmat_1211 = vectors2rotmat_bk(rhand_vec_t[:,11:12], rhand_vec_ppt_1211, torch.float32)
    local_swing_rotmat_45 = torch.matmul(global_rotmat_1110.transpose(2, 3), global_rotmat_1211)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_45)
    
    ## -------------------------r-ring-----------------------------------------
    ## 49
    global_rotmat_130 = vectors2rotmat_bk(rhand_vec_t[:,12:13], rhand_vec_pt[:,12:13], torch.float32)  # [b,1,3,3]
    global_rotmat_1413 = vectors2rotmat_bk(rhand_vec_t[:,13:14], rhand_vec_pt[:,13:14], torch.float32)
    local_swing_rotmat_49 = torch.matmul(global_rotmat_130.transpose(2, 3), global_rotmat_1413)
    accumulate_rotmat = local_swing_rotmat_49.clone()  # [b, 1, 3, 3]
   
    ## 50
    rhand_vec_ppt_1413 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,13:14])  # [b,1,3,3]
    rhand_vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,14:15])
    
    global_rotmat_1413 = vectors2rotmat_bk(rhand_vec_t[:,13:14], rhand_vec_ppt_1413, torch.float32)
    global_rotmat_1514 = vectors2rotmat_bk(rhand_vec_t[:,14:15], rhand_vec_ppt_1514, torch.float32)
    local_swing_rotmat_50 = torch.matmul(global_rotmat_1413.transpose(2, 3), global_rotmat_1514)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_50)
    
    ## 51
    rhand_vec_ppt_1514 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,14:15])  # [b,1,3,3]
    rhand_vec_ppt_1615 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,15:16])
    
    global_rotmat_1514 = vectors2rotmat_bk(rhand_vec_t[:,14:15], rhand_vec_ppt_1514, torch.float32)
    global_rotmat_1615 = vectors2rotmat_bk(rhand_vec_t[:,15:16], rhand_vec_ppt_1615, torch.float32)
    local_swing_rotmat_51 = torch.matmul(global_rotmat_1514.transpose(2, 3), global_rotmat_1615)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_51)
    
    ## -------------------------r-Pinky----------------------------------------
    ## 46
    global_rotmat_170 = vectors2rotmat_bk(rhand_vec_t[:,16:17], rhand_vec_pt[:,16:17], torch.float32)  # [b,1,3,3]
    global_rotmat_1817 = vectors2rotmat_bk(rhand_vec_t[:,17:18], rhand_vec_pt[:,17:18], torch.float32)
    local_swing_rotmat_46 = torch.matmul(global_rotmat_170.transpose(2, 3), global_rotmat_1817)
    accumulate_rotmat = local_swing_rotmat_46.clone()  # [b, 1, 3, 3]
   
    ## 47
    rhand_vec_ppt_1817 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,17:18])  # [b,1,3,3]
    rhand_vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,18:19])
    
    global_rotmat_1817 = vectors2rotmat_bk(rhand_vec_t[:,17:18], rhand_vec_ppt_1817, torch.float32)
    global_rotmat_1918 = vectors2rotmat_bk(rhand_vec_t[:,18:19], rhand_vec_ppt_1918, torch.float32)
    local_swing_rotmat_47 = torch.matmul(global_rotmat_1817.transpose(2, 3), global_rotmat_1918)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_47)
    
    ## 48
    rhand_vec_ppt_1918 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,18:19])  # [b,1,3,3]
    rhand_vec_ppt_2019 = torch.matmul(accumulate_rotmat.transpose(2,3), rhand_vec_pt[:,19:20])
    
    global_rotmat_1918 = vectors2rotmat_bk(rhand_vec_t[:,18:19], rhand_vec_ppt_1918, torch.float32)
    global_rotmat_2019 = vectors2rotmat_bk(rhand_vec_t[:,19:20], rhand_vec_ppt_2019, torch.float32)
    local_swing_rotmat_48 = torch.matmul(global_rotmat_1918.transpose(2, 3), global_rotmat_2019)
    accumulate_rotmat = torch.matmul(accumulate_rotmat, local_swing_rotmat_48)
    
    
    ## -----------------------Local_rotmat-------------------------------------
    iden = torch.eye(3).unsqueeze(0).to(device)
    
    ## global orient
    global_orient_rotmat = root_rotmat
    
    ## body
    body_local_rotmat = iden.unsqueeze(1).repeat(batch_size,21,1,1)
    body_local_rotmat[:,0] = local_swing_rotmat_1
    body_local_rotmat[:,3] = local_swing_rotmat_4
    body_local_rotmat[:,6] = local_swing_rotmat_7
    
    body_local_rotmat[:,1] = local_swing_rotmat_2
    body_local_rotmat[:,4] = local_swing_rotmat_5
    body_local_rotmat[:,7] = local_swing_rotmat_8
    
    body_local_rotmat[:,2] = local_swing_rotmat_3
    body_local_rotmat[:,5] = local_swing_rotmat_6
    body_local_rotmat[:,8] = local_rotmat_9
    body_local_rotmat[:,11] = local_swing_rotmat_12
    
    body_local_rotmat[:,12] = local_swing_rotmat_13
    body_local_rotmat[:,15] = local_swing_rotmat_16
    body_local_rotmat[:,17] = local_swing_rotmat_18
    
    body_local_rotmat[:,13] = local_swing_rotmat_14
    body_local_rotmat[:,16] = local_swing_rotmat_17
    body_local_rotmat[:,18] = local_swing_rotmat_19
    
    
    body_local_rotmat[:,19] = lhand_wrist_rotmat.squeeze(1)
    body_local_rotmat[:,20] = rhand_wrist_rotmat.squeeze(1)
    
    
    lhand_local_rotmat = iden.unsqueeze(1).repeat(batch_size,15,1,1)
    lhand_local_rotmat[:,0:1] = local_swing_rotmat_25
    lhand_local_rotmat[:,1:2] = local_swing_rotmat_26
    lhand_local_rotmat[:,2:3] = local_swing_rotmat_27
    lhand_local_rotmat[:,3:4] = local_swing_rotmat_28
    lhand_local_rotmat[:,4:5] = local_swing_rotmat_29
    lhand_local_rotmat[:,5:6] = local_swing_rotmat_30
    lhand_local_rotmat[:,6:7] = local_swing_rotmat_31
    lhand_local_rotmat[:,7:8] = local_swing_rotmat_32
    lhand_local_rotmat[:,8:9] = local_swing_rotmat_33 
    lhand_local_rotmat[:,9:10] = local_swing_rotmat_34
    lhand_local_rotmat[:,10:11] = local_swing_rotmat_35
    lhand_local_rotmat[:,11:12] = local_swing_rotmat_36
    lhand_local_rotmat[:,12:13] = local_swing_rotmat_37
    lhand_local_rotmat[:,13:14] = local_swing_rotmat_38
    lhand_local_rotmat[:,14:15] = local_swing_rotmat_39

    rhand_local_rotmat = iden.unsqueeze(1).repeat(batch_size,15,1,1) 
    rhand_local_rotmat[:,0:1] = local_swing_rotmat_40
    rhand_local_rotmat[:,1:2] = local_swing_rotmat_41
    rhand_local_rotmat[:,2:3] = local_swing_rotmat_42
    rhand_local_rotmat[:,3:4] = local_swing_rotmat_43
    rhand_local_rotmat[:,4:5] = local_swing_rotmat_44
    rhand_local_rotmat[:,5:6] = local_swing_rotmat_45
    rhand_local_rotmat[:,6:7] = local_swing_rotmat_46
    rhand_local_rotmat[:,7:8] = local_swing_rotmat_47
    rhand_local_rotmat[:,8:9] = local_swing_rotmat_48 
    rhand_local_rotmat[:,9:10] = local_swing_rotmat_49
    rhand_local_rotmat[:,10:11] = local_swing_rotmat_50
    rhand_local_rotmat[:,11:12] = local_swing_rotmat_51
    rhand_local_rotmat[:,12:13] = local_swing_rotmat_52
    rhand_local_rotmat[:,13:14] = local_swing_rotmat_53
    rhand_local_rotmat[:,14:15] = local_swing_rotmat_54
    
    return global_orient_rotmat, body_local_rotmat, lhand_local_rotmat, rhand_local_rotmat











