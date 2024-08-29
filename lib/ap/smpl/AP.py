# -*- coding: utf-8 -*-
import torch

from lib.utils.ik_utils import batch_get_pelvis_orient, batch_get_neck_orient, vectors2rotmat_bk

from lib.utils.si_utils import get_bl_from_pos, distance

def SMPL_AP_V1(t_pos, p_pos, parent, children):

    device = p_pos.device
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # bl/bd
    t_pos_bl = get_bl_from_pos(t_pos, parent)

    ## 0
    q_pos_0 = torch.zeros([t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    root_rotmat = batch_get_pelvis_orient(p_pos.unsqueeze(-1), t_pos.unsqueeze(-1), parent[1:24], children, torch.float32).unsqueeze(1)  # [b,1,1,3,3]
   
    ## ---------------------left leg-------------------------------------------
    ### 1
    q_pos_1 = torch.matmul(root_rotmat, t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 4
    p4_q1_bd = (p_pos[:,4:5]-q_pos_1) / distance(p_pos[:,4:5], q_pos_1).unsqueeze(-1)
    q_pos_4 = q_pos_1 + p4_q1_bd * t_pos_bl[:,4:5].unsqueeze(-1)
    
    ### 7
    p7_q4_bd = (p_pos[:,7:8]-q_pos_4) / distance(p_pos[:,7:8], q_pos_4).unsqueeze(-1)
    q_pos_7 = q_pos_4 + p7_q4_bd * t_pos_bl[:,7:8].unsqueeze(-1)
  
    #### 10
    p10_q7_bd = (p_pos[:,10:11]-q_pos_7) / distance(p_pos[:,10:11], q_pos_7).unsqueeze(-1)
    q_pos_10 = q_pos_7 + p10_q7_bd * t_pos_bl[:,10:11].unsqueeze(-1)
    

    ### ---------------------Right leg---------------------------------------------
    #### 2
    q_pos_2 = torch.matmul(root_rotmat, t_pos[:,2:3,:].unsqueeze(-1)).squeeze(-1)
  
    #### 5
    p5_q2_bd = (p_pos[:,5:6]-q_pos_2) / distance(p_pos[:,5:6], q_pos_2).unsqueeze(-1)
    q_pos_5 = q_pos_2 + p5_q2_bd * t_pos_bl[:,5:6].unsqueeze(-1)
 
    #### 8
    p8_q5_bd = (p_pos[:,8:9]-q_pos_5) / distance(p_pos[:,8:9], q_pos_5).unsqueeze(-1)
    q_pos_8 = q_pos_5 + p8_q5_bd * t_pos_bl[:,8:9].unsqueeze(-1)
 
    #### 11
    p11_q8_bd = (p_pos[:,11:12]-q_pos_8) / distance(p_pos[:,11:12], q_pos_8).unsqueeze(-1)
    q_pos_11 = q_pos_8 + p11_q8_bd * t_pos_bl[:,11:12].unsqueeze(-1)
   
   
    ### ---------------------Spine---------------------------------------------
    #### 3
    q_pos_3 = torch.matmul(root_rotmat, t_pos[:,3:4,:].unsqueeze(-1)).squeeze(-1)
 
    #### 6
    p6_q3_bd = (p_pos[:,6:7]-q_pos_3) / distance(p_pos[:,6:7], q_pos_3).unsqueeze(-1)
    q_pos_6 = q_pos_3 + p6_q3_bd * t_pos_bl[:,6:7].unsqueeze(-1)
  
    #### 9
    p9_q6_bd = (p_pos[:,9:10]-q_pos_6) / distance(p_pos[:,9:10], q_pos_6).unsqueeze(-1)
    q_pos_9 = q_pos_6 + p9_q6_bd * t_pos_bl[:,9:10].unsqueeze(-1)

    p_pos[:,9:10] = q_pos_9
    vec_pt = p_pos[:, 1:] - p_pos[:, parent[1:]]
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    spine3_rotmat = batch_get_neck_orient(vec_pt.unsqueeze(-1), vec_t.unsqueeze(-1), parent[1:24], children, torch.float32).unsqueeze(1)
    
    #### 12
    q_pos_12 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,11:12,:].unsqueeze(-1)).squeeze(-1)
  
    #### 15
    p15_q12_bd = (p_pos[:,15:16]-q_pos_12) / distance(p_pos[:,15:16], q_pos_12).unsqueeze(-1)
    q_pos_15 = q_pos_12 + p15_q12_bd * t_pos_bl[:,15:16].unsqueeze(-1)
    

    ### ---------------------左----------------------------------------------------
    #### 13
    q_pos_13 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 16
    p16_q13_bd = (p_pos[:,16:17]-q_pos_13) / distance(p_pos[:,16:17], q_pos_13).unsqueeze(-1)
    q_pos_16 = q_pos_13 + p16_q13_bd * t_pos_bl[:,16:17].unsqueeze(-1)
    
    #### 18
    p18_q16_bd = (p_pos[:,18:19]-q_pos_16) / distance(p_pos[:,18:19], q_pos_16).unsqueeze(-1)
    q_pos_18 = q_pos_16 + p18_q16_bd * t_pos_bl[:,18:19].unsqueeze(-1)
   
    #### 20
    p20_q18_bd = (p_pos[:,20:21]-q_pos_18) / distance(p_pos[:,20:21], q_pos_18).unsqueeze(-1)
    q_pos_20 = q_pos_18 + p20_q18_bd * t_pos_bl[:,20:21].unsqueeze(-1)
   
    #### 22
    p22_q20_bd = (p_pos[:,22:23]-q_pos_20) / distance(p_pos[:,22:23], q_pos_20).unsqueeze(-1)
    q_pos_22 = q_pos_20 + p22_q20_bd * t_pos_bl[:,22:23].unsqueeze(-1)
   
    ### ---------------------右----------------------------------------------------
    #### 14
    q_pos_14 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,13:14,:].unsqueeze(-1)).squeeze(-1)
  
    #### 17
    p17_q14_bd = (p_pos[:,17:18]-q_pos_14) / distance(p_pos[:,17:18], q_pos_14).unsqueeze(-1)
    q_pos_17 = q_pos_14 + p17_q14_bd * t_pos_bl[:,17:18].unsqueeze(-1)
  
    #### 19
    p19_q17_bd = (p_pos[:,19:20]-q_pos_17) / distance(p_pos[:,19:20], q_pos_17).unsqueeze(-1)
    q_pos_19 = q_pos_17 + p19_q17_bd * t_pos_bl[:,19:20].unsqueeze(-1)
  
    #### 21
    p21_q19_bd = (p_pos[:,21:22]-q_pos_19) / distance(p_pos[:,21:22], q_pos_19).unsqueeze(-1)
    q_pos_21 = q_pos_19 + p21_q19_bd * t_pos_bl[:,21:22].unsqueeze(-1)
  
    #### 23
    p23_q21_bd = (p_pos[:,23:24]-q_pos_21) / distance(p_pos[:,23:24], q_pos_21).unsqueeze(-1)
    q_pos_23 = q_pos_21 + p23_q21_bd * t_pos_bl[:,23:24].unsqueeze(-1)
  
    q_pos_list = [q_pos_0, q_pos_1, q_pos_2, q_pos_3, q_pos_4, q_pos_5, q_pos_6, q_pos_7, q_pos_8, q_pos_9,
              q_pos_10, q_pos_11, q_pos_12, q_pos_13, q_pos_14, q_pos_15, q_pos_16, q_pos_17, q_pos_18, q_pos_19, q_pos_20,
              q_pos_21, q_pos_22, q_pos_23]
    
    q_pos = torch.cat(q_pos_list, dim=1)

    return q_pos



def move(Am, Bm, Cm, C, bl_AmBm, bl_BmCm):
    
    Bm_Am_bd = (Bm - Am) / distance(Bm, Am).unsqueeze(-1) 
    C_Am_bd = (C - Am) / distance(C, Am).unsqueeze(-1)
    Cm_Am_bd = (Cm - Am) / distance(Cm, Am).unsqueeze(-1) 
    
    rotmat_1 = vectors2rotmat_bk(Cm_Am_bd.unsqueeze(-1), C_Am_bd.unsqueeze(-1), dtype=torch.float32)
   
    rot_Bm_Am_bd = torch.matmul(rotmat_1, Bm_Am_bd.unsqueeze(-1)).squeeze(-1)    
    Bm = Am + rot_Bm_Am_bd * bl_AmBm.unsqueeze(-1)
    
    C_Bm_bd = (C - Bm) / distance(C, Bm).unsqueeze(-1)
    Cm = Bm + C_Bm_bd * bl_BmCm.unsqueeze(-1)

    return Bm, Cm

def itertive_move(Am, Bm, Cm, C, bl_AmBm, bl_BmCm, iter_num):
    for i in range(iter_num):
        Bm, Cm = move(Am, Bm, Cm, C, bl_AmBm, bl_BmCm)
    
    return Bm, Cm

def SMPL_AP_V2(t_pos, p_pos, parent, children, iter_num = 3):

    device = p_pos.device
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # bl/bd
    t_pos_bl = get_bl_from_pos(t_pos, parent)

    ## 0
    q_pos_0 = torch.zeros([t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    root_rotmat = batch_get_pelvis_orient(p_pos.unsqueeze(-1), t_pos.unsqueeze(-1), parent[1:24], children, torch.float32).unsqueeze(1)  # [b,1,1,3,3]
   
    ## ---------------------left leg-------------------------------------------
    ### 1
    q_pos_1 = torch.matmul(root_rotmat, t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 4 and 7
    #### old
    p4_q1_bd = (p_pos[:,4:5]-q_pos_1) / distance(p_pos[:,4:5], q_pos_1).unsqueeze(-1)
    q_pos_4 = q_pos_1 + p4_q1_bd * t_pos_bl[:,4:5].unsqueeze(-1)
    
    p7_q4_bd = (p_pos[:,7:8]-q_pos_4) / distance(p_pos[:,7:8], q_pos_4).unsqueeze(-1)
    q_pos_7 = q_pos_4 + p7_q4_bd * t_pos_bl[:,7:8].unsqueeze(-1)
    
    #### new  
    q_pos_4, q_pos_7 = itertive_move(q_pos_1, q_pos_4, q_pos_7, p_pos[:,7:8], t_pos_bl[:, 4:5], t_pos_bl[:, 7:8], iter_num=iter_num)
    
    #### 10
    p10_q7_bd = (p_pos[:,10:11]-q_pos_7) / distance(p_pos[:,10:11], q_pos_7).unsqueeze(-1)
    q_pos_10 = q_pos_7 + p10_q7_bd * t_pos_bl[:,10:11].unsqueeze(-1)
    

    ### ---------------------Right leg---------------------------------------------
    #### 2
    q_pos_2 = torch.matmul(root_rotmat, t_pos[:,2:3,:].unsqueeze(-1)).squeeze(-1)
  
    #### 5 and 8
    ##### old
    p5_q2_bd = (p_pos[:,5:6]-q_pos_2) / distance(p_pos[:,5:6], q_pos_2).unsqueeze(-1)
    q_pos_5 = q_pos_2 + p5_q2_bd * t_pos_bl[:,5:6].unsqueeze(-1)
 
    p8_q5_bd = (p_pos[:,8:9]-q_pos_5) / distance(p_pos[:,8:9], q_pos_5).unsqueeze(-1)
    q_pos_8 = q_pos_5 + p8_q5_bd * t_pos_bl[:,8:9].unsqueeze(-1)
    
    ##### new 
    q_pos_5, q_pos_8 = itertive_move(q_pos_2, q_pos_5, q_pos_8, p_pos[:, 8:9], t_pos_bl[:, 5:6], t_pos_bl[:,8:9], iter_num=iter_num)
    
    #### 11
    p11_q8_bd = (p_pos[:,11:12]-q_pos_8) / distance(p_pos[:,11:12], q_pos_8).unsqueeze(-1)
    q_pos_11 = q_pos_8 + p11_q8_bd * t_pos_bl[:,11:12].unsqueeze(-1)
   
   
    ### ---------------------Spine---------------------------------------------
    #### 3
    q_pos_3 = torch.matmul(root_rotmat, t_pos[:,3:4,:].unsqueeze(-1)).squeeze(-1)
 
    #### 6 and 9
    ##### old
    p6_q3_bd = (p_pos[:,6:7]-q_pos_3) / distance(p_pos[:,6:7], q_pos_3).unsqueeze(-1)
    q_pos_6 = q_pos_3 + p6_q3_bd * t_pos_bl[:,6:7].unsqueeze(-1)
  
    p9_q6_bd = (p_pos[:,9:10]-q_pos_6) / distance(p_pos[:,9:10], q_pos_6).unsqueeze(-1)
    q_pos_9 = q_pos_6 + p9_q6_bd * t_pos_bl[:,9:10].unsqueeze(-1)
    
    ##### new
    q_pos_6, q_pos_9 = itertive_move(q_pos_3, q_pos_6, q_pos_9, p_pos[:, 9:10], t_pos_bl[:,6:7], t_pos_bl[:,9:10], iter_num=iter_num)
    
    p_pos[:,9:10] = q_pos_9
    vec_pt = p_pos[:, 1:] - p_pos[:, parent[1:]]
    vec_t = t_pos[:, 1:] - t_pos[:, parent[1:]]
    spine3_rotmat = batch_get_neck_orient(vec_pt.unsqueeze(-1), vec_t.unsqueeze(-1), parent[1:24], children, torch.float32).unsqueeze(1)
    
    #### 12
    q_pos_12 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,11:12,:].unsqueeze(-1)).squeeze(-1)
  
    #### 15
    p15_q12_bd = (p_pos[:,15:16]-q_pos_12) / distance(p_pos[:,15:16], q_pos_12).unsqueeze(-1)
    q_pos_15 = q_pos_12 + p15_q12_bd * t_pos_bl[:,15:16].unsqueeze(-1)
    
    ### ---------------------left----------------------------------------------------
    #### 13
    q_pos_13 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 16 and 18 
    ##### old
    p16_q13_bd = (p_pos[:,16:17]-q_pos_13) / distance(p_pos[:,16:17], q_pos_13).unsqueeze(-1)
    q_pos_16 = q_pos_13 + p16_q13_bd * t_pos_bl[:,16:17].unsqueeze(-1)
    
    p18_q16_bd = (p_pos[:,18:19]-q_pos_16) / distance(p_pos[:,18:19], q_pos_16).unsqueeze(-1)
    q_pos_18 = q_pos_16 + p18_q16_bd * t_pos_bl[:,18:19].unsqueeze(-1)
    
    ##### new 
    q_pos_16, q_pos_18 = itertive_move(q_pos_13, q_pos_16, q_pos_18, p_pos[:,18:19], t_pos_bl[:,16:17], t_pos_bl[:, 18:19], iter_num=iter_num)
    
    
    #### 20 and 22
    ##### old
    p20_q18_bd = (p_pos[:,20:21]-q_pos_18) / distance(p_pos[:,20:21], q_pos_18).unsqueeze(-1)
    q_pos_20 = q_pos_18 + p20_q18_bd * t_pos_bl[:,20:21].unsqueeze(-1)
   
    p22_q20_bd = (p_pos[:,22:23]-q_pos_20) / distance(p_pos[:,22:23], q_pos_20).unsqueeze(-1)
    q_pos_22 = q_pos_20 + p22_q20_bd * t_pos_bl[:,22:23].unsqueeze(-1)
    
    ##### new 
    q_pos_20, q_pos_22 = itertive_move(q_pos_18, q_pos_20, q_pos_22, p_pos[:,22:23], t_pos_bl[:,20:21], t_pos_bl[:, 22:23], iter_num=iter_num)
    
    ### ---------------------right----------------------------------------------------
    #### 14
    q_pos_14 = q_pos_9 + torch.matmul(spine3_rotmat, vec_t[:,13:14,:].unsqueeze(-1)).squeeze(-1)
  
    #### 17 and 19
    ##### old
    p17_q14_bd = (p_pos[:,17:18]-q_pos_14) / distance(p_pos[:,17:18], q_pos_14).unsqueeze(-1)
    q_pos_17 = q_pos_14 + p17_q14_bd * t_pos_bl[:,17:18].unsqueeze(-1)
  
    p19_q17_bd = (p_pos[:,19:20]-q_pos_17) / distance(p_pos[:,19:20], q_pos_17).unsqueeze(-1)
    q_pos_19 = q_pos_17 + p19_q17_bd * t_pos_bl[:,19:20].unsqueeze(-1)
    
    ##### new 
    q_pos_17, q_pos_19 = itertive_move(q_pos_14, q_pos_17, q_pos_19, p_pos[:,19:20], t_pos_bl[:,17:18], t_pos_bl[:, 19:20], iter_num=iter_num)
    
    #### 21 and 23
    ##### old
    p21_q19_bd = (p_pos[:,21:22]-q_pos_19) / distance(p_pos[:,21:22], q_pos_19).unsqueeze(-1)
    q_pos_21 = q_pos_19 + p21_q19_bd * t_pos_bl[:,21:22].unsqueeze(-1)
    
    p23_q21_bd = (p_pos[:,23:24]-q_pos_21) / distance(p_pos[:,23:24], q_pos_21).unsqueeze(-1)
    q_pos_23 = q_pos_21 + p23_q21_bd * t_pos_bl[:,23:24].unsqueeze(-1)
    
    ##### new 
    q_pos_21, q_pos_23 = itertive_move(q_pos_19, q_pos_21, q_pos_23, p_pos[:,23:24], t_pos_bl[:,21:22], t_pos_bl[:, 23:24], iter_num=iter_num)
    
    q_pos_list = [q_pos_0, q_pos_1, q_pos_2, q_pos_3, q_pos_4, q_pos_5, q_pos_6, q_pos_7, q_pos_8, q_pos_9,
              q_pos_10, q_pos_11, q_pos_12, q_pos_13, q_pos_14, q_pos_15, q_pos_16, q_pos_17, q_pos_18, q_pos_19, q_pos_20,
              q_pos_21, q_pos_22, q_pos_23]
    
    q_pos = torch.cat(q_pos_list, dim=1)

    return q_pos

