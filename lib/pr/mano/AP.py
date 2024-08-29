# -*- coding: utf-8 -*-

import torch

from lib.utils.ik_utils import  batch_get_wrist_orient, vectors2rotmat_bk

from lib.utils.si_utils import get_bl_from_pos, distance

def MANO_AP_V1(t_pos, p_pos, parent, children):

    device = p_pos.device
    
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # bl/bd
    t_pos_bl = get_bl_from_pos(t_pos, parent)

    ## 0
    q_pos_0 = torch.zeros([t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    wrist_rotmat = batch_get_wrist_orient(p_pos.unsqueeze(-1), t_pos.unsqueeze(-1), parent, children, torch.float32).unsqueeze(1) # [b,1,1,3,3]
   
    ## ---------------------Thumb-------------------------------------------
    ### 1
    q_pos_1 = torch.matmul(wrist_rotmat, t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2
    p2_q1_bd = (p_pos[:,2:3]-q_pos_1) / distance(p_pos[:,2:3], q_pos_1).unsqueeze(-1)
    q_pos_2 = q_pos_1 + p2_q1_bd * t_pos_bl[:,2:3].unsqueeze(-1)
    
    ### 3
    p3_q2_bd = (p_pos[:,3:4]-q_pos_2) / distance(p_pos[:,3:4], q_pos_2).unsqueeze(-1)
    q_pos_3 = q_pos_2 + p3_q2_bd * t_pos_bl[:,3:4].unsqueeze(-1)
  
    ### 4
    p4_q3_bd = (p_pos[:,4:5]-q_pos_3) / distance(p_pos[:,4:5], q_pos_3).unsqueeze(-1)
    q_pos_4 = q_pos_3 + p4_q3_bd * t_pos_bl[:,4:5].unsqueeze(-1)
    

    ### ---------------------Index---------------------------------------------
    #### 5
    q_pos_5 = torch.matmul(wrist_rotmat, t_pos[:,5:6,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6
    p6_q5_bd = (p_pos[:,6:7]-q_pos_5) / distance(p_pos[:,6:7], q_pos_5).unsqueeze(-1)
    q_pos_6 = q_pos_5 + p6_q5_bd * t_pos_bl[:,6:7].unsqueeze(-1)
 
    #### 7
    p7_q6_bd = (p_pos[:,7:8]-q_pos_6) / distance(p_pos[:,7:8], q_pos_6).unsqueeze(-1)
    q_pos_7 = q_pos_6 + p7_q6_bd * t_pos_bl[:,7:8].unsqueeze(-1)
 
    #### 8
    p8_q7_bd = (p_pos[:,8:9]-q_pos_7) / distance(p_pos[:,8:9], q_pos_7).unsqueeze(-1)
    q_pos_8 = q_pos_7 + p8_q7_bd * t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### ---------------------Middle---------------------------------------------
    #### 9
    q_pos_9 = torch.matmul(wrist_rotmat, t_pos[:,9:10,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10
    p10_q9_bd = (p_pos[:,10:11]-q_pos_9) / distance(p_pos[:,10:11], q_pos_9).unsqueeze(-1)
    q_pos_10 = q_pos_9 + p10_q9_bd * t_pos_bl[:,10:11].unsqueeze(-1)
  
    #### 11
    p11_q10_bd = (p_pos[:,11:12]-q_pos_10) / distance(p_pos[:,11:12], q_pos_10).unsqueeze(-1)
    q_pos_11 = q_pos_10 + p11_q10_bd * t_pos_bl[:,11:12].unsqueeze(-1)
 
    #### 12
    p12_q11_bd = (p_pos[:,12:13]-q_pos_11) / distance(p_pos[:,12:13], q_pos_11).unsqueeze(-1)
    q_pos_12 = q_pos_11 + p12_q11_bd * t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    q_pos_13 = torch.matmul(wrist_rotmat, t_pos[:,13:14,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14
    p14_q13_bd = (p_pos[:,14:15]-q_pos_13) / distance(p_pos[:,14:15], q_pos_13).unsqueeze(-1)
    q_pos_14 = q_pos_13 + p14_q13_bd * t_pos_bl[:,14:15].unsqueeze(-1)
    
    #### 15
    p15_q14_bd = (p_pos[:,15:16]-q_pos_14) / distance(p_pos[:,15:16], q_pos_14).unsqueeze(-1)
    q_pos_15 = q_pos_14 + p15_q14_bd * t_pos_bl[:,15:16].unsqueeze(-1)
   
    #### 16
    p16_q15_bd = (p_pos[:,16:17]-q_pos_15) / distance(p_pos[:,16:17], q_pos_15).unsqueeze(-1)
    q_pos_16 = q_pos_15 + p16_q15_bd * t_pos_bl[:,16:17].unsqueeze(-1)
   
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    q_pos_17 = torch.matmul(wrist_rotmat, t_pos[:,17:18,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18
    p18_q17_bd = (p_pos[:,18:19]-q_pos_17) / distance(p_pos[:,18:19], q_pos_17).unsqueeze(-1)
    q_pos_18 = q_pos_17 + p18_q17_bd * t_pos_bl[:,18:19].unsqueeze(-1)
    
    #### 19
    p19_q18_bd = (p_pos[:,19:20]-q_pos_18) / distance(p_pos[:,19:20], q_pos_18).unsqueeze(-1)
    q_pos_19 = q_pos_18 + p19_q18_bd * t_pos_bl[:,19:20].unsqueeze(-1)
   
    #### 20
    p20_q19_bd = (p_pos[:,20:21]-q_pos_19) / distance(p_pos[:,20:21], q_pos_19).unsqueeze(-1)
    q_pos_20 = q_pos_19 + p20_q19_bd * t_pos_bl[:,20:21].unsqueeze(-1)
  
    
    q_pos_list = [q_pos_0, q_pos_1, q_pos_2, q_pos_3, q_pos_4, q_pos_5, q_pos_6, q_pos_7, q_pos_8, q_pos_9,
              q_pos_10, q_pos_11, q_pos_12, q_pos_13, q_pos_14, q_pos_15, q_pos_16, q_pos_17, q_pos_18, q_pos_19, q_pos_20]
    
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


def MANO_AP_V2(t_pos, p_pos, parent, children, iter_num=3):

    device = p_pos.device
    
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # bl/bd
    t_pos_bl = get_bl_from_pos(t_pos, parent)

    ## 0
    q_pos_0 = torch.zeros([t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    wrist_rotmat = batch_get_wrist_orient(p_pos.unsqueeze(-1), t_pos.unsqueeze(-1), parent, children, torch.float32).unsqueeze(1) # [b,1,1,3,3]
   
    ## ---------------------Thumb-------------------------------------------
    ### 1
    q_pos_1 = torch.matmul(wrist_rotmat, t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2 and 3
    #### old
    p2_q1_bd = (p_pos[:,2:3]-q_pos_1) / distance(p_pos[:,2:3], q_pos_1).unsqueeze(-1)
    q_pos_2 = q_pos_1 + p2_q1_bd * t_pos_bl[:,2:3].unsqueeze(-1)
    
    p3_q2_bd = (p_pos[:,3:4]-q_pos_2) / distance(p_pos[:,3:4], q_pos_2).unsqueeze(-1)
    q_pos_3 = q_pos_2 + p3_q2_bd * t_pos_bl[:,3:4].unsqueeze(-1)
  
    #### new
    q_pos_2, q_pos_3 = itertive_move(q_pos_1, q_pos_2, q_pos_3, p_pos[:,3:4], t_pos_bl[:, 2:3], t_pos_bl[:, 3:4], iter_num=iter_num)
  
    ### 4
    p4_q3_bd = (p_pos[:,4:5]-q_pos_3) / distance(p_pos[:,4:5], q_pos_3).unsqueeze(-1)
    q_pos_4 = q_pos_3 + p4_q3_bd * t_pos_bl[:,4:5].unsqueeze(-1)
    
    
    ### ---------------------Index---------------------------------------------
    #### 5
    q_pos_5 = torch.matmul(wrist_rotmat, t_pos[:,5:6,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6 and 7
    ##### old
    p6_q5_bd = (p_pos[:,6:7]-q_pos_5) / distance(p_pos[:,6:7], q_pos_5).unsqueeze(-1)
    q_pos_6 = q_pos_5 + p6_q5_bd * t_pos_bl[:,6:7].unsqueeze(-1)
 
    p7_q6_bd = (p_pos[:,7:8]-q_pos_6) / distance(p_pos[:,7:8], q_pos_6).unsqueeze(-1)
    q_pos_7 = q_pos_6 + p7_q6_bd * t_pos_bl[:,7:8].unsqueeze(-1)
 
    ##### new
    q_pos_6, q_pos_7 = itertive_move(q_pos_5, q_pos_6, q_pos_7, p_pos[:,7:8], t_pos_bl[:, 6:7], t_pos_bl[:, 7:8], iter_num=iter_num)
 
    #### 8
    p8_q7_bd = (p_pos[:,8:9]-q_pos_7) / distance(p_pos[:,8:9], q_pos_7).unsqueeze(-1)
    q_pos_8 = q_pos_7 + p8_q7_bd * t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### ---------------------Middle---------------------------------------------
    #### 9
    q_pos_9 = torch.matmul(wrist_rotmat, t_pos[:,9:10,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10 and 11
    ##### old
    p10_q9_bd = (p_pos[:,10:11]-q_pos_9) / distance(p_pos[:,10:11], q_pos_9).unsqueeze(-1)
    q_pos_10 = q_pos_9 + p10_q9_bd * t_pos_bl[:,10:11].unsqueeze(-1)

    p11_q10_bd = (p_pos[:,11:12]-q_pos_10) / distance(p_pos[:,11:12], q_pos_10).unsqueeze(-1)
    q_pos_11 = q_pos_10 + p11_q10_bd * t_pos_bl[:,11:12].unsqueeze(-1)
 
    ##### new
    q_pos_10, q_pos_11 = itertive_move(q_pos_9, q_pos_10, q_pos_11, p_pos[:,11:12], t_pos_bl[:, 10:11], t_pos_bl[:, 11:12], iter_num=iter_num)
 
    #### 12
    p12_q11_bd = (p_pos[:,12:13]-q_pos_11) / distance(p_pos[:,12:13], q_pos_11).unsqueeze(-1)
    q_pos_12 = q_pos_11 + p12_q11_bd * t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    q_pos_13 = torch.matmul(wrist_rotmat, t_pos[:,13:14,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14 nad 15
    ##### old
    p14_q13_bd = (p_pos[:,14:15]-q_pos_13) / distance(p_pos[:,14:15], q_pos_13).unsqueeze(-1)
    q_pos_14 = q_pos_13 + p14_q13_bd * t_pos_bl[:,14:15].unsqueeze(-1)
    
    p15_q14_bd = (p_pos[:,15:16]-q_pos_14) / distance(p_pos[:,15:16], q_pos_14).unsqueeze(-1)
    q_pos_15 = q_pos_14 + p15_q14_bd * t_pos_bl[:,15:16].unsqueeze(-1)
    
    ##### new
    q_pos_14, q_pos_15 = itertive_move(q_pos_13, q_pos_14, q_pos_15, p_pos[:,15:16], t_pos_bl[:, 14:15], t_pos_bl[:, 15:16], iter_num=iter_num)
    
    #### 16
    p16_q15_bd = (p_pos[:,16:17]-q_pos_15) / distance(p_pos[:,16:17], q_pos_15).unsqueeze(-1)
    q_pos_16 = q_pos_15 + p16_q15_bd * t_pos_bl[:,16:17].unsqueeze(-1)
   
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    q_pos_17 = torch.matmul(wrist_rotmat, t_pos[:,17:18,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18 and 19
    ##### old
    p18_q17_bd = (p_pos[:,18:19]-q_pos_17) / distance(p_pos[:,18:19], q_pos_17).unsqueeze(-1)
    q_pos_18 = q_pos_17 + p18_q17_bd * t_pos_bl[:,18:19].unsqueeze(-1)
    
    p19_q18_bd = (p_pos[:,19:20]-q_pos_18) / distance(p_pos[:,19:20], q_pos_18).unsqueeze(-1)
    q_pos_19 = q_pos_18 + p19_q18_bd * t_pos_bl[:,19:20].unsqueeze(-1)
   
    ##### new
    q_pos_18, q_pos_19 = itertive_move(q_pos_17, q_pos_18, q_pos_19, p_pos[:,19:20], t_pos_bl[:, 18:19], t_pos_bl[:, 19:20], iter_num=iter_num)
   
    #### 20
    p20_q19_bd = (p_pos[:,20:21]-q_pos_19) / distance(p_pos[:,20:21], q_pos_19).unsqueeze(-1)
    q_pos_20 = q_pos_19 + p20_q19_bd * t_pos_bl[:,20:21].unsqueeze(-1)
  
    
    q_pos_list = [q_pos_0, q_pos_1, q_pos_2, q_pos_3, q_pos_4, q_pos_5, q_pos_6, q_pos_7, q_pos_8, q_pos_9,
              q_pos_10, q_pos_11, q_pos_12, q_pos_13, q_pos_14, q_pos_15, q_pos_16, q_pos_17, q_pos_18, q_pos_19, q_pos_20]
    
    q_pos = torch.cat(q_pos_list, dim=1)

    return q_pos