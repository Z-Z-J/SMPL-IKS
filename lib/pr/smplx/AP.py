# -*- coding: utf-8 -*-
import torch

from lib.utils.ik_utils import batch_get_pelvis_orient, batch_get_neck_orient, batch_get_wrist_orient, vectors2rotmat_bk

from lib.utils.si_utils import get_bl_from_pos, distance

def SMPLX_AP_V1(t_pos, p_pos):

    device = p_pos.device
    
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # -------------------------Body-AP-----------------------------------------
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    # bl/bd
    body_t_pos_bl = get_bl_from_pos(body_t_pos, body_parent)

    ## 0
    body_q_pos_0 = torch.zeros([body_t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    root_rotmat = batch_get_pelvis_orient(body_p_pos.unsqueeze(-1), body_t_pos.unsqueeze(-1), body_parent[1:24], body_children, torch.float32).unsqueeze(1)  # [b,1,1,3,3]
   
    ## ---------------------left leg-------------------------------------------
    ### 1
    body_q_pos_1 = torch.matmul(root_rotmat, body_t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 4
    body_p4_q1_bd = (body_p_pos[:,4:5]-body_q_pos_1) / distance(body_p_pos[:,4:5], body_q_pos_1).unsqueeze(-1)
    body_q_pos_4 = body_q_pos_1 + body_p4_q1_bd * body_t_pos_bl[:,4:5].unsqueeze(-1)
    
    ### 7
    body_p7_q4_bd = (body_p_pos[:,7:8]-body_q_pos_4) / distance(body_p_pos[:,7:8], body_q_pos_4).unsqueeze(-1)
    body_q_pos_7 = body_q_pos_4 + body_p7_q4_bd * body_t_pos_bl[:,7:8].unsqueeze(-1)
  
    #### 10
    body_p10_q7_bd = (body_p_pos[:,10:11]-body_q_pos_7) / distance(body_p_pos[:,10:11], body_q_pos_7).unsqueeze(-1)
    body_q_pos_10 = body_q_pos_7 + body_p10_q7_bd * body_t_pos_bl[:,10:11].unsqueeze(-1)
    

    ### ---------------------Right leg---------------------------------------------
    #### 2
    body_q_pos_2 = torch.matmul(root_rotmat, body_t_pos[:,2:3,:].unsqueeze(-1)).squeeze(-1)
  
    #### 5
    body_p5_q2_bd = (body_p_pos[:,5:6]-body_q_pos_2) / distance(body_p_pos[:,5:6], body_q_pos_2).unsqueeze(-1)
    body_q_pos_5 = body_q_pos_2 + body_p5_q2_bd * body_t_pos_bl[:,5:6].unsqueeze(-1)
 
    #### 8
    body_p8_q5_bd = (body_p_pos[:,8:9]-body_q_pos_5) / distance(body_p_pos[:,8:9], body_q_pos_5).unsqueeze(-1)
    body_q_pos_8 = body_q_pos_5 + body_p8_q5_bd * body_t_pos_bl[:,8:9].unsqueeze(-1)
 
    #### 11
    body_p11_q8_bd = (body_p_pos[:,11:12]-body_q_pos_8) / distance(body_p_pos[:,11:12], body_q_pos_8).unsqueeze(-1)
    body_q_pos_11 = body_q_pos_8 + body_p11_q8_bd * body_t_pos_bl[:,11:12].unsqueeze(-1)
   
   
    ### ---------------------Spine---------------------------------------------
    #### 3
    body_q_pos_3 = torch.matmul(root_rotmat, body_t_pos[:,3:4,:].unsqueeze(-1)).squeeze(-1)
 
    #### 6
    body_p6_q3_bd = (body_p_pos[:,6:7]-body_q_pos_3) / distance(body_p_pos[:,6:7], body_q_pos_3).unsqueeze(-1)
    body_q_pos_6 = body_q_pos_3 + body_p6_q3_bd * body_t_pos_bl[:,6:7].unsqueeze(-1)
  
    #### 9
    body_p9_q6_bd = (body_p_pos[:,9:10]-body_q_pos_6) / distance(body_p_pos[:,9:10], body_q_pos_6).unsqueeze(-1)
    body_q_pos_9 = body_q_pos_6 + body_p9_q6_bd * body_t_pos_bl[:,9:10].unsqueeze(-1)

    body_p_pos[:,9:10] = body_q_pos_9
    body_vec_pt = body_p_pos[:, 1:] - body_p_pos[:, body_parent[1:]]
    body_vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    spine3_rotmat = batch_get_neck_orient(body_vec_pt.unsqueeze(-1), body_vec_t.unsqueeze(-1), body_parent[1:24], body_children, torch.float32).unsqueeze(1)
    
    #### 12
    body_q_pos_12 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,11:12,:].unsqueeze(-1)).squeeze(-1)
  
    #### 15
    body_p15_q12_bd = (body_p_pos[:,15:16]-body_q_pos_12) / distance(body_p_pos[:,15:16], body_q_pos_12).unsqueeze(-1)
    body_q_pos_15 = body_q_pos_12 + body_p15_q12_bd * body_t_pos_bl[:,15:16].unsqueeze(-1)
    

    ### ---------------------left----------------------------------------------------
    #### 13
    body_q_pos_13 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 16
    body_p16_q13_bd = (body_p_pos[:,16:17]-body_q_pos_13) / distance(body_p_pos[:,16:17], body_q_pos_13).unsqueeze(-1)
    body_q_pos_16 = body_q_pos_13 + body_p16_q13_bd * body_t_pos_bl[:,16:17].unsqueeze(-1)
    
    #### 18
    body_p18_q16_bd = (body_p_pos[:,18:19]-body_q_pos_16) / distance(body_p_pos[:,18:19], body_q_pos_16).unsqueeze(-1)
    body_q_pos_18 = body_q_pos_16 + body_p18_q16_bd * body_t_pos_bl[:,18:19].unsqueeze(-1)
   
    #### 20
    body_p20_q18_bd = (body_p_pos[:,20:21]-body_q_pos_18) / distance(body_p_pos[:,20:21], body_q_pos_18).unsqueeze(-1)
    body_q_pos_20 = body_q_pos_18 + body_p20_q18_bd * body_t_pos_bl[:,20:21].unsqueeze(-1)
   

    ### ---------------------right----------------------------------------------------
    #### 14
    body_q_pos_14 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,13:14,:].unsqueeze(-1)).squeeze(-1)
  
    #### 17
    body_p17_q14_bd = (body_p_pos[:,17:18]-body_q_pos_14) / distance(body_p_pos[:,17:18], body_q_pos_14).unsqueeze(-1)
    body_q_pos_17 = body_q_pos_14 + body_p17_q14_bd * body_t_pos_bl[:,17:18].unsqueeze(-1)
  
    #### 19
    body_p19_q17_bd = (body_p_pos[:,19:20]-body_q_pos_17) / distance(body_p_pos[:,19:20], body_q_pos_17).unsqueeze(-1)
    body_q_pos_19 = body_q_pos_17 + body_p19_q17_bd * body_t_pos_bl[:,19:20].unsqueeze(-1)
  
    #### 21
    body_p21_q19_bd = (body_p_pos[:,21:22]-body_q_pos_19) / distance(body_p_pos[:,21:22], body_q_pos_19).unsqueeze(-1)
    body_q_pos_21 = body_q_pos_19 + body_p21_q19_bd * body_t_pos_bl[:,21:22].unsqueeze(-1)
  

    body_q_pos_list = [body_q_pos_0, body_q_pos_1, body_q_pos_2, body_q_pos_3, body_q_pos_4, body_q_pos_5, body_q_pos_6, body_q_pos_7, body_q_pos_8, body_q_pos_9,
              body_q_pos_10, body_q_pos_11, body_q_pos_12, body_q_pos_13, body_q_pos_14, body_q_pos_15, body_q_pos_16, body_q_pos_17, body_q_pos_18, body_q_pos_19, 
              body_q_pos_20, body_q_pos_21]
    
    body_q_pos = torch.cat(body_q_pos_list, dim=1)
    
    
    # -------------------------Lhand-IK-----------------------------------------
    lhand_index = torch.tensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70])
    lhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    lhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    lhand_t_pos = t_pos[:,lhand_index]
    lhand_p_pos = p_pos[:,lhand_index]
    
    # bl/bd
    lhand_t_pos_bl = get_bl_from_pos(lhand_t_pos, lhand_parent)
    
    lhand_p_pos[:,0:1] = body_q_pos_20
    lhand_vec_pt = lhand_p_pos[:, 1:] - lhand_p_pos[:, lhand_parent[1:]]
    lhand_vec_t = lhand_t_pos[:, 1:] - lhand_t_pos[:, lhand_parent[1:]]
    lhand_wrist_rotmat = batch_get_wrist_orient(lhand_vec_pt.unsqueeze(-1), lhand_vec_t.unsqueeze(-1), lhand_parent, lhand_children, torch.float32).unsqueeze(1)

    ## ---------------------Thumb-------------------------------------------
    ### 1
    lhand_q_pos_1 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,0:1,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2
    lhand_p2_q1_bd = (lhand_p_pos[:,2:3]-lhand_q_pos_1) / distance(lhand_p_pos[:,2:3], lhand_q_pos_1).unsqueeze(-1)
    lhand_q_pos_2 = lhand_q_pos_1 + lhand_p2_q1_bd * lhand_t_pos_bl[:,2:3].unsqueeze(-1)
    
    ### 3
    lhand_p3_q2_bd = (lhand_p_pos[:,3:4]-lhand_q_pos_2) / distance(lhand_p_pos[:,3:4], lhand_q_pos_2).unsqueeze(-1)
    lhand_q_pos_3 = lhand_q_pos_2 + lhand_p3_q2_bd * lhand_t_pos_bl[:,3:4].unsqueeze(-1)
  
    ### 4
    lhand_p4_q3_bd = (lhand_p_pos[:,4:5]-lhand_q_pos_3) / distance(lhand_p_pos[:,4:5], lhand_q_pos_3).unsqueeze(-1)
    lhand_q_pos_4 = lhand_q_pos_3 + lhand_p4_q3_bd * lhand_t_pos_bl[:,4:5].unsqueeze(-1)
    

    ### ---------------------Index---------------------------------------------
    #### 5
    lhand_q_pos_5 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,4:5,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6
    lhand_p6_q5_bd = (lhand_p_pos[:,6:7]-lhand_q_pos_5) / distance(lhand_p_pos[:,6:7], lhand_q_pos_5).unsqueeze(-1)
    lhand_q_pos_6 = lhand_q_pos_5 + lhand_p6_q5_bd * lhand_t_pos_bl[:,6:7].unsqueeze(-1)
 
    #### 7
    lhand_p7_q6_bd = (lhand_p_pos[:,7:8]-lhand_q_pos_6) / distance(lhand_p_pos[:,7:8], lhand_q_pos_6).unsqueeze(-1)
    lhand_q_pos_7 = lhand_q_pos_6 + lhand_p7_q6_bd * lhand_t_pos_bl[:,7:8].unsqueeze(-1)
 
    #### 8
    lhand_p8_q7_bd = (lhand_p_pos[:,8:9]-lhand_q_pos_7) / distance(lhand_p_pos[:,8:9], lhand_q_pos_7).unsqueeze(-1)
    lhand_q_pos_8 = lhand_q_pos_7 + lhand_p8_q7_bd * lhand_t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### ---------------------Middle---------------------------------------------
    #### 9
    lhand_q_pos_9 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,8:9,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10
    lhand_p10_q9_bd = (lhand_p_pos[:,10:11]-lhand_q_pos_9) / distance(lhand_p_pos[:,10:11], lhand_q_pos_9).unsqueeze(-1)
    lhand_q_pos_10 = lhand_q_pos_9 + lhand_p10_q9_bd * lhand_t_pos_bl[:,10:11].unsqueeze(-1)
  
    #### 11
    lhand_p11_q10_bd = (lhand_p_pos[:,11:12]-lhand_q_pos_10) / distance(lhand_p_pos[:,11:12], lhand_q_pos_10).unsqueeze(-1)
    lhand_q_pos_11 = lhand_q_pos_10 + lhand_p11_q10_bd * lhand_t_pos_bl[:,11:12].unsqueeze(-1)
 
    #### 12
    lhand_p12_q11_bd = (lhand_p_pos[:,12:13]-lhand_q_pos_11) / distance(lhand_p_pos[:,12:13], lhand_q_pos_11).unsqueeze(-1)
    lhand_q_pos_12 = lhand_q_pos_11 + lhand_p12_q11_bd * lhand_t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    lhand_q_pos_13 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14
    lhand_p14_q13_bd = (lhand_p_pos[:,14:15]-lhand_q_pos_13) / distance(lhand_p_pos[:,14:15], lhand_q_pos_13).unsqueeze(-1)
    lhand_q_pos_14 = lhand_q_pos_13 + lhand_p14_q13_bd * lhand_t_pos_bl[:,14:15].unsqueeze(-1)
    
    #### 15
    lhand_p15_q14_bd = (lhand_p_pos[:,15:16]-lhand_q_pos_14) / distance(lhand_p_pos[:,15:16], lhand_q_pos_14).unsqueeze(-1)
    lhand_q_pos_15 = lhand_q_pos_14 + lhand_p15_q14_bd * lhand_t_pos_bl[:,15:16].unsqueeze(-1)
   
    #### 16
    lhand_p16_q15_bd = (lhand_p_pos[:,16:17]-lhand_q_pos_15) / distance(lhand_p_pos[:,16:17], lhand_q_pos_15).unsqueeze(-1)
    lhand_q_pos_16 = lhand_q_pos_15 + lhand_p16_q15_bd * lhand_t_pos_bl[:,16:17].unsqueeze(-1)
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    lhand_q_pos_17 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,16:17,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18
    lhand_p18_q17_bd = (lhand_p_pos[:,18:19]-lhand_q_pos_17) / distance(lhand_p_pos[:,18:19], lhand_q_pos_17).unsqueeze(-1)
    lhand_q_pos_18 = lhand_q_pos_17 + lhand_p18_q17_bd * lhand_t_pos_bl[:,18:19].unsqueeze(-1)
    
    #### 19
    lhand_p19_q18_bd = (lhand_p_pos[:,19:20]-lhand_q_pos_18) / distance(lhand_p_pos[:,19:20], lhand_q_pos_18).unsqueeze(-1)
    lhand_q_pos_19 = lhand_q_pos_18 + lhand_p19_q18_bd * lhand_t_pos_bl[:,19:20].unsqueeze(-1)
   
    #### 20
    lhand_p20_q19_bd = (lhand_p_pos[:,20:21]-lhand_q_pos_19) / distance(lhand_p_pos[:,20:21], lhand_q_pos_19).unsqueeze(-1)
    lhand_q_pos_20 = lhand_q_pos_19 + lhand_p20_q19_bd * lhand_t_pos_bl[:,20:21].unsqueeze(-1)    

    lhand_q_pos_list = [lhand_q_pos_1, lhand_q_pos_2, lhand_q_pos_3, lhand_q_pos_4, lhand_q_pos_5, lhand_q_pos_6, lhand_q_pos_7, lhand_q_pos_8, lhand_q_pos_9,
              lhand_q_pos_10, lhand_q_pos_11, lhand_q_pos_12, lhand_q_pos_13, lhand_q_pos_14, lhand_q_pos_15, lhand_q_pos_16, lhand_q_pos_17, 
              lhand_q_pos_18, lhand_q_pos_19, lhand_q_pos_20]
    
    lhand_q_pos = torch.cat(lhand_q_pos_list, dim=1)


    # -------------------------Rhand-IK-----------------------------------------
    rhand_index = torch.tensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
    rhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    rhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    rhand_t_pos = t_pos[:,rhand_index]
    rhand_p_pos = p_pos[:,rhand_index]
    
    # bl/bd
    rhand_t_pos_bl = get_bl_from_pos(rhand_t_pos, rhand_parent)
    
    rhand_p_pos[:,0:1] = body_q_pos_21
    rhand_vec_pt = rhand_p_pos[:, 1:] - rhand_p_pos[:, rhand_parent[1:]]
    rhand_vec_t = rhand_t_pos[:, 1:] - rhand_t_pos[:, rhand_parent[1:]]
    rhand_wrist_rotmat = batch_get_wrist_orient(rhand_vec_pt.unsqueeze(-1), rhand_vec_t.unsqueeze(-1), rhand_parent, rhand_children, torch.float32).unsqueeze(1)

    ## ---------------------Thumb-------------------------------------------
    ### 1
    rhand_q_pos_1 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,0:1,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2
    rhand_p2_q1_bd = (rhand_p_pos[:,2:3]-rhand_q_pos_1) / distance(rhand_p_pos[:,2:3], rhand_q_pos_1).unsqueeze(-1)
    rhand_q_pos_2 = rhand_q_pos_1 + rhand_p2_q1_bd * rhand_t_pos_bl[:,2:3].unsqueeze(-1)
    
    ### 3
    rhand_p3_q2_bd = (rhand_p_pos[:,3:4]-rhand_q_pos_2) / distance(rhand_p_pos[:,3:4], rhand_q_pos_2).unsqueeze(-1)
    rhand_q_pos_3 = rhand_q_pos_2 + rhand_p3_q2_bd * rhand_t_pos_bl[:,3:4].unsqueeze(-1)
  
    ### 4
    rhand_p4_q3_bd = (rhand_p_pos[:,4:5]-rhand_q_pos_3) / distance(rhand_p_pos[:,4:5], rhand_q_pos_3).unsqueeze(-1)
    rhand_q_pos_4 = rhand_q_pos_3 + rhand_p4_q3_bd * rhand_t_pos_bl[:,4:5].unsqueeze(-1)
    

    ### ---------------------Index---------------------------------------------
    #### 5
    rhand_q_pos_5 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,4:5,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6
    rhand_p6_q5_bd = (rhand_p_pos[:,6:7]-rhand_q_pos_5) / distance(rhand_p_pos[:,6:7], rhand_q_pos_5).unsqueeze(-1)
    rhand_q_pos_6 = rhand_q_pos_5 + rhand_p6_q5_bd * rhand_t_pos_bl[:,6:7].unsqueeze(-1)
 
    #### 7
    rhand_p7_q6_bd = (rhand_p_pos[:,7:8]-rhand_q_pos_6) / distance(rhand_p_pos[:,7:8], rhand_q_pos_6).unsqueeze(-1)
    rhand_q_pos_7 = rhand_q_pos_6 + rhand_p7_q6_bd * rhand_t_pos_bl[:,7:8].unsqueeze(-1)
 
    #### 8
    rhand_p8_q7_bd = (rhand_p_pos[:,8:9]-rhand_q_pos_7) / distance(rhand_p_pos[:,8:9], rhand_q_pos_7).unsqueeze(-1)
    rhand_q_pos_8 = rhand_q_pos_7 + rhand_p8_q7_bd * rhand_t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### ---------------------Middle---------------------------------------------
    #### 9
    rhand_q_pos_9 = body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,8:9,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10
    rhand_p10_q9_bd = (rhand_p_pos[:,10:11]-rhand_q_pos_9) / distance(rhand_p_pos[:,10:11], rhand_q_pos_9).unsqueeze(-1)
    rhand_q_pos_10 = rhand_q_pos_9 + rhand_p10_q9_bd * rhand_t_pos_bl[:,10:11].unsqueeze(-1)
  
    #### 11
    rhand_p11_q10_bd = (rhand_p_pos[:,11:12]-rhand_q_pos_10) / distance(rhand_p_pos[:,11:12], rhand_q_pos_10).unsqueeze(-1)
    rhand_q_pos_11 = rhand_q_pos_10 + rhand_p11_q10_bd * rhand_t_pos_bl[:,11:12].unsqueeze(-1)
 
    #### 12
    rhand_p12_q11_bd = (rhand_p_pos[:,12:13]-rhand_q_pos_11) / distance(rhand_p_pos[:,12:13], rhand_q_pos_11).unsqueeze(-1)
    rhand_q_pos_12 = rhand_q_pos_11 + rhand_p12_q11_bd * rhand_t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    rhand_q_pos_13 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14
    rhand_p14_q13_bd = (rhand_p_pos[:,14:15]-rhand_q_pos_13) / distance(rhand_p_pos[:,14:15], rhand_q_pos_13).unsqueeze(-1)
    rhand_q_pos_14 = rhand_q_pos_13 + rhand_p14_q13_bd * rhand_t_pos_bl[:,14:15].unsqueeze(-1)
    
    #### 15
    rhand_p15_q14_bd = (rhand_p_pos[:,15:16]-rhand_q_pos_14) / distance(rhand_p_pos[:,15:16], rhand_q_pos_14).unsqueeze(-1)
    rhand_q_pos_15 = rhand_q_pos_14 + rhand_p15_q14_bd * rhand_t_pos_bl[:,15:16].unsqueeze(-1)
   
    #### 16
    rhand_p16_q15_bd = (rhand_p_pos[:,16:17]-rhand_q_pos_15) / distance(rhand_p_pos[:,16:17], rhand_q_pos_15).unsqueeze(-1)
    rhand_q_pos_16 = rhand_q_pos_15 + rhand_p16_q15_bd * rhand_t_pos_bl[:,16:17].unsqueeze(-1)
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    rhand_q_pos_17 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,16:17,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18
    rhand_p18_q17_bd = (rhand_p_pos[:,18:19]-rhand_q_pos_17) / distance(rhand_p_pos[:,18:19], rhand_q_pos_17).unsqueeze(-1)
    rhand_q_pos_18 = rhand_q_pos_17 + rhand_p18_q17_bd * rhand_t_pos_bl[:,18:19].unsqueeze(-1)
    
    #### 19
    rhand_p19_q18_bd = (rhand_p_pos[:,19:20]-rhand_q_pos_18) / distance(rhand_p_pos[:,19:20], rhand_q_pos_18).unsqueeze(-1)
    rhand_q_pos_19 = rhand_q_pos_18 + rhand_p19_q18_bd * rhand_t_pos_bl[:,19:20].unsqueeze(-1)
   
    #### 20
    rhand_p20_q19_bd = (rhand_p_pos[:,20:21]-rhand_q_pos_19) / distance(rhand_p_pos[:,20:21], rhand_q_pos_19).unsqueeze(-1)
    rhand_q_pos_20 = rhand_q_pos_19 + rhand_p20_q19_bd * rhand_t_pos_bl[:,20:21].unsqueeze(-1)    

    rhand_q_pos_list = [rhand_q_pos_1, rhand_q_pos_2, rhand_q_pos_3, rhand_q_pos_4, rhand_q_pos_5, rhand_q_pos_6, rhand_q_pos_7, rhand_q_pos_8, rhand_q_pos_9,
              rhand_q_pos_10, rhand_q_pos_11, rhand_q_pos_12, rhand_q_pos_13, rhand_q_pos_14, rhand_q_pos_15, rhand_q_pos_16, rhand_q_pos_17, 
              rhand_q_pos_18, rhand_q_pos_19, rhand_q_pos_20]
    
    rhand_q_pos = torch.cat(rhand_q_pos_list, dim=1)

    q_pos = p_pos.clone()
    q_pos[:,body_index] = body_q_pos
    q_pos[:,lhand_index[1:]] = lhand_q_pos
    q_pos[:,rhand_index[1:]] = rhand_q_pos
    
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

def SMPLX_AP_V2(t_pos, p_pos, iter_num = 3):

    device = p_pos.device
    p_pos = p_pos - p_pos[:,0:1]
    t_pos = t_pos - t_pos[:,0:1]
    
    # -------------------------Body-AP-----------------------------------------
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
    body_children = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21])
    body_index = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    body_t_pos = t_pos[:,body_index]
    body_p_pos = p_pos[:,body_index]
    
    # bl/bd
    body_t_pos_bl = get_bl_from_pos(body_t_pos, body_parent)

    ## 0
    body_q_pos_0 = torch.zeros([body_t_pos_bl.shape[0],1,3], dtype=torch.float32).to(device)          # [b,1,3]
    root_rotmat = batch_get_pelvis_orient(body_p_pos.unsqueeze(-1), body_t_pos.unsqueeze(-1), body_parent[1:24], body_children, torch.float32).unsqueeze(1)  # [b,1,1,3,3]
   
    ## ---------------------left leg-------------------------------------------
    ### 1
    body_q_pos_1 = torch.matmul(root_rotmat, body_t_pos[:,1:2,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 4 and 7
    #### old
    body_p4_q1_bd = (body_p_pos[:,4:5]-body_q_pos_1) / distance(body_p_pos[:,4:5], body_q_pos_1).unsqueeze(-1)
    body_q_pos_4 = body_q_pos_1 + body_p4_q1_bd * body_t_pos_bl[:,4:5].unsqueeze(-1)
    
    body_p7_q4_bd = (body_p_pos[:,7:8]-body_q_pos_4) / distance(body_p_pos[:,7:8], body_q_pos_4).unsqueeze(-1)
    body_q_pos_7 = body_q_pos_4 + body_p7_q4_bd * body_t_pos_bl[:,7:8].unsqueeze(-1)
  
    #### new  
    body_q_pos_4, body_q_pos_7 = itertive_move(body_q_pos_1, body_q_pos_4, body_q_pos_7, body_p_pos[:,7:8], body_t_pos_bl[:, 4:5], body_t_pos_bl[:, 7:8], iter_num=iter_num)  
  
    #### 10
    body_p10_q7_bd = (body_p_pos[:,10:11]-body_q_pos_7) / distance(body_p_pos[:,10:11], body_q_pos_7).unsqueeze(-1)
    body_q_pos_10 = body_q_pos_7 + body_p10_q7_bd * body_t_pos_bl[:,10:11].unsqueeze(-1)
    

    ### ---------------------Right leg---------------------------------------------
    #### 2
    body_q_pos_2 = torch.matmul(root_rotmat, body_t_pos[:,2:3,:].unsqueeze(-1)).squeeze(-1)
  
    #### 5 and 8
    ##### old
    body_p5_q2_bd = (body_p_pos[:,5:6]-body_q_pos_2) / distance(body_p_pos[:,5:6], body_q_pos_2).unsqueeze(-1)
    body_q_pos_5 = body_q_pos_2 + body_p5_q2_bd * body_t_pos_bl[:,5:6].unsqueeze(-1)
 
    body_p8_q5_bd = (body_p_pos[:,8:9]-body_q_pos_5) / distance(body_p_pos[:,8:9], body_q_pos_5).unsqueeze(-1)
    body_q_pos_8 = body_q_pos_5 + body_p8_q5_bd * body_t_pos_bl[:,8:9].unsqueeze(-1)
 
    ##### new 
    body_q_pos_5, body_q_pos_8 = itertive_move(body_q_pos_2, body_q_pos_5, body_q_pos_8, body_p_pos[:, 8:9], body_t_pos_bl[:, 5:6], body_t_pos_bl[:,8:9], iter_num=iter_num)    
 
    #### 11
    body_p11_q8_bd = (body_p_pos[:,11:12]-body_q_pos_8) / distance(body_p_pos[:,11:12], body_q_pos_8).unsqueeze(-1)
    body_q_pos_11 = body_q_pos_8 + body_p11_q8_bd * body_t_pos_bl[:,11:12].unsqueeze(-1)
   
   
    ### ---------------------Spine---------------------------------------------
    #### 3
    body_q_pos_3 = torch.matmul(root_rotmat, body_t_pos[:,3:4,:].unsqueeze(-1)).squeeze(-1)
 
    #### 6 and 9
    ##### old
    body_p6_q3_bd = (body_p_pos[:,6:7]-body_q_pos_3) / distance(body_p_pos[:,6:7], body_q_pos_3).unsqueeze(-1)
    body_q_pos_6 = body_q_pos_3 + body_p6_q3_bd * body_t_pos_bl[:,6:7].unsqueeze(-1)
  
    body_p9_q6_bd = (body_p_pos[:,9:10]-body_q_pos_6) / distance(body_p_pos[:,9:10], body_q_pos_6).unsqueeze(-1)
    body_q_pos_9 = body_q_pos_6 + body_p9_q6_bd * body_t_pos_bl[:,9:10].unsqueeze(-1)

    ##### new
    body_q_pos_6, body_q_pos_9 = itertive_move(body_q_pos_3, body_q_pos_6, body_q_pos_9, body_p_pos[:, 9:10], body_t_pos_bl[:,6:7], body_t_pos_bl[:,9:10], iter_num=iter_num)

    body_p_pos[:,9:10] = body_q_pos_9
    body_vec_pt = body_p_pos[:, 1:] - body_p_pos[:, body_parent[1:]]
    body_vec_t = body_t_pos[:, 1:] - body_t_pos[:, body_parent[1:]]
    spine3_rotmat = batch_get_neck_orient(body_vec_pt.unsqueeze(-1), body_vec_t.unsqueeze(-1), body_parent[1:24], body_children, torch.float32).unsqueeze(1)
    
    #### 12
    body_q_pos_12 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,11:12,:].unsqueeze(-1)).squeeze(-1)
  
    #### 15
    body_p15_q12_bd = (body_p_pos[:,15:16]-body_q_pos_12) / distance(body_p_pos[:,15:16], body_q_pos_12).unsqueeze(-1)
    body_q_pos_15 = body_q_pos_12 + body_p15_q12_bd * body_t_pos_bl[:,15:16].unsqueeze(-1)
    

    ### ---------------------left----------------------------------------------------
    #### 13
    body_q_pos_13 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 16 and 18
    ##### old
    body_p16_q13_bd = (body_p_pos[:,16:17]-body_q_pos_13) / distance(body_p_pos[:,16:17], body_q_pos_13).unsqueeze(-1)
    body_q_pos_16 = body_q_pos_13 + body_p16_q13_bd * body_t_pos_bl[:,16:17].unsqueeze(-1)
    
    body_p18_q16_bd = (body_p_pos[:,18:19]-body_q_pos_16) / distance(body_p_pos[:,18:19], body_q_pos_16).unsqueeze(-1)
    body_q_pos_18 = body_q_pos_16 + body_p18_q16_bd * body_t_pos_bl[:,18:19].unsqueeze(-1)
   
    ##### new 
    body_q_pos_16, body_q_pos_18 = itertive_move(body_q_pos_13, body_q_pos_16, body_q_pos_18, body_p_pos[:,18:19], body_t_pos_bl[:,16:17], body_t_pos_bl[:, 18:19], iter_num=iter_num) 
   
    #### 20
    body_p20_q18_bd = (body_p_pos[:,20:21]-body_q_pos_18) / distance(body_p_pos[:,20:21], body_q_pos_18).unsqueeze(-1)
    body_q_pos_20 = body_q_pos_18 + body_p20_q18_bd * body_t_pos_bl[:,20:21].unsqueeze(-1)
   

    ### ---------------------right----------------------------------------------------
    #### 14
    body_q_pos_14 = body_q_pos_9 + torch.matmul(spine3_rotmat, body_vec_t[:,13:14,:].unsqueeze(-1)).squeeze(-1)

    #### 17 and 19
    ##### old
    body_p17_q14_bd = (body_p_pos[:,17:18]-body_q_pos_14) / distance(body_p_pos[:,17:18], body_q_pos_14).unsqueeze(-1)
    body_q_pos_17 = body_q_pos_14 + body_p17_q14_bd * body_t_pos_bl[:,17:18].unsqueeze(-1)
  
    body_p19_q17_bd = (body_p_pos[:,19:20]-body_q_pos_17) / distance(body_p_pos[:,19:20], body_q_pos_17).unsqueeze(-1)
    body_q_pos_19 = body_q_pos_17 + body_p19_q17_bd * body_t_pos_bl[:,19:20].unsqueeze(-1)
  
    ##### new 
    body_q_pos_17, body_q_pos_19 = itertive_move(body_q_pos_14, body_q_pos_17, body_q_pos_19, body_p_pos[:,19:20], body_t_pos_bl[:,17:18], body_t_pos_bl[:, 19:20], iter_num=iter_num)  
  
    #### 21
    body_p21_q19_bd = (body_p_pos[:,21:22]-body_q_pos_19) / distance(body_p_pos[:,21:22], body_q_pos_19).unsqueeze(-1)
    body_q_pos_21 = body_q_pos_19 + body_p21_q19_bd * body_t_pos_bl[:,21:22].unsqueeze(-1)
  

    body_q_pos_list = [body_q_pos_0, body_q_pos_1, body_q_pos_2, body_q_pos_3, body_q_pos_4, body_q_pos_5, body_q_pos_6, body_q_pos_7, body_q_pos_8, body_q_pos_9,
              body_q_pos_10, body_q_pos_11, body_q_pos_12, body_q_pos_13, body_q_pos_14, body_q_pos_15, body_q_pos_16, body_q_pos_17, body_q_pos_18, body_q_pos_19, 
              body_q_pos_20, body_q_pos_21]
    
    body_q_pos = torch.cat(body_q_pos_list, dim=1)
    
    
    # -------------------------Lhand-IK-----------------------------------------
    lhand_index = torch.tensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70])
    lhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    lhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    lhand_t_pos = t_pos[:,lhand_index]
    lhand_p_pos = p_pos[:,lhand_index]
    
    # bl/bd
    lhand_t_pos_bl = get_bl_from_pos(lhand_t_pos, lhand_parent)
    
    lhand_p_pos[:,0:1] = body_q_pos_20
    lhand_vec_pt = lhand_p_pos[:, 1:] - lhand_p_pos[:, lhand_parent[1:]]
    lhand_vec_t = lhand_t_pos[:, 1:] - lhand_t_pos[:, lhand_parent[1:]]
    lhand_wrist_rotmat = batch_get_wrist_orient(lhand_vec_pt.unsqueeze(-1), lhand_vec_t.unsqueeze(-1), lhand_parent, lhand_children, torch.float32).unsqueeze(1)

    ## ---------------------Thumb-------------------------------------------
    ### 1
    lhand_q_pos_1 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,0:1,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2 and 3
    #### old
    lhand_p2_q1_bd = (lhand_p_pos[:,2:3]-lhand_q_pos_1) / distance(lhand_p_pos[:,2:3], lhand_q_pos_1).unsqueeze(-1)
    lhand_q_pos_2 = lhand_q_pos_1 + lhand_p2_q1_bd * lhand_t_pos_bl[:,2:3].unsqueeze(-1)
    
    lhand_p3_q2_bd = (lhand_p_pos[:,3:4]-lhand_q_pos_2) / distance(lhand_p_pos[:,3:4], lhand_q_pos_2).unsqueeze(-1)
    lhand_q_pos_3 = lhand_q_pos_2 + lhand_p3_q2_bd * lhand_t_pos_bl[:,3:4].unsqueeze(-1)
    
    #### new
    lhand_q_pos_2, lhand_q_pos_3 = itertive_move(lhand_q_pos_1, lhand_q_pos_2, lhand_q_pos_3, lhand_p_pos[:,3:4], lhand_t_pos_bl[:, 2:3], lhand_t_pos_bl[:, 3:4], iter_num=iter_num)
    
    ### 4
    lhand_p4_q3_bd = (lhand_p_pos[:,4:5]-lhand_q_pos_3) / distance(lhand_p_pos[:,4:5], lhand_q_pos_3).unsqueeze(-1)
    lhand_q_pos_4 = lhand_q_pos_3 + lhand_p4_q3_bd * lhand_t_pos_bl[:,4:5].unsqueeze(-1)
    

    ### ---------------------Index---------------------------------------------
    #### 5
    lhand_q_pos_5 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,4:5,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6 and 7
    ##### old
    lhand_p6_q5_bd = (lhand_p_pos[:,6:7]-lhand_q_pos_5) / distance(lhand_p_pos[:,6:7], lhand_q_pos_5).unsqueeze(-1)
    lhand_q_pos_6 = lhand_q_pos_5 + lhand_p6_q5_bd * lhand_t_pos_bl[:,6:7].unsqueeze(-1)
 
    lhand_p7_q6_bd = (lhand_p_pos[:,7:8]-lhand_q_pos_6) / distance(lhand_p_pos[:,7:8], lhand_q_pos_6).unsqueeze(-1)
    lhand_q_pos_7 = lhand_q_pos_6 + lhand_p7_q6_bd * lhand_t_pos_bl[:,7:8].unsqueeze(-1)
    
    ##### new
    lhand_q_pos_6, lhand_q_pos_7 = itertive_move(lhand_q_pos_5, lhand_q_pos_6, lhand_q_pos_7, lhand_p_pos[:,7:8], lhand_t_pos_bl[:, 6:7], lhand_t_pos_bl[:, 7:8], iter_num=iter_num)
    
    #### 8
    lhand_p8_q7_bd = (lhand_p_pos[:,8:9]-lhand_q_pos_7) / distance(lhand_p_pos[:,8:9], lhand_q_pos_7).unsqueeze(-1)
    lhand_q_pos_8 = lhand_q_pos_7 + lhand_p8_q7_bd * lhand_t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### ---------------------Middle---------------------------------------------
    #### 9
    lhand_q_pos_9 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,8:9,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10 and 11
    ##### old
    lhand_p10_q9_bd = (lhand_p_pos[:,10:11]-lhand_q_pos_9) / distance(lhand_p_pos[:,10:11], lhand_q_pos_9).unsqueeze(-1)
    lhand_q_pos_10 = lhand_q_pos_9 + lhand_p10_q9_bd * lhand_t_pos_bl[:,10:11].unsqueeze(-1)
  
    lhand_p11_q10_bd = (lhand_p_pos[:,11:12]-lhand_q_pos_10) / distance(lhand_p_pos[:,11:12], lhand_q_pos_10).unsqueeze(-1)
    lhand_q_pos_11 = lhand_q_pos_10 + lhand_p11_q10_bd * lhand_t_pos_bl[:,11:12].unsqueeze(-1)
    
    ##### new
    lhand_q_pos_10, lhand_q_pos_11 = itertive_move(lhand_q_pos_9, lhand_q_pos_10, lhand_q_pos_11, lhand_p_pos[:,11:12], lhand_t_pos_bl[:, 10:11], lhand_t_pos_bl[:, 11:12], iter_num=iter_num)
    
    #### 12
    lhand_p12_q11_bd = (lhand_p_pos[:,12:13]-lhand_q_pos_11) / distance(lhand_p_pos[:,12:13], lhand_q_pos_11).unsqueeze(-1)
    lhand_q_pos_12 = lhand_q_pos_11 + lhand_p12_q11_bd * lhand_t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    lhand_q_pos_13 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14 and 15
    ##### old
    lhand_p14_q13_bd = (lhand_p_pos[:,14:15]-lhand_q_pos_13) / distance(lhand_p_pos[:,14:15], lhand_q_pos_13).unsqueeze(-1)
    lhand_q_pos_14 = lhand_q_pos_13 + lhand_p14_q13_bd * lhand_t_pos_bl[:,14:15].unsqueeze(-1)
    
    lhand_p15_q14_bd = (lhand_p_pos[:,15:16]-lhand_q_pos_14) / distance(lhand_p_pos[:,15:16], lhand_q_pos_14).unsqueeze(-1)
    lhand_q_pos_15 = lhand_q_pos_14 + lhand_p15_q14_bd * lhand_t_pos_bl[:,15:16].unsqueeze(-1)
    
    ##### new
    lhand_q_pos_14, lhand_q_pos_15 = itertive_move(lhand_q_pos_13, lhand_q_pos_14, lhand_q_pos_15, lhand_p_pos[:,15:16], lhand_t_pos_bl[:, 14:15], lhand_t_pos_bl[:, 15:16], iter_num=iter_num)
    
    #### 16
    lhand_p16_q15_bd = (lhand_p_pos[:,16:17]-lhand_q_pos_15) / distance(lhand_p_pos[:,16:17], lhand_q_pos_15).unsqueeze(-1)
    lhand_q_pos_16 = lhand_q_pos_15 + lhand_p16_q15_bd * lhand_t_pos_bl[:,16:17].unsqueeze(-1)
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    lhand_q_pos_17 = body_q_pos_20 + torch.matmul(lhand_wrist_rotmat, lhand_vec_t[:,16:17,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18 and 19
    lhand_p18_q17_bd = (lhand_p_pos[:,18:19]-lhand_q_pos_17) / distance(lhand_p_pos[:,18:19], lhand_q_pos_17).unsqueeze(-1)
    lhand_q_pos_18 = lhand_q_pos_17 + lhand_p18_q17_bd * lhand_t_pos_bl[:,18:19].unsqueeze(-1)

    lhand_p19_q18_bd = (lhand_p_pos[:,19:20]-lhand_q_pos_18) / distance(lhand_p_pos[:,19:20], lhand_q_pos_18).unsqueeze(-1)
    lhand_q_pos_19 = lhand_q_pos_18 + lhand_p19_q18_bd * lhand_t_pos_bl[:,19:20].unsqueeze(-1)
    
    ##### new
    lhand_q_pos_18, lhand_q_pos_19 = itertive_move(lhand_q_pos_17, lhand_q_pos_18, lhand_q_pos_19, lhand_p_pos[:,19:20], lhand_t_pos_bl[:, 18:19], lhand_t_pos_bl[:, 19:20], iter_num=iter_num)
    
    #### 20
    lhand_p20_q19_bd = (lhand_p_pos[:,20:21]-lhand_q_pos_19) / distance(lhand_p_pos[:,20:21], lhand_q_pos_19).unsqueeze(-1)
    lhand_q_pos_20 = lhand_q_pos_19 + lhand_p20_q19_bd * lhand_t_pos_bl[:,20:21].unsqueeze(-1)    

    lhand_q_pos_list = [lhand_q_pos_1, lhand_q_pos_2, lhand_q_pos_3, lhand_q_pos_4, lhand_q_pos_5, lhand_q_pos_6, lhand_q_pos_7, lhand_q_pos_8, lhand_q_pos_9,
              lhand_q_pos_10, lhand_q_pos_11, lhand_q_pos_12, lhand_q_pos_13, lhand_q_pos_14, lhand_q_pos_15, lhand_q_pos_16, lhand_q_pos_17, 
              lhand_q_pos_18, lhand_q_pos_19, lhand_q_pos_20]
    
    lhand_q_pos = torch.cat(lhand_q_pos_list, dim=1)


    # -------------------------Rhand-IK-----------------------------------------
    rhand_index = torch.tensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
    rhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
    rhand_children = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    
    rhand_t_pos = t_pos[:,rhand_index]
    rhand_p_pos = p_pos[:,rhand_index]
    
    # bl/bd
    rhand_t_pos_bl = get_bl_from_pos(rhand_t_pos, rhand_parent)
    
    rhand_p_pos[:,0:1] = body_q_pos_21
    rhand_vec_pt = rhand_p_pos[:, 1:] - rhand_p_pos[:, rhand_parent[1:]]
    rhand_vec_t = rhand_t_pos[:, 1:] - rhand_t_pos[:, rhand_parent[1:]]
    rhand_wrist_rotmat = batch_get_wrist_orient(rhand_vec_pt.unsqueeze(-1), rhand_vec_t.unsqueeze(-1), rhand_parent, rhand_children, torch.float32).unsqueeze(1)

    ## ---------------------Thumb-------------------------------------------
    ### 1
    rhand_q_pos_1 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,0:1,:].unsqueeze(-1)).squeeze(-1)  # [b,1,3]
  
    ### 2 and 3
    #### old
    rhand_p2_q1_bd = (rhand_p_pos[:,2:3]-rhand_q_pos_1) / distance(rhand_p_pos[:,2:3], rhand_q_pos_1).unsqueeze(-1)
    rhand_q_pos_2 = rhand_q_pos_1 + rhand_p2_q1_bd * rhand_t_pos_bl[:,2:3].unsqueeze(-1)
    
    rhand_p3_q2_bd = (rhand_p_pos[:,3:4]-rhand_q_pos_2) / distance(rhand_p_pos[:,3:4], rhand_q_pos_2).unsqueeze(-1)
    rhand_q_pos_3 = rhand_q_pos_2 + rhand_p3_q2_bd * rhand_t_pos_bl[:,3:4].unsqueeze(-1)
    
    #### new
    rhand_q_pos_2, rhand_q_pos_3 = itertive_move(rhand_q_pos_1, rhand_q_pos_2, rhand_q_pos_3, rhand_p_pos[:,3:4], rhand_t_pos_bl[:, 2:3], rhand_t_pos_bl[:, 3:4], iter_num=iter_num)
    
    ### 4
    rhand_p4_q3_bd = (rhand_p_pos[:,4:5]-rhand_q_pos_3) / distance(rhand_p_pos[:,4:5], rhand_q_pos_3).unsqueeze(-1)
    rhand_q_pos_4 = rhand_q_pos_3 + rhand_p4_q3_bd * rhand_t_pos_bl[:,4:5].unsqueeze(-1)
    

    ### ---------------------Index---------------------------------------------
    #### 5
    rhand_q_pos_5 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,4:5,:].unsqueeze(-1)).squeeze(-1)
  
    #### 6 and 7
    ##### old
    rhand_p6_q5_bd = (rhand_p_pos[:,6:7]-rhand_q_pos_5) / distance(rhand_p_pos[:,6:7], rhand_q_pos_5).unsqueeze(-1)
    rhand_q_pos_6 = rhand_q_pos_5 + rhand_p6_q5_bd * rhand_t_pos_bl[:,6:7].unsqueeze(-1)
 
    rhand_p7_q6_bd = (rhand_p_pos[:,7:8]-rhand_q_pos_6) / distance(rhand_p_pos[:,7:8], rhand_q_pos_6).unsqueeze(-1)
    rhand_q_pos_7 = rhand_q_pos_6 + rhand_p7_q6_bd * rhand_t_pos_bl[:,7:8].unsqueeze(-1)
 
    ##### new
    rhand_q_pos_6, rhand_q_pos_7 = itertive_move(rhand_q_pos_5, rhand_q_pos_6, rhand_q_pos_7, rhand_p_pos[:,7:8], rhand_t_pos_bl[:, 6:7], rhand_t_pos_bl[:, 7:8], iter_num=iter_num)
    
    #### 8
    rhand_p8_q7_bd = (rhand_p_pos[:,8:9]-rhand_q_pos_7) / distance(rhand_p_pos[:,8:9], rhand_q_pos_7).unsqueeze(-1)
    rhand_q_pos_8 = rhand_q_pos_7 + rhand_p8_q7_bd * rhand_t_pos_bl[:,8:9].unsqueeze(-1)
   
   
    ### --------------------Middle---------------------------------------------
    #### 9
    rhand_q_pos_9 = body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,8:9,:].unsqueeze(-1)).squeeze(-1)
 
    #### 10 and 11
    ##### old
    rhand_p10_q9_bd = (rhand_p_pos[:,10:11]-rhand_q_pos_9) / distance(rhand_p_pos[:,10:11], rhand_q_pos_9).unsqueeze(-1)
    rhand_q_pos_10 = rhand_q_pos_9 + rhand_p10_q9_bd * rhand_t_pos_bl[:,10:11].unsqueeze(-1)
  
    rhand_p11_q10_bd = (rhand_p_pos[:,11:12]-rhand_q_pos_10) / distance(rhand_p_pos[:,11:12], rhand_q_pos_10).unsqueeze(-1)
    rhand_q_pos_11 = rhand_q_pos_10 + rhand_p11_q10_bd * rhand_t_pos_bl[:,11:12].unsqueeze(-1)
    
    ##### new
    rhand_q_pos_10, rhand_q_pos_11 = itertive_move(rhand_q_pos_9, rhand_q_pos_10, rhand_q_pos_11, rhand_p_pos[:,11:12], rhand_t_pos_bl[:, 10:11], rhand_t_pos_bl[:, 11:12], iter_num=iter_num)
    
    #### 12
    rhand_p12_q11_bd = (rhand_p_pos[:,12:13]-rhand_q_pos_11) / distance(rhand_p_pos[:,12:13], rhand_q_pos_11).unsqueeze(-1)
    rhand_q_pos_12 = rhand_q_pos_11 + rhand_p12_q11_bd * rhand_t_pos_bl[:,12:13].unsqueeze(-1)
 

    ### ----------------------Ring---------------------------------------------
    #### 13
    rhand_q_pos_13 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,12:13,:].unsqueeze(-1)).squeeze(-1)
  
    #### 14 and 15
    rhand_p14_q13_bd = (rhand_p_pos[:,14:15]-rhand_q_pos_13) / distance(rhand_p_pos[:,14:15], rhand_q_pos_13).unsqueeze(-1)
    rhand_q_pos_14 = rhand_q_pos_13 + rhand_p14_q13_bd * rhand_t_pos_bl[:,14:15].unsqueeze(-1)
    
    rhand_p15_q14_bd = (rhand_p_pos[:,15:16]-rhand_q_pos_14) / distance(rhand_p_pos[:,15:16], rhand_q_pos_14).unsqueeze(-1)
    rhand_q_pos_15 = rhand_q_pos_14 + rhand_p15_q14_bd * rhand_t_pos_bl[:,15:16].unsqueeze(-1)
    
    ##### new
    rhand_q_pos_14, rhand_q_pos_15 = itertive_move(rhand_q_pos_13, rhand_q_pos_14, rhand_q_pos_15, rhand_p_pos[:,15:16], rhand_t_pos_bl[:, 14:15], rhand_t_pos_bl[:, 15:16], iter_num=iter_num)
    
    #### 16
    rhand_p16_q15_bd = (rhand_p_pos[:,16:17]-rhand_q_pos_15) / distance(rhand_p_pos[:,16:17], rhand_q_pos_15).unsqueeze(-1)
    rhand_q_pos_16 = rhand_q_pos_15 + rhand_p16_q15_bd * rhand_t_pos_bl[:,16:17].unsqueeze(-1)
   
    ### ----------------------Pinky--------------------------------------------
    #### 17
    rhand_q_pos_17 =  body_q_pos_21 + torch.matmul(rhand_wrist_rotmat, rhand_vec_t[:,16:17,:].unsqueeze(-1)).squeeze(-1)
  
    #### 18 and 19
    ##### old
    rhand_p18_q17_bd = (rhand_p_pos[:,18:19]-rhand_q_pos_17) / distance(rhand_p_pos[:,18:19], rhand_q_pos_17).unsqueeze(-1)
    rhand_q_pos_18 = rhand_q_pos_17 + rhand_p18_q17_bd * rhand_t_pos_bl[:,18:19].unsqueeze(-1)
    
    rhand_p19_q18_bd = (rhand_p_pos[:,19:20]-rhand_q_pos_18) / distance(rhand_p_pos[:,19:20], rhand_q_pos_18).unsqueeze(-1)
    rhand_q_pos_19 = rhand_q_pos_18 + rhand_p19_q18_bd * rhand_t_pos_bl[:,19:20].unsqueeze(-1)
    
    ##### new
    rhand_q_pos_18, rhand_q_pos_19 = itertive_move(rhand_q_pos_17, rhand_q_pos_18, rhand_q_pos_19, rhand_p_pos[:,19:20], rhand_t_pos_bl[:, 18:19], rhand_t_pos_bl[:, 19:20], iter_num=iter_num)
    
    #### 20
    rhand_p20_q19_bd = (rhand_p_pos[:,20:21]-rhand_q_pos_19) / distance(rhand_p_pos[:,20:21], rhand_q_pos_19).unsqueeze(-1)
    rhand_q_pos_20 = rhand_q_pos_19 + rhand_p20_q19_bd * rhand_t_pos_bl[:,20:21].unsqueeze(-1)    

    rhand_q_pos_list = [rhand_q_pos_1, rhand_q_pos_2, rhand_q_pos_3, rhand_q_pos_4, rhand_q_pos_5, rhand_q_pos_6, rhand_q_pos_7, rhand_q_pos_8, rhand_q_pos_9,
              rhand_q_pos_10, rhand_q_pos_11, rhand_q_pos_12, rhand_q_pos_13, rhand_q_pos_14, rhand_q_pos_15, rhand_q_pos_16, rhand_q_pos_17, 
              rhand_q_pos_18, rhand_q_pos_19, rhand_q_pos_20]
    
    rhand_q_pos = torch.cat(rhand_q_pos_list, dim=1)

    q_pos = p_pos.clone()
    q_pos[:,body_index] = body_q_pos
    q_pos[:,lhand_index[1:]] = lhand_q_pos
    q_pos[:,rhand_index[1:]] = rhand_q_pos
    
    return q_pos

