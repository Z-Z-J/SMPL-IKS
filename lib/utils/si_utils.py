# -*- coding: utf-8 -*-

import torch

def distance(position1, position2):
    vector = torch.abs(position1 - position2)
    return torch.norm(vector, dim=-1)

def get_bl_from_pos(position_3d, parent):
    """
    Function:
        Get bone length from points3d.
    Arguments:
        points3d: [b, 24, 3]
    Returns:
        bl: [b,14]
    """
    b,n,c = position_3d.shape
    device = position_3d.device
    bl_list = []  
    for i in range(n):
        if i == 0:
            bl_list.append(torch.zeros([b], dtype=torch.float32).to(device))
        else:
            bl_list.append(distance(position_3d[:, i], position_3d[:, parent[i]]))
    
    length = torch.stack(bl_list, dim=1)
    return length

def get_bl_14_from_pos(position_3d, parent):
    """
    Function:
        Get bone length from positions3d.
    Arguments:
        position_3d: [b, 24, 4]
    Returns:
        bl: [b,14]
    """
    bl = get_bl_from_pos(position_3d, parent)    
    bl_10 = (bl[:,1] + bl[:,2]) / 2.
    bl_41 = (bl[:,4] + bl[:,5]) / 2.
    bl_74 = (bl[:,7] + bl[:,8]) / 2.
    bl_107 = (bl[:,10] + bl[:,11]) /2.
    bl_30 = bl[:,3]
    bl_63 = bl[:,6]
    bl_96 = bl[:,9]
    bl_129 = bl[:,12]
    bl_1512 = bl[:,15]
    bl_139 = (bl[:,13] + bl[:,14]) / 2.
    bl_1613 = (bl[:,16] + bl[:,17]) / 2.
    bl_1816 = (bl[:,18] + bl[:,19]) / 2.
    bl_2018 = (bl[:,20] + bl[:,21]) / 2.
    bl_2220 = (bl[:,22] + bl[:,23]) / 2.

    bl_list = [bl_10, bl_41, bl_74, bl_107, bl_30, bl_63, bl_96,
               bl_129, bl_1512, bl_139, bl_1613, bl_1816, bl_2018, bl_2220]
    
    bl = torch.stack(bl_list, dim=-1)
    
    return bl

def get_bl_33_from_pos(position_3d):
    """
    Function:
        Get bone length from positions3d.
    Arguments:
        position_3d: [b, 127, 4]
    Returns:
        bl: [b,33]
    """
    body_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    body_parent = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
  
    lhand_index = torch.tensor([20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70])
    lhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])
  
    rhand_index = torch.tensor([21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75])
    rhand_parent = torch.tensor([-1,0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19])

    # body
    body_bl = get_bl_from_pos(position_3d[:, body_index], body_parent)
     
    lbody_bl_index = [1, 4, 7, 10, 3, 6, 9, 12, 15, 13, 16, 18, 20]
    lbody_bl = body_bl[:,lbody_bl_index]
     
    rbody_bl_index = [2, 5, 8, 11, 3, 6, 9, 12, 15, 14, 17, 19, 21]
    rbody_bl = body_bl[:,rbody_bl_index]
     
    body_bl = (lbody_bl + rbody_bl) / 2.
     
    # hand
    lhand_bl = get_bl_from_pos(position_3d[:, lhand_index], lhand_parent)
    rhand_bl = get_bl_from_pos(position_3d[:, rhand_index], rhand_parent)
    
    hand_bl = (lhand_bl[:,1:] + rhand_bl[:,1:]) / 2.
     
    bl = torch.cat([body_bl, hand_bl], dim=-1) 

    return bl


def get_bd_from_pos(position_3d, parent):
    """
    Function:
        Get bone direction from points_3d
    Arguments:
        position_3d: [b, 24, 3]
    Returns:
        bd: [b,24,3]
    """
    b, n, c = position_3d.shape

    bd_list = []
    for i in range(n):
        if i == 0:
            bd_list.append(torch.zeros([b,c], dtype=torch.float32))
        else:
            bd_list.append((position_3d[:, i]-position_3d[:, parent[i]]) / distance(position_3d[:, i], position_3d[:, parent[i]]).unsqueeze(-1))
    
    direction = torch.stack(bd_list,dim=1)
    return direction