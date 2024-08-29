import torch
from lib.utils.si_utils import get_bl_14_from_pos, get_bl_from_pos, get_bl_33_from_pos 



def SMPL_SI_LR(t_pos, p_pos, parent, A1):
    t_pos_bl = get_bl_14_from_pos(t_pos, parent) * 1000
    p_pos_bl = get_bl_14_from_pos(p_pos, parent) * 1000
    bl = p_pos_bl - t_pos_bl

    ##----------------------------------
    A1 = A1.unsqueeze(0)
    b = bl.unsqueeze(-1)

    x = torch.matmul(A1.permute(0,2,1), A1)
    x = torch.matmul(torch.inverse(x), A1.permute(0,2,1))
    x = torch.matmul(x,b).squeeze(-1)
    
    return x

def MANO_SI_LR(t_pos, p_pos, parent, A1):
    t_pos_bl = get_bl_from_pos(t_pos, parent)[:,1:] * 1000
    p_pos_bl = get_bl_from_pos(p_pos, parent)[:,1:] * 1000
    bl = p_pos_bl - t_pos_bl


    ##---------------------------------
    A1 = A1.unsqueeze(0)
    b = bl.unsqueeze(-1)

    x = torch.matmul(A1.permute(0,2,1), A1)
    x = torch.matmul(torch.inverse(x), A1.permute(0,2,1))
    x = torch.matmul(x,b).squeeze(-1)
 
    return x


def SMPLX_SI_LR(t_pos, p_pos, A1):
    t_pos_bl = get_bl_33_from_pos(t_pos) * 1000
    p_pos_bl = get_bl_33_from_pos(p_pos) * 1000
    bl = p_pos_bl - t_pos_bl
    
    ##---------------------------------
    A1 = A1.unsqueeze(0)
    b = bl.unsqueeze(-1)

    x = torch.matmul(A1.permute(0,2,1), A1)
    x = torch.matmul(torch.inverse(x), A1.permute(0,2,1))
    x = torch.matmul(x,b).squeeze(-1)
    return x