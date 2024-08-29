# -*- coding: utf-8 -*-

import torch
from torch.nn import functional as F

def save_obj(verts, faces, path):
   """
   Save the SMPL model into .obj file.
   Parameter:
   ---------
   path: Path to save.
   """
   with open(path, 'w') as fp:
       for v in verts:
           fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
       for f in faces + 1:
           fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat
    
def rotmat_to_aa(rotmat):
    batch_size = rotmat.shape[0]

    r11 = rotmat[:, 0, 0]
    r22 = rotmat[:, 1, 1]
    r33 = rotmat[:, 2, 2]

    r12 = rotmat[:, 0, 1]
    r21 = rotmat[:, 1, 0]
    r13 = rotmat[:, 0, 2]
    r31 = rotmat[:, 2, 0]
    r23 = rotmat[:, 1, 2]
    r32 = rotmat[:, 2, 1]
    
    angle = torch.acos((r11 + r22 + r33 - 1) / 2).unsqueeze(dim=1)
    '''
    if -1e-6 < (r11 + r22 + r33 - 1) / 2 + 1 < 1e-6:
        angle = torch.acos(-torch.ones_like(r11)).unsqueeze(dim=1)
    else:
        angle = torch.acos((r11 + r22 + r33 - 1) / 2).unsqueeze(dim=1)
    '''
    axis = torch.zeros((batch_size, 3))
    axis[:, 0] = r32 - r23
    axis[:, 1] = r13 - r31
    axis[:, 2] = r21 - r12
    axis = axis / (2 * torch.sin(angle) + 1e-8)

    aa = axis * angle
    return aa, axis, angle

def get_twist_rotmat(rot_mats, vec, dtype):
    
    batch_size = rot_mats.shape[0]
    device = rot_mats.device
        
    u = vec
    rot = rot_mats

    v = torch.matmul(rot, u)

    u_norm = torch.norm(u, dim=1, keepdim=True)
    v_norm = torch.norm(v, dim=1, keepdim=True)

    axis = torch.cross(u, v, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(u * v, dim=1, keepdim=True) / (u_norm * v_norm + 1e-8)
    sin = axis_norm / (u_norm * v_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_pivot = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    twist_rotmat = torch.matmul(rot_mat_pivot.transpose(1, 2), rot)
      
    return rot_mat_pivot, twist_rotmat

def get_twist(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)               # [b,j,3,1]
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    
    childs = -torch.ones((parents.shape[0]), dtype=parents.dtype, device=parents.device)
    for i in range(1, parents.shape[0]):
        childs[parents[i]] = i

    dtype = rot_mats.dtype
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    angle_twist = []
    rotmat_list = []
    error = False
    for i in range(1, parents.shape[0]):
        
        if childs[i] < 0:
            angle_twist.append(torch.zeros((batch_size, 1), dtype=rot_mats.dtype, device=rot_mats.device))
            continue

        u = rel_joints[:, childs[i]]
        rot = rot_mats[:, i]

        v = torch.matmul(rot, u)

        u_norm = torch.norm(u, dim=1, keepdim=True)
        v_norm = torch.norm(v, dim=1, keepdim=True)

        axis = torch.cross(u, v, dim=1)
        axis_norm = torch.norm(axis, dim=1, keepdim=True)

        # (B, 1, 1)
        cos = torch.sum(u * v, dim=1, keepdim=True) / (u_norm * v_norm + 1e-8)
        sin = axis_norm / (u_norm * v_norm + 1e-8)

        # (B, 3, 1)
        axis = axis / (axis_norm + 1e-8)

        # Convert location revolve to rot_mat by rodrigues
        # (B, 1, 1)
        rx, ry, rz = torch.split(axis, 1, dim=1)
        zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

        K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
            .view((batch_size, 3, 3))
        ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
        rot_mat_pivot = ident + sin * K + (1 - cos) * torch.bmm(K, K)

        rot_mat_twist = torch.matmul(rot_mat_pivot.transpose(1, 2), rot)
        rotmat_list.append(rot_mat_twist)
        _, axis, angle = rotmat_to_aa(rot_mat_twist)

        axis = axis / torch.norm(axis, dim=1, keepdim=True)
        spin_axis = u / u_norm
        spin_axis = spin_axis.squeeze(-1)

        pos = torch.norm(spin_axis - axis, dim=1)
        neg = torch.norm(spin_axis + axis, dim=1)
        if float(neg) < float(pos):
            try:
                assert float(pos) > 1.9, (pos, neg)
                angle_twist.append(-1 * angle)
            except AssertionError:
                angle_twist.append(torch.ones_like(angle) * -999)
                error = True
        else:
            try:
                assert float(neg) > 1.9, (pos, neg, axis, angle, rot_mat_twist)
                angle_twist.append(angle)
            except AssertionError:
                angle_twist.append(torch.ones_like(angle) * -999)
                error = True

    angle_twist = torch.stack(angle_twist, dim=1)
    if error:
        angle_twist[:] = -999

    return rotmat_list


def vectors2rotmat(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    device = vec_final.device

    # (B, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=1, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=1, keepdim=True)
    
    
    axis = torch.cross(vec_rest, vec_final, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=1, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
   
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_loc

def vectors2rotmat_bk(vec_rest, vec_final, dtype):
    batch_size = vec_final.shape[0]
    len_indices = vec_final.shape[1]
    device = vec_final.device

    # (B, K, 1, 1)
    vec_final_norm = torch.norm(vec_final, dim=2, keepdim=True)
    vec_rest_norm = torch.norm(vec_rest, dim=2, keepdim=True)
    
    
    axis = torch.cross(vec_rest, vec_final, dim=2)
    axis_norm = torch.norm(axis, dim=2, keepdim=True)

    # (B, K, 1, 1)
    cos = torch.sum(vec_rest * vec_final, dim=2, keepdim=True) / (vec_rest_norm * vec_final_norm + 1e-8)
    sin = axis_norm / (vec_rest_norm * vec_final_norm + 1e-8)

    # (B, K, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, K, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=2)
    zeros = torch.zeros((batch_size, len_indices, 1, 1), dtype=dtype, device=device)
   
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=2) \
        .view((batch_size, len_indices, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).reshape(1, 1, 3, 3)
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)

    return rot_mat_loc


def batch_get_orient(vec_p, vec_t, rotmat, dtype):
    
    p_parent_loc = vec_p[:,0].clone()
    p_child_loc = vec_p[:,1].clone()
    t_parent_loc = vec_t[:,0].clone()
    t_child_loc = vec_t[:,1].clone()
    
   
    t_parent_norm = torch.norm(t_parent_loc, dim=1, keepdim=True)
    t_parent_norm = t_parent_loc / (t_parent_norm + 1e-8) 
    
    p_child_loc = torch.matmul(rotmat.transpose(1,2), p_child_loc)
    
    
    p_child_m_loc = p_child_loc - torch.sum(p_child_loc * t_parent_norm, dim=1, keepdim=True) * t_parent_norm
    t_child_m_loc = t_child_loc - torch.sum(t_child_loc * t_parent_norm, dim=1, keepdim=True) * t_parent_norm
    twist_rotmat = vectors2rotmat(t_child_m_loc, p_child_m_loc, dtype)
    
    p_child_loc = torch.matmul(twist_rotmat.transpose(1,2), p_child_loc)
    
    swing_rotmat = vectors2rotmat(t_child_loc, p_child_loc, dtype)

    return twist_rotmat, swing_rotmat

def get_orient_svd(rel_pose_skeleton, rel_rest_pose):

    S = rel_rest_pose.bmm(rel_pose_skeleton.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    U, _, V = torch.svd(S_non_zero)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    # rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2))) 
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v #det (X) =-1, we form X' = V' U^t which is the desired rotation.
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_get_pelvis_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device

    assert children[0] == 3
    pelvis_child = [int(children[0])] #[3, 1, 2] pelvis的child joint
    for i in range(1, parents.shape[0]):
        if parents[i] == 0 and i not in pelvis_child:
            pelvis_child.append(i)

   
    spine_final_loc = rel_pose_skeleton[:, int(children[0])].clone() #rel_pose_skeleton[:,3] = pose_skeleton[:,3](spine1) - pose_skeleton[:,0](spine0);vector q
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8) 
   
    spine_rest_loc = rel_rest_pose[:, int(children[0])].clone() #rel_rest_pose[:,3] = rest_pose[:,3](spine1) - rest_pose[:,0](spine0);vector t
  
    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype) 

    assert torch.sum(torch.isnan(rot_mat_spine)) == 0, ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child: 
        if child == int(children[0]): continue 
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone() 
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone() 
    center_final_loc = center_final_loc / (len(pelvis_child) - 1) # /2
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1) # /2

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc) 
    
    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm


    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)
  
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat

def batch_get_neck_orient(rel_pose_skeleton, rel_rest_pose, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device
   
    #assert children[9] == 14
    pelvis_child = [int(children[9])] 
    for i in range(1, parents.shape[0]):
        if parents[i] == 9 and i not in pelvis_child:
            pelvis_child.append(i)
    
 
    spine_final_loc = rel_pose_skeleton[:, int(children[9])].clone() #rel_pose_skeleton[:,3] = pose_skeleton[:,3](spine1) - pose_skeleton[:,0](spine0);vector q
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8) 
   
    spine_rest_loc = rel_rest_pose[:, int(children[9])].clone() #rel_rest_pose[:,3] = rest_pose[:,3](spine1) - rest_pose[:,0](spine0);vector t

    rot_mat_spine = vectors2rotmat(spine_rest_loc, spine_final_loc, dtype) 

    assert torch.sum(torch.isnan(rot_mat_spine)) == 0, ('rot_mat_spine', rot_mat_spine)
    center_final_loc = 0
    center_rest_loc = 0
    for child in pelvis_child: 
        if child == int(children[9]): continue 
        center_final_loc = center_final_loc + rel_pose_skeleton[:, child].clone() 
        center_rest_loc = center_rest_loc + rel_rest_pose[:, child].clone() 
    center_final_loc = center_final_loc / (len(pelvis_child) - 1) # /2
    center_rest_loc = center_rest_loc / (len(pelvis_child) - 1) # /2

    center_rest_loc = torch.matmul(rot_mat_spine, center_rest_loc) 
   
    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm


  
    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_spine)

    return rot_mat


def batch_get_spine3_twist(rel_pose_skeleton, rel_rest_skeleton, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device
   
    # action-pose
    spine_final_loc = rel_pose_skeleton[:, 8].clone() # vec 9-6
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8) 
    

    center_final_loc = torch.cross(rel_pose_skeleton[:,12], rel_pose_skeleton[:,13]) 
    center_rest_loc = torch.cross(rel_rest_skeleton[:,12], rel_rest_skeleton[:,13])


    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm


    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_center

def batch_get_wrist_twist(rel_pose_skeleton, rel_rest_skeleton, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device
   
    # action-pose
    spine_final_loc = rel_pose_skeleton[:, 0].clone() # vec 20-18
    spine_norm = torch.norm(spine_final_loc, dim=1, keepdim=True)
    spine_norm = spine_final_loc / (spine_norm + 1e-8) 
    
    final_vec_1 = (rel_pose_skeleton[:, 5].clone() + rel_pose_skeleton[:,9].clone()) / 2.
    final_vec_2 = (rel_pose_skeleton[:, 13].clone() + rel_pose_skeleton[:,17].clone()) / 2.
    
    rest_vec_1 = (rel_rest_skeleton[:, 5].clone() + rel_rest_skeleton[:,9].clone()) / 2.
    rest_vec_2 = (rel_rest_skeleton[:, 13].clone() + rel_rest_skeleton[:,17].clone()) / 2.
     
    center_final_loc = torch.cross(final_vec_1, final_vec_2) 
    center_rest_loc = torch.cross(rest_vec_1, rest_vec_2)


  
    center_final_loc = center_final_loc - torch.sum(center_final_loc * spine_norm, dim=1, keepdim=True) * spine_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * spine_norm, dim=1, keepdim=True) * spine_norm


    
    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat_center


def batch_get_wrist_orient(rel_pose_skeleton, rel_rest_skeleton, parents, children, dtype):
    batch_size = rel_pose_skeleton.shape[0]
    device = rel_pose_skeleton.device
   

    # action-pose
    wrist_final_loc = (rel_pose_skeleton[:, 5-1].clone() + rel_pose_skeleton[:,9-1].clone()) / 2.
    wrist_norm = torch.norm(wrist_final_loc, dim=1, keepdim=True)
    wrist_norm = wrist_final_loc / (wrist_norm + 1e-8)
     
    # T-pose
    wrist_rest_loc = (rel_rest_skeleton[:, 5-1].clone() + rel_rest_skeleton[:,9-1].clone()) / 2.
   
 
    rot_mat_wrist = vectors2rotmat(wrist_rest_loc, wrist_final_loc, dtype) 

    assert torch.sum(torch.isnan(rot_mat_wrist)) == 0, ('rot_mat_spine', rot_mat_wrist)
    
    
    center_final_loc = (rel_pose_skeleton[:, 13-1].clone() + rel_pose_skeleton[:, 17-1].clone()) / 2.
    center_rest_loc = (rel_rest_skeleton[:, 13-1].clone() + rel_rest_skeleton[:, 17-1].clone()) / 2.

    center_rest_loc = torch.matmul(rot_mat_wrist, center_rest_loc) 

  
    center_final_loc = center_final_loc - torch.sum(center_final_loc * wrist_norm, dim=1, keepdim=True) * wrist_norm
    center_rest_loc = center_rest_loc - torch.sum(center_rest_loc * wrist_norm, dim=1, keepdim=True) * wrist_norm

    center_final_loc_norm = torch.norm(center_final_loc, dim=1, keepdim=True)
    center_rest_loc_norm = torch.norm(center_rest_loc, dim=1, keepdim=True)

    # (B, 3, 1)
    axis = torch.cross(center_rest_loc, center_final_loc, dim=1)
    axis_norm = torch.norm(axis, dim=1, keepdim=True)

    # (B, 1, 1)
    cos = torch.sum(center_rest_loc * center_final_loc, dim=1, keepdim=True) / (center_rest_loc_norm * center_final_loc_norm + 1e-8)
    sin = axis_norm / (center_rest_loc_norm * center_final_loc_norm + 1e-8)

    assert torch.sum(torch.isnan(cos)
                     ) == 0, ('cos', cos)
    assert torch.sum(torch.isnan(sin)
                     ) == 0, ('sin', sin)
    # (B, 3, 1)
    axis = axis / (axis_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    # (B, 1, 1)
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat_center = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    rot_mat = torch.matmul(rot_mat_center, rot_mat_wrist)

    return rot_mat




def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3,3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa

def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis

def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

def local_fk(vec_t, local_rotmat, parent):
    batch_size = vec_t.shape[0]
    xyzStruct = [dict() for x in range(len(parent))]
    
    for i in range(len(parent)): 
       
        thisRotation = local_rotmat[:,i,:,:] # B 3,3
         
        if parent[i] == -1:  # root 节点
            xyzStruct[i]['xyz'] = torch.zeros([batch_size,3,1], dtype=torch.float32)
            xyzStruct[i]['rotation'] = thisRotation
        else:
            xyzStruct[i]['xyz'] = xyzStruct[parent[i]]['xyz'] + torch.matmul(xyzStruct[parent[i]]['rotation'], vec_t[:,i-1])
            xyzStruct[i]['rotation'] = torch.matmul(xyzStruct[parent[i]]['rotation'], thisRotation)
              
    xyz = [xyzStruct[i]['xyz'] for i in range(len(parent))]
    xyz = torch.stack(xyz, dim=1)
    
    return xyz