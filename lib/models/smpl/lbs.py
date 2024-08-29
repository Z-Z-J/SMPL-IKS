# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F

def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents,
        lbs_weights, pose2rot=True, return_swing_twist_rotmat=False, return_twist_angle=False, dtype=torch.float32):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        rot_mats: torch.tensor BxJx3x3
            The rotation matrics of each joints
    '''
    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    
  
    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        if pose.numel() == batch_size * 24 * 4:
            rot_mats = quat_to_rotmat(pose.reshape(batch_size * 24, 4)).reshape(batch_size, 24, 3, 3)
        else:
            rot_mats = batch_rodrigues(
                pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, posedirs) \
            .view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents[:24], dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    if return_swing_twist_rotmat:
        swing_rotmat, twist_rotmat = get_swing_twist_rotmat(rot_mats, J.clone(), parents)
    else:
        swing_rotmat, twist_rotmat = None, None
    if return_twist_angle:
        twist_angle = get_twist_angle(rot_mats, J.clone(), parents)
    else:
        twist_angle = None
 
    return verts, J, J_transformed, rot_mats, swing_rotmat, twist_rotmat, twist_angle

def get_swing_twist_rotmat(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # modified by xuchao
    childs = -torch.ones((parents.shape[0]), dtype=parents.dtype, device=parents.device)
    for i in range(1, parents.shape[0]):
        childs[parents[i]] = i
  
    dtype = rot_mats.dtype
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    rotmat_twist = []
    rotmat_swing = []
 
    iden = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device).unsqueeze(0).repeat(batch_size,1,1)
    for i in range(1, parents.shape[0]):
        # modified by xuchao
        if childs[i] < 0:
            rotmat_twist.append(iden)
            rotmat_swing.append(iden)
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
        
        rotmat_twist.append(rot_mat_twist) 
        rotmat_swing.append(rot_mat_pivot)
   
    rotmat_swing = torch.stack(rotmat_swing, dim=1)
    rotmat_twist = torch.stack(rotmat_twist, dim=1)
  
    return rotmat_swing, rotmat_twist

def get_twist(rot_mats, joints, parents):
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # modified by xuchao
    childs = -torch.ones((parents.shape[0]), dtype=parents.dtype, device=parents.device)
    for i in range(1, parents.shape[0]):
        childs[parents[i]] = i

    dtype = rot_mats.dtype
    batch_size = rot_mats.shape[0]
    device = rot_mats.device

    angle_twist = []
    error = False
    for i in range(1, parents.shape[0]):
        # modified by xuchao
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

    return angle_twist

def get_twist_angle(rot_mats, joints, parents):
    twist_angle = []
    for i in range(rot_mats.shape[0]):
        twist_angle.append(get_twist(rot_mats[i:i+1,:], joints[i:i+1,:], parents))
    
    twist_angle = torch.cat(twist_angle, dim=0)
    return twist_angle

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

def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


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


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents, dtype=torch.float32):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints. (Template Pose)
    parents : torch.tensor BxN
        The kinematic tree of each object
    dtype : torch.dtype, optional:
        The data type of the created tensors, the default is torch.float32

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, K + 1, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        # (B, 4, 4) x (B, 4, 4)
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, K + 1, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms



def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat
