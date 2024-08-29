from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn

from lib.models.smpl.lbs import lbs

try:
    import cPickle as pk
except ImportError:
    import pickle as pk


ModelOutput = namedtuple('ModelOutput', 
                         ['vertices', 'joints_t', 'joints',
                          'rot_mats', 'swing_rotmat', 'twist_rotmat', 'twist_angle'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPLLayer(nn.Module):
 
    def __init__(self,
                 model_path,
                 kid_template_path,
                 gender='neutral',
                 age='adult',
                 dtype=torch.float32,
                ):
        ''' SMPL model layers

        Parameters:
        ----------
        model_path: str
            The path to the folder or to the file where the model
            parameters are stored
        gender: str, optional
            Which gender to load
        '''
        super(SMPLLayer, self).__init__()


        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = Struct(**pk.load(smpl_file, encoding='latin1'))

        self.gender = gender
        
        self.age = age
        
        self.dtype = dtype

        self.faces = self.smpl_data.f

        ''' Register Buffer '''
        # Faces
        self.register_buffer('faces_tensor',
                             to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # The vertices of the template model, (6890, 3)
        self.register_buffer('v_template',
                             to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))
        
        # kids shapedirs
        shapedirs = self.smpl_data.shapedirs
        if self.age == 'kid':
            v_template_smil = np.load(kid_template_path)
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(
              v_template_smil - self.smpl_data.v_template, axis=2)
            shapedirs = np.concatenate(
                (shapedirs[:, :, :10], v_template_diff), axis=2)
        
        # The shape components
        # Shape blend shapes basis, (6890, 3, 10)
        self.register_buffer(
            'shapedirs',
            to_tensor(to_np(shapedirs), dtype=dtype))

        # Pose blend shape basis: 6890 x 3 x 23*9, reshaped to 6890*3 x 23*9
        num_pose_basis = self.smpl_data.posedirs.shape[-1]
        # 23*9 x 6890*3
        posedirs = np.reshape(self.smpl_data.posedirs, [-1, num_pose_basis]).T
     
        self.register_buffer('posedirs',
                             to_tensor(to_np(posedirs), dtype=dtype))

        # Vertices to Joints location (23 + 1, 6890)
        self.register_buffer(
            'J_regressor',
            to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))
        
        # indices of parents for each joints
        parents = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
      
        # (24,)
        self.register_buffer('parents', parents)

        # (6890, 23 + 1)
        self.register_buffer('lbs_weights',
                             to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

    def forward(self,
                body_pose,
                betas,
                global_orient=None,
                transl=None,
                return_swing_twist_rotmat=False,
                return_twist_angle=False):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            pose_axis_angle: torch.tensor, optional, shape Bx(J*3)
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            betas: torch.tensor, optional, shape Bx10
                It can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            global_orient: torch.tensor, optional, shape Bx3
                Global Orientations.
            transl: torch.tensor, optional, shape Bx3
                Global Translations.
            return_verts: bool, optional
                Return the vertices. (default=True)
            Returns
            -------
        '''
        # batch_size = pose_axis_angle.shape[0]
        
        if global_orient is not None:
           full_pose = torch.cat([global_orient, body_pose], dim=1)
        else:
           full_pose = body_pose
           
        if full_pose.shape[-1] == 9:
            pose2rot = False
        elif full_pose.shape[-1] == 3 and full_pose.shape[-2] == 3 and full_pose.shape[-3] == 24:
            pose2rot = False
        else:
            pose2rot = True
     
        # vertices: (B, N, 3), joints: (B, K, 3)
        vertices, joints_t, joints, rot_mats, swing_rotmat, twist_rotmat, twist_angle = lbs(betas, full_pose, self.v_template,
                                                                 self.shapedirs, self.posedirs,
                                                                 self.J_regressor, self.parents,
                                                                 self.lbs_weights, pose2rot=pose2rot, return_swing_twist_rotmat=return_swing_twist_rotmat, 
                                                                 return_twist_angle=return_twist_angle, dtype=self.dtype)

        output = ModelOutput(
            vertices=vertices, joints_t=joints_t, joints=joints, rot_mats=rot_mats, swing_rotmat=swing_rotmat, twist_rotmat=twist_rotmat, twist_angle=twist_angle)
        return output

    