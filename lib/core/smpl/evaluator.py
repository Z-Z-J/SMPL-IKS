import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.utils.utils import move_dict_to_device
from lib.ik.smpl.HybrIK import get_body_part_func
from lib.ik.smpl.AnalyIK import SMPL_AnalyIK_V3
from lib.si.SI import SMPL_SI_LR

from lib.utils.si_utils import get_bl_14_from_pos
from lib.ap.smpl.AP import SMPL_AP_V1, SMPL_AP_V2
from lib.core.smpl.config import SMPL_SI_DATA_PATH

logger = logging.getLogger(__name__)


class Evaluator():
    def __init__(
            self,
            test_loaders,
            generator,
            smpl,
            device=None,
        ):
        self.parent = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]) 
        self.children = torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19,
                20, 21, 22, 23, -1, -1]) # [24]
        
        data = np.load(SMPL_SI_DATA_PATH, allow_pickle=True)['lr'].item()
        self.A1 = data['A1'][:,:10].to(device) 
        
        self.valid_loader = test_loaders
        
        self.generator = generator
        
        self.smpl = smpl
        
        self.arm_part_func = get_body_part_func('ARM')
        self.spine_part_func = get_body_part_func('SPINE')
        self.leg_part_func = get_body_part_func('LEG')
                
        self.arm_twist_index = [12,15,17,19,13,16,18,20] 
        self.spine_twist_index =  [2,5,11]
        self.leg_twist_index = [0,3,6,1,4,7]
        
        
        self.device = device
        
        self.evaluation_accumulators = dict.fromkeys(['bone_error','analyik_pos_error','analyik_vert_error','theory_pos_error','theory_vert_error',
                                                              'hybrik_pos_error','hybrik_vert_error'])
        
        if self.device is None:
          self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def validate(self):
        self.generator.eval()
        
        start = time.time()
        
        summary_string = ''
        
        bar = Bar('Validation', fill='#', max=len(self.valid_loader))
        
        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []
        
        for i, target in enumerate(self.valid_loader):
            
            move_dict_to_device(target, self.device)
            
            # <============
            with torch.no_grad():
                # gt
                if target['beta'].shape[-1] != 10:
                    betas = target['beta'][...,:10]
                else:
                    betas = target['beta']
                gt_beta = betas.view(-1, 10)
                gt_theta = target['theta'].view(-1, 72)
                
                # p pos
                smpl_out = self.smpl(gt_theta, gt_beta)
                vert = smpl_out.vertices 
                p_pos = smpl_out.joints.contiguous()
                p_pos = p_pos - p_pos[:,0:1]
                
                gt_bl = get_bl_14_from_pos(p_pos, self.parent)
                
                # t0 pos
                beta_0 = torch.zeros([gt_theta.shape[0],10], dtype=torch.float32).to(self.device) 
                theta_0 = torch.zeros([gt_theta.shape[0],24,3], dtype=torch.float32).to(self.device)
                smpl_out = self.smpl(theta_0, beta_0)
                t0_pos = smpl_out.joints_t.contiguous()
                t0_pos = t0_pos - t0_pos[:,0:1]
                
                # ======================SMPL-SI======================= #
                ## SMPL SI
                pred_beta = SMPL_SI_LR(t0_pos, p_pos, self.parent, self.A1)
                smpl_out = self.smpl(theta_0, pred_beta)
                t_pos = smpl_out.joints_t.contiguous()
                t_pos = t_pos - t_pos[:,0:1]
                
                pred_bl = get_bl_14_from_pos(t_pos, self.parent)    
                
                # ======================SMPL-SI======================= #
                q_pos = SMPL_AP_V1(t_pos, p_pos, self.parent, self.children)
                
                # ======================SMPL-AnalyIK================== #
                analyik_theta = SMPL_AnalyIK_V3(t_pos, q_pos, self.parent, self.children)
                
                smpl_output = self.smpl(analyik_theta, pred_beta)
                analyik_vert = smpl_output.vertices 
                analyik_p_pos = smpl_output.joints.contiguous()
                analyik_p_pos = analyik_p_pos - analyik_p_pos[:,0:1]   
                
                # ========================theory====================== #
                gt_twist_angle = target['twist_angle'].view(-1,23,1) 
                gt_cos = torch.cos(gt_twist_angle)
                gt_sin = torch.sin(gt_twist_angle)
                gt_phi = torch.cat([gt_cos,gt_sin], dim=-1)
                
                arm_gt_phi = gt_phi[:, self.arm_twist_index]
                spine_gt_phi = gt_phi[:, self.spine_twist_index]
                leg_gt_phi = gt_phi[:, self.leg_twist_index]
                
                arm_gt_theta, _ = self.arm_part_func(t_pos, p_pos, analyik_theta, arm_gt_phi)
                spine_gt_theta, _ = self.spine_part_func(t_pos, p_pos, analyik_theta, spine_gt_phi)
                leg_gt_theta, _ = self.leg_part_func(t_pos, p_pos, analyik_theta, leg_gt_phi)
                
                theory_theta = analyik_theta.clone()
                theory_theta = theory_theta.view(-1,24,3)
                theory_theta[:,[13,16,18,20,14,17,19,21]] = arm_gt_theta[:,[13,16,18,20,14,17,19,21]]
                theory_theta[:,[3,6,9,12]] = spine_gt_theta[:,[3,6,9,12]]
                theory_theta[:,[1,4,7,2,5,8]] = leg_gt_theta[:,[1,4,7,2,5,8]]
                
                smpl_out = self.smpl(theory_theta, pred_beta)
                theory_vert = smpl_out.vertices
                theory_p_pos = smpl_out.joints
                theory_p_pos = theory_p_pos - theory_p_pos[:,0:1]
                
                # =====================SMPL-HybrIK======================= #
                theta = analyik_theta.clone()
                theta[:,0] = 0.
                smpl_out = self.smpl(theta, pred_beta)
                tt_pos = smpl_out.joints_t.contiguous()
                tt_pos = tt_pos - tt_pos[:,0:1]
                pp_pos = smpl_out.joints.contiguous()
                pp_pos = pp_pos - pp_pos[:,0:1]
                
                arm_pred_phi, spine_pred_phi, leg_pred_phi = self.generator(pp_pos, pred_beta)
                
                arm_pred_phi = arm_pred_phi / (torch.norm(arm_pred_phi, dim=2, keepdim=True) + 1e-8)
                spine_pred_phi = spine_pred_phi / (torch.norm(spine_pred_phi, dim=2, keepdim=True) + 1e-8)
                leg_pred_phi = leg_pred_phi / (torch.norm(leg_pred_phi, dim=2, keepdim=True) + 1e-8)
                
                arm_pred_theta, _ = self.arm_part_func(t_pos, q_pos, analyik_theta, arm_pred_phi)
                spine_pred_theta, _ = self.spine_part_func(t_pos, q_pos, analyik_theta, spine_pred_phi)
                leg_pred_theta, _ = self.leg_part_func(t_pos, q_pos, analyik_theta, leg_pred_phi)
                
                hybrik_theta = analyik_theta.clone()
                hybrik_theta = hybrik_theta.view(-1,24,3)
                hybrik_theta[:,[13,16,18,20,14,17,19,21]] = arm_pred_theta[:,[13,16,18,20,14,17,19,21]]
                hybrik_theta[:,[3,6,9,12]] = spine_pred_theta[:,[3,6,9,12]]
                hybrik_theta[:,[1,4,7,2,5,8]] = leg_pred_theta[:,[1,4,7,2,5,8]]
                
                smpl_out = self.smpl(hybrik_theta, pred_beta)
                hybrik_vert = smpl_out.vertices
                hybrik_p_pos = smpl_out.joints
                hybrik_p_pos = hybrik_p_pos - hybrik_p_pos[:,0:1] 
                
               
                #--------------------------------------------------------------
                errors_1 = torch.sqrt(((analyik_p_pos - p_pos) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_1 = np.mean(errors_1) 
    
                errors_2 = torch.sqrt(((analyik_vert - vert) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_2 = np.mean(errors_2) 
                
                errors_3 = torch.sqrt(((theory_p_pos - p_pos) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_3 = np.mean(errors_3) 
    
                errors_4 = torch.sqrt(((theory_vert - vert) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_4 = np.mean(errors_4) 
                
                errors_5 = torch.sqrt(((hybrik_p_pos - p_pos) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_5 = np.mean(errors_5)
                
                errors_6 = torch.sqrt(((hybrik_vert - vert) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_6 = np.mean(errors_6)  
                
                errors_7 = torch.abs(gt_bl - pred_bl).mean(dim=-1).cpu().numpy()
                errors_7 = np.mean(errors_7)
                
                self.evaluation_accumulators['analyik_pos_error'].append(errors_1)
                self.evaluation_accumulators['analyik_vert_error'].append(errors_2) 
                
                self.evaluation_accumulators['theory_pos_error'].append(errors_3)
                self.evaluation_accumulators['theory_vert_error'].append(errors_4) 
                
                self.evaluation_accumulators['hybrik_pos_error'].append(errors_5)
                self.evaluation_accumulators['hybrik_vert_error'].append(errors_6)
                
                self.evaluation_accumulators['bone_error'].append(errors_7)
                
            # ============>
            
            batch_time = time.time() - start
            
            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                            f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            bar.suffix = summary_string
            bar.next()
            
        bar.finish()
        
        logger.info(summary_string)
    
    def evaluate(self):
        for k, v in self.evaluation_accumulators.items():
           self.evaluation_accumulators[k] = np.vstack(v)
          
        analyik_pos_error = self.evaluation_accumulators['analyik_pos_error']
        analyik_vert_error = self.evaluation_accumulators['analyik_vert_error']
        
        theory_pos_error = self.evaluation_accumulators['theory_pos_error']
        theory_vert_error = self.evaluation_accumulators['theory_vert_error'] 
        
        hybrik_pos_error = self.evaluation_accumulators['hybrik_pos_error']
        hybrik_vert_error = self.evaluation_accumulators['hybrik_vert_error']
        
        bone_error = self.evaluation_accumulators['bone_error']
        
        analyik_pos_error = torch.from_numpy(analyik_pos_error).float()
        analyik_vert_error = torch.from_numpy(analyik_vert_error).float()
        
        theory_pos_error = torch.from_numpy(theory_pos_error).float()
        theory_vert_error = torch.from_numpy(theory_vert_error).float()
        
        hybrik_pos_error = torch.from_numpy(hybrik_pos_error).float()
        hybrik_vert_error = torch.from_numpy(hybrik_vert_error).float() 
        
        bone_error =  torch.from_numpy(bone_error).float() 
        
        m2mm = 1000
        
        # MPJPE MPVE
        analyik_mpjpe = torch.mean(analyik_pos_error).cpu().numpy() * m2mm
        analyik_mpve = torch.mean(analyik_vert_error).cpu().numpy() * m2mm
       
        theory_mpjpe = torch.mean(theory_pos_error).cpu().numpy() * m2mm
        theory_mpve = torch.mean(theory_vert_error).cpu().numpy() * m2mm     
       
        hybrik_mpjpe = torch.mean(hybrik_pos_error).cpu().numpy() * m2mm
        hybrik_mpve = torch.mean(hybrik_vert_error).cpu().numpy() * m2mm
        
        bone_mpbe =  torch.mean(bone_error).cpu().numpy() * m2mm
        
        eval_dict = {
            'bone_mpbe': bone_mpbe,
            'analyik_mpjpe': analyik_mpjpe,
            'analyik_mpve': analyik_mpve,
            'theory_mpjpe': theory_mpjpe,
            'theory_mpve': theory_mpve,  
            'hybrik_mpjpe': hybrik_mpjpe,
            'hybrik_mpve': hybrik_mpve, 
            }
        
        log_str = ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        print(log_str)
    
    def run(self):
        self.validate()
        self.evaluate()
        
        
        
        
        
        
