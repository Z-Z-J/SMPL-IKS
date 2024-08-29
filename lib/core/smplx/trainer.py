# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.utils.ik_utils import batch_rodrigues, rotation_matrix_to_angle_axis
from lib.ik.smplx.AnalyIK import SMPLX_AnalyIK_V1
from lib.ik.smplx.HybrIK import get_body_part_func
from lib.utils.utils import move_dict_to_device, AverageMeter

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            optimizer,
            criterion,
            smplx,
            
            start_epoch,
            end_epoch,
            lr_scheduler=None,
            device=None,
            writer=None,
            logdir='output',
            performance_type='min',
    ):
        
        # Prepare dataloaders
        self.train_loader, self.valid_loader = data_loaders
        
        # Models and optimizers
        self.generator = generator
        self.optimizer = optimizer
        
        self.smplx = smplx
        
        self.jts_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
                           20, 37, 38, 39, 66, 25, 26, 27, 67, 28, 29, 30, 68, 34, 35, 36, 69, 31, 32, 33, 70, 
                           21, 52, 53, 54, 71, 40, 41, 42, 72, 43, 44, 45, 73, 49, 50, 51, 74, 46, 47, 48, 75]
        
        self.leg_twist_index = [0,3,6,1,4,7]
        self.spine_twist_index = [2,5,11]
        self.arm_twist_index = [12,15,17,13,16,18,
                                    36,37,38,24,25,26,27,28,29,33,34,35,30,31,32,
                                    51,52,53,39,40,41,42,43,44,48,49,50,45,46,47]
        
        self.leg_refine_func = get_body_part_func('LEG')
        self.spine_refine_func = get_body_part_func('SPINE')
        self.arm_refine_func = get_body_part_func('ARM')
        
        # Training parameters
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        
        self.device = device
        self.writer = writer
        self.logdir = logdir
        
        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        
     
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')
        
        self.evaluation_accumulators = dict.fromkeys(['analyik_pos_error','analyik_vert_error','theory_pos_error','theory_vert_error',
                                                              'hybrik_pos_error','hybrik_vert_error'])
        
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)
        
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def train(self):
        # Single epoch training routine
        
        losses = AverageMeter()
        
        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
            }
        
        self.generator.train()
        
        start = time.time()
        
        summary_string = ''
        
        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=len(self.train_loader))
        self.num_iters_per_epoch = len(self.train_loader)
        
        for i, target in enumerate(self.train_loader):
             
            move_dict_to_device(target, self.device)    
            
            timer['data'] = time.time() - start
            
            # <======== Feedforward generator 
            start = time.time()
            
            ### P-pos 
            betas = target['betas'].view(-1, 10).contiguous()
            gt_global_orient = target['global_orient'].view(-1, 1, 3).contiguous()
            gt_body_pose = target['body_pose'].view(-1, 21, 3).contiguous()
            gt_lhand_pose = target['lhand_pose'].view(-1, 15, 3).contiguous()
            gt_rhand_pose = target['rhand_pose'].view(-1, 15, 3).contiguous()
            
            gt_global_orient_rotmat = batch_rodrigues(gt_global_orient.view(-1, 3)).view(-1, 1, 3, 3)
            gt_body_pose_rotmat = batch_rodrigues(gt_body_pose.view(-1,3)).view(-1, 21, 3, 3)
            gt_lhand_pose_rotmat = batch_rodrigues(gt_lhand_pose.view(-1,3)).view(-1, 15, 3, 3)
            gt_rhand_pose_rotmat = batch_rodrigues(gt_rhand_pose.view(-1,3)).view(-1, 15, 3, 3)
            
            bs = gt_global_orient_rotmat.shape[0]
            beta_0 = torch.zeros([bs, 10], dtype=torch.float32).to(self.device)
            expression_0 = torch.zeros([bs, 10], dtype=torch.float32).to(self.device)
            gt_global_orient_rotmat_0 = batch_rodrigues(torch.zeros([bs, 1, 3], dtype=torch.float32).view(-1, 3)).view(-1, 1, 3, 3).to(self.device)
            jaw_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
            leye_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
            reye_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
            
            smplx_out = self.smplx(
                betas = betas,
                global_orient = gt_global_orient_rotmat_0, body_pose = gt_body_pose_rotmat,
                left_hand_pose = gt_lhand_pose_rotmat, right_hand_pose = gt_rhand_pose_rotmat,
                expression = expression_0, jaw_pose = jaw_pose_0,
                leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                )
            p_pos = smplx_out.joints.contiguous()                                 # [b,24,3]
            p_pos = p_pos - p_pos[:,0:1]
            verts = smplx_out.vertices.contiguous()
            
            ### T-pos
            global_orient_rotmat_0 = batch_rodrigues(torch.zeros([bs, 1, 3], dtype=torch.float32).view(-1, 3)).view(-1, 1, 3, 3).to(self.device)
            body_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 21, 3], dtype=torch.float32).view(-1, 3)).view(-1, 21, 3, 3).to(self.device)
            lhand_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 15, 3], dtype=torch.float32).view(-1, 3)).view([-1, 15, 3, 3]).to(self.device)
            rhand_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 15, 3], dtype=torch.float32).view(-1, 3)).view([-1, 15, 3, 3]).to(self.device)
            
            smplx_out = self.smplx(
                betas = betas,
                global_orient = global_orient_rotmat_0, body_pose = body_pose_rotmat_0,
                left_hand_pose = lhand_pose_rotmat_0, right_hand_pose = rhand_pose_rotmat_0,
                expression = expression_0, jaw_pose = jaw_pose_0,
                leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                )
            t_pos = smplx_out.joints.contiguous()
            t_pos = t_pos - t_pos[:,0:1]
            
            ### twist phi
            
            gt_leg_twist_angle = target['twist_angle'][:,0,self.leg_twist_index]
            gt_leg_twist_cos = torch.cos(gt_leg_twist_angle) 
            gt_leg_twist_sin = torch.sin(gt_leg_twist_angle)
            gt_leg_twist_phi = torch.cat([gt_leg_twist_cos,gt_leg_twist_sin], dim=-1)
            
            gt_spine_twist_angle = target['twist_angle'][:,0,self.spine_twist_index]
            gt_spine_twist_cos = torch.cos(gt_spine_twist_angle) 
            gt_spine_twist_sin = torch.sin(gt_spine_twist_angle)
            gt_spine_twist_phi = torch.cat([gt_spine_twist_cos,gt_spine_twist_sin], dim=-1)
            
            gt_arm_hand_twist_angle = target['twist_angle'][:,0,self.arm_twist_index]
            gt_arm_hand_twist_cos = torch.cos(gt_arm_hand_twist_angle) 
            gt_arm_hand_twist_sin = torch.sin(gt_arm_hand_twist_angle)
            gt_arm_hand_twist_phi = torch.cat([gt_arm_hand_twist_cos,gt_arm_hand_twist_sin], dim=-1)
    
            ### SMPLX_AnalyIK
            analyik_global_orient_rotmat, analyik_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat = SMPLX_AnalyIK_V1(t_pos, p_pos)
           
            smplx_out = self.smplx(
                betas = betas,
                global_orient = analyik_global_orient_rotmat, body_pose = analyik_body_pose_rotmat,
                left_hand_pose = analyik_lhand_pose_rotmat, right_hand_pose = analyik_rhand_pose_rotmat,
                expression = expression_0, jaw_pose = jaw_pose_0,
                leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                )
            analyik_p_pos = smplx_out.joints.contiguous() 
            analyik_p_pos = analyik_p_pos - analyik_p_pos[:,0:1]
            analyik_verts = smplx_out.vertices.contiguous() 
            
            
            ### 理论上限
            theory_body_pose_rotmat = self.leg_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, gt_leg_twist_phi)
            theory_body_pose_rotmat = self.spine_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, theory_body_pose_rotmat, gt_spine_twist_phi)
            theory_body_pose_rotmat, theory_lhand_pose_rotmat, theory_rhand_pose_rotmat = self.arm_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, 
                                                                        theory_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat, gt_arm_hand_twist_phi)
            
            smplx_out = self.smplx(
                betas = betas,
                global_orient = analyik_global_orient_rotmat, body_pose = theory_body_pose_rotmat,
                left_hand_pose = theory_lhand_pose_rotmat, right_hand_pose = theory_rhand_pose_rotmat,
                expression = expression_0, jaw_pose = jaw_pose_0,
                leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                )
            theory_p_pos = smplx_out.joints.contiguous()
            theory_p_pos = theory_p_pos - theory_p_pos[:,0:1]
            theory_verts = smplx_out.vertices.contiguous() 
            
            
            ### SMPLX_HybrIK
            pred_leg_twist_phi, pred_spine_twist_phi, pred_arm_hand_twist_phi = self.generator(p_pos, betas)
            pred_leg_twist_phi = pred_leg_twist_phi / (torch.norm(pred_leg_twist_phi, dim=2, keepdim=True) + 1e-8)
            pred_spine_twist_phi = pred_spine_twist_phi / (torch.norm(pred_spine_twist_phi, dim=2, keepdim=True) + 1e-8)
            pred_arm_hand_twist_phi = pred_arm_hand_twist_phi / (torch.norm(pred_arm_hand_twist_phi, dim=2, keepdim=True) + 1e-8)
            
            hybrik_body_pose_rotmat = self.leg_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, pred_leg_twist_phi)
            hybrik_body_pose_rotmat = self.spine_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, hybrik_body_pose_rotmat, pred_spine_twist_phi)
            hybrik_body_pose_rotmat, hybrik_lhand_pose_rotmat, hybrik_rhand_pose_rotmat = self.arm_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, 
                                                                        hybrik_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat, pred_arm_hand_twist_phi)
            
            smplx_out = self.smplx(
                betas = betas,
                global_orient = analyik_global_orient_rotmat, body_pose = hybrik_body_pose_rotmat,
                left_hand_pose = hybrik_lhand_pose_rotmat, right_hand_pose = hybrik_rhand_pose_rotmat,
                expression = expression_0, jaw_pose = jaw_pose_0,
                leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                )
            hybrik_p_pos = smplx_out.joints.contiguous()
            hybrik_p_pos = hybrik_p_pos - hybrik_p_pos[:,0:1]
            hybrik_verts = smplx_out.vertices.contiguous() 
            
            
            hybrik_body_pose = rotation_matrix_to_angle_axis(hybrik_body_pose_rotmat.view(-1,3,3)).view(-1,21,3).contiguous()
            hybrik_lhand_pose = rotation_matrix_to_angle_axis(hybrik_lhand_pose_rotmat.view(-1,3,3)).view(-1,15,3).contiguous()
            hybrik_rhand_pose = rotation_matrix_to_angle_axis(hybrik_rhand_pose_rotmat.view(-1,3,3)).view(-1,15,3).contiguous()
            
            
            timer['forward'] = time.time() - start
            start = time.time()
            
            gen_loss, loss_dict = self.criterion(
                p_pos[:,self.jts_index],
                verts,
                analyik_p_pos[:,self.jts_index],
                analyik_verts,
                theory_p_pos[:,self.jts_index],
                theory_verts,
                hybrik_p_pos[:,self.jts_index],
                hybrik_verts,
                
                gt_leg_twist_phi,
                gt_spine_twist_phi,
                gt_arm_hand_twist_phi,
                pred_leg_twist_phi,
                pred_spine_twist_phi,
                pred_arm_hand_twist_phi, 
                
                gt_body_pose,
                gt_lhand_pose,
                gt_rhand_pose,
                hybrik_body_pose,
                hybrik_lhand_pose,
                hybrik_rhand_pose,
                )
            
            timer['loss'] = time.time() - start
            
            # <======== Backprop generator
            start = time.time()
            self.optimizer.zero_grad()
            gen_loss.backward()
            self.optimizer.step()
            
            # <======== Log training info
            total_loss = gen_loss
            losses.update(total_loss.item(), bs)
            
            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()
            
            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | Total: {bar.elapsed_td} | ' \
                            f'ETA: {bar.eta_td:} | loss: {losses.avg:.4f}'
            
            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.4f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)
            
            for k,v in timer.items():
                summary_string += f' | {k}: {v:.4f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)
            
            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()
            
            if torch.isnan(total_loss):
                exit('Nan value in loss, exiting!...')
            # =======>
        
        bar.finish()
        
        logger.info(summary_string)
    
    
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
                ### P-pos 
                betas = target['betas'].view(-1, 10).contiguous()
                gt_global_orient = target['global_orient'].view(-1, 1, 3).contiguous()
                gt_body_pose = target['body_pose'].view(-1, 21, 3).contiguous()
                gt_lhand_pose = target['lhand_pose'].view(-1, 15, 3).contiguous()
                gt_rhand_pose = target['rhand_pose'].view(-1, 15, 3).contiguous()
                
                gt_global_orient_rotmat = batch_rodrigues(gt_global_orient.view(-1, 3)).view(-1, 1, 3, 3)
                gt_body_pose_rotmat = batch_rodrigues(gt_body_pose.view(-1,3)).view(-1, 21, 3, 3)
                gt_lhand_pose_rotmat = batch_rodrigues(gt_lhand_pose.view(-1,3)).view(-1, 15, 3, 3)
                gt_rhand_pose_rotmat = batch_rodrigues(gt_rhand_pose.view(-1,3)).view(-1, 15, 3, 3)
                
                bs = gt_global_orient_rotmat.shape[0]
                beta_0 = torch.zeros([bs, 10], dtype=torch.float32).to(self.device)
                expression_0 = torch.zeros([bs, 10], dtype=torch.float32).to(self.device)
                gt_global_orient_rotmat_0 = batch_rodrigues(torch.zeros([bs, 1, 3], dtype=torch.float32).view(-1, 3)).view(-1, 1, 3, 3).to(self.device)
                jaw_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
                leye_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
                reye_pose_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view([-1, 3, 3]).to(self.device)
                
                smplx_out = self.smplx(
                    betas = betas,
                    global_orient = gt_global_orient_rotmat_0, body_pose = gt_body_pose_rotmat,
                    left_hand_pose = gt_lhand_pose_rotmat, right_hand_pose = gt_rhand_pose_rotmat,
                    expression = expression_0, jaw_pose = jaw_pose_0,
                    leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                    )
                p_pos = smplx_out.joints.contiguous()                                 # [b,24,3]
                p_pos = p_pos - p_pos[:,0:1]
                verts = smplx_out.vertices.contiguous()
                
                ### T-pos
                global_orient_rotmat_0 = batch_rodrigues(torch.zeros([bs, 3], dtype=torch.float32)).view(-1, 1, 3, 3).to(self.device)
                body_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 21, 3], dtype=torch.float32).view(-1,3)).view(-1, 21, 3, 3).to(self.device)
                lhand_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 15, 3], dtype=torch.float32).view(-1, 3)).view([-1, 15, 3, 3]).to(self.device)
                rhand_pose_rotmat_0 = batch_rodrigues(torch.zeros([bs, 15, 3], dtype=torch.float32).view(-1, 3)).view([-1, 15, 3, 3]).to(self.device)
                
                smplx_out = self.smplx(
                    betas = betas,
                    global_orient = global_orient_rotmat_0, body_pose = body_pose_rotmat_0,
                    left_hand_pose = lhand_pose_rotmat_0, right_hand_pose = rhand_pose_rotmat_0,
                    expression = expression_0, jaw_pose = jaw_pose_0,
                    leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                    )
                t_pos = smplx_out.joints.contiguous()
                t_pos = t_pos - t_pos[:,0:1]
                
                ### twist phi
                gt_leg_twist_angle = target['twist_angle'][:,0,self.leg_twist_index]
                gt_leg_twist_cos = torch.cos(gt_leg_twist_angle) 
                gt_leg_twist_sin = torch.sin(gt_leg_twist_angle)
                gt_leg_twist_phi = torch.cat([gt_leg_twist_cos,gt_leg_twist_sin], dim=-1)
                
                gt_spine_twist_angle = target['twist_angle'][:,0,self.spine_twist_index]
                gt_spine_twist_cos = torch.cos(gt_spine_twist_angle) 
                gt_spine_twist_sin = torch.sin(gt_spine_twist_angle)
                gt_spine_twist_phi = torch.cat([gt_spine_twist_cos,gt_spine_twist_sin], dim=-1)
                
                gt_arm_hand_twist_angle = target['twist_angle'][:,0,self.arm_twist_index]
                gt_arm_hand_twist_cos = torch.cos(gt_arm_hand_twist_angle) 
                gt_arm_hand_twist_sin = torch.sin(gt_arm_hand_twist_angle)
                gt_arm_hand_twist_phi = torch.cat([gt_arm_hand_twist_cos,gt_arm_hand_twist_sin], dim=-1)
    
                ### SMPLX_AnalyIK
                analyik_global_orient_rotmat, analyik_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat = SMPLX_AnalyIK_V1(t_pos, p_pos)
               
                smplx_out = self.smplx(
                    betas = betas,
                    global_orient = analyik_global_orient_rotmat, body_pose = analyik_body_pose_rotmat,
                    left_hand_pose = analyik_lhand_pose_rotmat, right_hand_pose = analyik_rhand_pose_rotmat,
                    expression = expression_0, jaw_pose = jaw_pose_0,
                    leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                    )
                analyik_p_pos = smplx_out.joints.contiguous() 
                analyik_p_pos = analyik_p_pos - analyik_p_pos[:,0:1]
                analyik_verts = smplx_out.vertices.contiguous() 
                
                ### 理论上限
                theory_body_pose_rotmat = self.leg_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, gt_leg_twist_phi)
                theory_body_pose_rotmat = self.spine_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, theory_body_pose_rotmat, gt_spine_twist_phi)
                theory_body_pose_rotmat, theory_lhand_pose_rotmat, theory_rhand_pose_rotmat = self.arm_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, 
                                                                            theory_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat, gt_arm_hand_twist_phi)
                
                smplx_out = self.smplx(
                    betas = betas,
                    global_orient = analyik_global_orient_rotmat, body_pose = theory_body_pose_rotmat,
                    left_hand_pose = theory_lhand_pose_rotmat, right_hand_pose = theory_rhand_pose_rotmat,
                    expression = expression_0, jaw_pose = jaw_pose_0,
                    leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                    )
                theory_p_pos = smplx_out.joints.contiguous()
                theory_p_pos = theory_p_pos - theory_p_pos[:,0:1]
                theory_verts = smplx_out.vertices.contiguous() 
                
                ### SMPLX_HybrIK
                pred_leg_twist_phi, pred_spine_twist_phi, pred_arm_hand_twist_phi = self.generator(p_pos, betas)
                pred_leg_twist_phi = pred_leg_twist_phi / (torch.norm(pred_leg_twist_phi, dim=2, keepdim=True) + 1e-8)
                pred_spine_twist_phi = pred_spine_twist_phi / (torch.norm(pred_spine_twist_phi, dim=2, keepdim=True) + 1e-8)
                pred_arm_hand_twist_phi = pred_arm_hand_twist_phi / (torch.norm(pred_arm_hand_twist_phi, dim=2, keepdim=True) + 1e-8)
                
                hybrik_body_pose_rotmat = self.leg_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, analyik_body_pose_rotmat, pred_leg_twist_phi)
                hybrik_body_pose_rotmat = self.spine_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, hybrik_body_pose_rotmat, pred_spine_twist_phi)
                hybrik_body_pose_rotmat, hybrik_lhand_pose_rotmat, hybrik_rhand_pose_rotmat = self.arm_refine_func(t_pos, p_pos, analyik_global_orient_rotmat, 
                                                                            hybrik_body_pose_rotmat, analyik_lhand_pose_rotmat, analyik_rhand_pose_rotmat, pred_arm_hand_twist_phi)
                
                smplx_out = self.smplx(
                    betas = betas,
                    global_orient = analyik_global_orient_rotmat, body_pose = hybrik_body_pose_rotmat,
                    left_hand_pose = hybrik_lhand_pose_rotmat, right_hand_pose = hybrik_rhand_pose_rotmat,
                    expression = expression_0, jaw_pose = jaw_pose_0,
                    leye_pose = leye_pose_0, reye_pose = reye_pose_0,
                    )
                hybrik_p_pos = smplx_out.joints.contiguous()
                hybrik_p_pos = hybrik_p_pos - hybrik_p_pos[:,0:1]
                hybrik_verts = smplx_out.vertices.contiguous() 
              
                #--------------------------------------------------------------
                errors_1 = torch.sqrt(((analyik_p_pos[:,self.jts_index] - p_pos[:,self.jts_index]) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_1 = np.mean(errors_1) 
    
                errors_2 = torch.sqrt(((analyik_verts - verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_2 = np.mean(errors_2) 
                
                errors_3 = torch.sqrt(((theory_p_pos[:,self.jts_index] - p_pos[:,self.jts_index]) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_3 = np.mean(errors_3) 
    
                errors_4 = torch.sqrt(((theory_verts - verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_4 = np.mean(errors_4) 
                
                errors_5 = torch.sqrt(((hybrik_p_pos[:,self.jts_index] - p_pos[:,self.jts_index]) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_5 = np.mean(errors_5)
                
                errors_6 = torch.sqrt(((hybrik_verts - verts) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                errors_6 = np.mean(errors_6)  
                
                self.evaluation_accumulators['analyik_pos_error'].append(errors_1)
                self.evaluation_accumulators['analyik_vert_error'].append(errors_2) 
                
                self.evaluation_accumulators['theory_pos_error'].append(errors_3)
                self.evaluation_accumulators['theory_vert_error'].append(errors_4) 
                
                self.evaluation_accumulators['hybrik_pos_error'].append(errors_5)
                self.evaluation_accumulators['hybrik_vert_error'].append(errors_6)
                
            # ============>
            
            batch_time = time.time() - start
            
            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                            f'Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'

            self.valid_global_step += 1
            bar.suffix = summary_string
            bar.next()
            
        bar.finish()
        
        logger.info(summary_string)
    
    
    def evaluate(self):
        
        for k,v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)
        
        analyik_pos_error = self.evaluation_accumulators['analyik_pos_error']
        analyik_vert_error = self.evaluation_accumulators['analyik_vert_error']
        
        theory_pos_error = self.evaluation_accumulators['theory_pos_error']
        theory_vert_error = self.evaluation_accumulators['theory_vert_error'] 
        
        hybrik_pos_error = self.evaluation_accumulators['hybrik_pos_error']
        hybrik_vert_error = self.evaluation_accumulators['hybrik_vert_error']

        analyik_pos_error = torch.from_numpy(analyik_pos_error).float()
        analyik_vert_error = torch.from_numpy(analyik_vert_error).float()
        
        theory_pos_error = torch.from_numpy(theory_pos_error).float()
        theory_vert_error = torch.from_numpy(theory_vert_error).float()
        
        hybrik_pos_error = torch.from_numpy(hybrik_pos_error).float()
        hybrik_vert_error = torch.from_numpy(hybrik_vert_error).float() 
        
        m2mm = 1000
        
        # MPJPE MPVE
        analyik_mpjpe = torch.mean(analyik_pos_error).cpu().numpy() * m2mm
        analyik_mpve = torch.mean(analyik_vert_error).cpu().numpy() * m2mm
       
        theory_mpjpe = torch.mean(theory_pos_error).cpu().numpy() * m2mm
        theory_mpve = torch.mean(theory_vert_error).cpu().numpy() * m2mm     
       
        hybrik_mpjpe = torch.mean(hybrik_pos_error).cpu().numpy() * m2mm
        hybrik_mpve = torch.mean(hybrik_vert_error).cpu().numpy() * m2mm
        
        eval_dict = {
            'analyik_mpjpe': analyik_mpjpe,
            'analyik_mpve': analyik_mpve,
            'theory_mpjpe': theory_mpjpe,
            'theory_mpve': theory_mpve,  
            'hybrik_mpjpe': hybrik_mpjpe,
            'hybrik_mpve': hybrik_mpve, 
            }
        
        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)
        
        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return hybrik_mpve
    
    
    def fit(self):
        
        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)
            
            # log the learning rate
            for param_group in self.optimizer.param_groups:
                print(f'Learning rate {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)
            
            logger.info(f'Epoch {epoch+1} performance: {performance:.4f}')
            
            self.save_model(performance, epoch)
        
        self.writer.close()
    
    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performace': performance,
            'gen_optimizer': self.optimizer.state_dict(),
            }            
        
        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance
        
        if is_best:
            logger.info('Best performance achived, saving it!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))
            
            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))
                
                
                