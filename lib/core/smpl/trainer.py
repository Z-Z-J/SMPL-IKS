import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.utils.utils import move_dict_to_device, AverageMeter
from lib.ik.smpl.HybrIK import get_body_part_func
from lib.ik.smpl.AnalyIK import SMPL_AnalyIK_V3

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            gen_optimizer,
            criterion,
            smpl,
           
            start_epoch,
            end_epoch,
            lr_scheduler=None,
            device=None,
            writer=None,
            logdir='output',
            performance_type='min',
    ):
        
        self.parent = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 20, 21]) 
        self.children = torch.tensor([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19,
                20, 21, 22, 23, -1, -1]) # [24]
        
        # Prepare dataloaders
        self.train_loader, self.valid_loader = data_loaders
        
        # Models and optimizers
        self.generator = generator
        self.gen_optimizer = gen_optimizer
        
        self.smpl = smpl
        
        self.leg_part_func = get_body_part_func('LEG')
        self.spine_part_func = get_body_part_func('SPINE')
        self.arm_part_func = get_body_part_func('ARM')
        
        self.leg_twist_index = [0,3,6,1,4,7]
        self.spine_twist_index = [2,5,11]
        self.arm_twist_index = [12,15,17,19,13,16,18,20]
    
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
            
            # gt 
            gt_theta = target['theta'].view(-1, 72)
            gt_theta[:,:3] = 0.
            beta = target['beta'].view(-1, 10)
            smpl_out = self.smpl(gt_theta, beta)
            vert = smpl_out.vertices
            t_pos = smpl_out.joints_t.contiguous()
            t_pos = t_pos - t_pos[:,0:1]
            p_pos = smpl_out.joints.contiguous()                                 # [b,24,3]
            p_pos = p_pos - p_pos[:,0:1]
            
            # analyik
            analyik_theta = SMPL_AnalyIK_V3(t_pos, p_pos, self.parent, self.children)
            smpl_output = self.smpl(analyik_theta, beta)
            analyik_vert = smpl_output.vertices 
            analyik_p_pos = smpl_output.joints.contiguous()
            analyik_p_pos = analyik_p_pos - analyik_p_pos[:,0:1]   
            
            # theory
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
            
            smpl_out = self.smpl(theory_theta, beta)
            theory_vert = smpl_out.vertices
            theory_p_pos = smpl_out.joints
            theory_p_pos = theory_p_pos - theory_p_pos[:,0:1]
            
            # hybrik
            arm_pred_phi, spine_pred_phi, leg_pred_phi = self.generator(p_pos, beta)
            
            arm_pred_phi = arm_pred_phi / (torch.norm(arm_pred_phi, dim=2, keepdim=True) + 1e-8)
            spine_pred_phi = spine_pred_phi / (torch.norm(spine_pred_phi, dim=2, keepdim=True) + 1e-8)
            leg_pred_phi = leg_pred_phi / (torch.norm(leg_pred_phi, dim=2, keepdim=True) + 1e-8)
            
            arm_pred_theta, _ = self.arm_part_func(t_pos, p_pos, analyik_theta, arm_pred_phi)
            spine_pred_theta, _ = self.spine_part_func(t_pos, p_pos, analyik_theta, spine_pred_phi)
            leg_pred_theta, _ = self.leg_part_func(t_pos, p_pos, analyik_theta, leg_pred_phi)
            
            hybrik_theta = analyik_theta.clone()
            hybrik_theta = hybrik_theta.view(-1,24,3)
            hybrik_theta[:,[13,16,18,20,14,17,19,21]] = arm_pred_theta[:,[13,16,18,20,14,17,19,21]]
            hybrik_theta[:,[3,6,9,12]] = spine_pred_theta[:,[3,6,9,12]]
            hybrik_theta[:,[1,4,7,2,5,8]] = leg_pred_theta[:,[1,4,7,2,5,8]]
            
            smpl_out = self.smpl(hybrik_theta, beta)
            hybrik_vert = smpl_out.vertices
            hybrik_p_pos = smpl_out.joints
            hybrik_p_pos = hybrik_p_pos - hybrik_p_pos[:,0:1]
         
            timer['forward'] = time.time() - start
            start = time.time()
            
            gen_loss, loss_dict = self.criterion(
                p_pos,
                vert,
                analyik_p_pos,
                analyik_vert,
                theory_p_pos,
                theory_vert,
                hybrik_p_pos,
                hybrik_vert,
                
                arm_gt_phi,
                spine_gt_phi,
                leg_gt_phi,
                arm_pred_phi,
                spine_pred_phi,
                leg_pred_phi, 
                
                arm_gt_theta,
                spine_gt_theta,
                leg_gt_theta,
                arm_pred_theta,
                spine_pred_theta,
                leg_pred_theta,
                )
            
            timer['loss'] = time.time() - start
            
            # <======== Backprop generator
            start = time.time()
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()
            
            # <======== Log training info
            total_loss = gen_loss
            losses.update(total_loss.item(), p_pos.size(0))
            
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
                # gt 
                gt_theta = target['theta'].view(-1, 72)
                gt_theta[:,:3] = 0.
                if target['beta'].shape[-1] != 10:
                    betas = target['beta'][...,:10]
                else:
                    betas = target['beta']
                beta = betas.view(-1, 10)
                smpl_out = self.smpl(gt_theta, beta)
                vert = smpl_out.vertices
                t_pos = smpl_out.joints_t.contiguous()
                t_pos = t_pos - t_pos[:,0:1]
                p_pos = smpl_out.joints.contiguous()                                 # [b,24,3]
                p_pos = p_pos - p_pos[:,0:1]
                
                # analyik
                analyik_theta = SMPL_AnalyIK_V3(t_pos, p_pos, self.parent, self.children)
                smpl_output = self.smpl(analyik_theta, beta)
                analyik_vert = smpl_output.vertices 
                analyik_p_pos = smpl_output.joints.contiguous()
                analyik_p_pos = analyik_p_pos - analyik_p_pos[:,0:1]   
                
                # theory
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
                
                smpl_out = self.smpl(theory_theta, beta)
                theory_vert = smpl_out.vertices
                theory_p_pos = smpl_out.joints
                theory_p_pos = theory_p_pos - theory_p_pos[:,0:1]
                
                # hybrik
                arm_pred_phi, spine_pred_phi, leg_pred_phi = self.generator(p_pos, beta)
                
                arm_pred_phi = arm_pred_phi / (torch.norm(arm_pred_phi, dim=2, keepdim=True) + 1e-8)
                spine_pred_phi = spine_pred_phi / (torch.norm(spine_pred_phi, dim=2, keepdim=True) + 1e-8)
                leg_pred_phi = leg_pred_phi / (torch.norm(leg_pred_phi, dim=2, keepdim=True) + 1e-8)
                
                arm_pred_theta, _ = self.arm_part_func(t_pos, p_pos, analyik_theta, arm_pred_phi)
                spine_pred_theta, _ = self.spine_part_func(t_pos, p_pos, analyik_theta, spine_pred_phi)
                leg_pred_theta, _ = self.leg_part_func(t_pos, p_pos, analyik_theta, leg_pred_phi)
                
                hybrik_theta = analyik_theta.clone()
                hybrik_theta = hybrik_theta.view(-1,24,3)
                hybrik_theta[:,[13,16,18,20,14,17,19,21]] = arm_pred_theta[:,[13,16,18,20,14,17,19,21]]
                hybrik_theta[:,[3,6,9,12]] = spine_pred_theta[:,[3,6,9,12]]
                hybrik_theta[:,[1,4,7,2,5,8]] = leg_pred_theta[:,[1,4,7,2,5,8]]
                
                smpl_out = self.smpl(hybrik_theta, beta)
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
            for param_group in self.gen_optimizer.param_groups:
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
            'gen_optimizer': self.gen_optimizer.state_dict(),
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
                
                
                