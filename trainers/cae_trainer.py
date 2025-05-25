import os
import time
import torch
import torchmetrics
from torch import optim
import torch.nn.functional as F
import numpy as np
from .base_trainer import BaseTrainer
import copy
import losses.pytorch_ssim as ssim
import losses.focal_loss as fc
import pandas as pd
from datetime import datetime

class Trainer(BaseTrainer):
    def __init__(self, model, cfg, device='cuda'):
        super(Trainer, self).__init__(model, cfg)
        self.cfg = cfg
        self.device = device
        self.model = model
              
    def start_train(self, train_dataset, val_dataset, save_model_dir, pretrained_file=None):        
        
        print(self.model)
        
        self._print_config()
        self._prepare_path(save_model_dir)   
        
        if not (pretrained_file is None): self._load_pretrained(pretrained_file, self.device)        
        
        print('\n----------------------------  start model training -----------------------------') 
        optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.LR)
       
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = self.cfg.DECAY_STEPS, gamma=self.cfg.DECAY_RATE)
       
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.AMP)
      
        self.model.to(device=self.device)

        # record properties during training
        df_summary = pd.DataFrame(columns = ['time', 'step', 'train cls loss', 'train cl loss', 'train accuracy'])
        df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"), index=False)
       
        max_acc = 0.0
        num_for_breaks = 0
        for epoch in range(1, self.cfg.EPOCHS+1):
            if num_for_breaks>self.cfg.EARLY_BREAK: 
                print('!!!!!! Early break training, accuracy not increased in contineous five epoches!')
                return
            print ('\n################################### epoch:'+str(epoch)+'/'+str(self.cfg.EPOCHS))
            self.model.train()
           
            t1 = time.time()
            use_focal_loss = True if epoch>=self.cfg.WARMUP else False
            contrastive_loss, cls_loss = self.__train_epoch(train_dataset, 
                                                                        optimizer, 
                                                                        grad_scaler, 
                                                                        scheduler,                                                                         
                                                                        use_focal_loss)
            t2 = time.time()
            
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print ('\nContrastive loss: %.3f; Cls loss: %0.3f; Lr: %.6f; Used time (s): %.4f' %
                    (contrastive_loss, cls_loss, current_lr, t2-t1)) 
            
            # perform online evaluation
            print('start evaluation:')
            self.model.eval()
            accuracys = []
           
            for step, (images, labels) in enumerate(val_dataset):
                images = images.permute(0, 3, 1, 2).to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    _, cls_pred, _ = self.model(images)
                    cls_pred = torch.argmax(F.softmax(cls_pred, dim=-1),dim=-1)
                  
                    acc = torchmetrics.functional.accuracy(cls_pred, labels, 'multiclass', num_classes=24)
                    
                    accuracys.append(acc)
            # print('evaluate done!')
            accuracys = torch.tensor(accuracys)
            mean_acc = np.mean(accuracys.detach().numpy())
            print('evaluate accuracy:', mean_acc)

            # save training properties
            current_time = "%s"%datetime.now()
            step = "Step[%d]"%epoch
           
            train_cls_loss = "%f"%cls_loss
            train_cl_loss = "%f"%contrastive_loss
            train_acc = "%g"%mean_acc
            # save training summary
            list = [current_time, step, train_cls_loss, train_cl_loss, train_acc]
            df_summary = pd.DataFrame([list])
            df_summary.to_csv(os.path.join(self.save_path, "training_summary.csv"),mode='a',header=False,index=False)

            if max_acc <= mean_acc:
                max_acc = mean_acc
                num_for_breaks = 0
                self.checkpoint_file = os.path.join(self.save_path, "best_weights.pth")  
            
                print ('Saving weights to %s' % (self.checkpoint_file))     
                self.model.eval()       
                torch.save(self.model.state_dict(), self.checkpoint_file)
            else:
                num_for_breaks+=1
            self._delete_old_weights(self.cfg.MAX_KEEPS_CHECKPOINTS) 
                
        print('\n---------------------------- model training completed ---------------------------')
            
    def __train_epoch(self, train_dataset, optimizer, grad_scaler, scheduler, use_focal_loss):
        losses = {'contrastive_loss':[], 'cls_loss':[]}
        
        for step, (images, labels) in enumerate(train_dataset):
            if step == self.cfg.STEPS_PER_EPOCH:break
            if self.cfg.AMP: images = images.type(torch.float16)

            feed_ims = images.permute(0, 3, 1, 2).to(self.device)
            labels = labels.to(self.device)
                             
            with torch.cuda.amp.autocast(enabled=self.cfg.AMP):                
                embedding, cls_pred, logit_scale = self.model(feed_ims)
             
            embedding1 = embedding[::2, ...]
            embedding2 = embedding[1::2, ...]

            contrastive_loss1to2 = self.__calc_contrastive_loss(embedding1, embedding2, logit_scale, use_focal_loss)
            contrastive_loss1to1 = self.__calc_contrastive_loss(embedding1, embedding1, logit_scale, use_focal_loss)
            contrastive_loss2to2 = self.__calc_contrastive_loss(embedding2, embedding2, logit_scale, use_focal_loss)
            contrastive_loss = contrastive_loss1to2 + contrastive_loss1to1 + contrastive_loss2to2

            cls_loss = self.__calc_cls_loss(cls_pred, labels, use_focal_loss)
           
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(contrastive_loss+2.0*cls_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()      
            
            losses['contrastive_loss'].append(contrastive_loss.detach().cpu().numpy())
            losses['cls_loss'].append(cls_loss.detach().cpu().numpy())
          
            self._draw_progress_bar(step+1, self.cfg.STEPS_PER_EPOCH)
        return np.mean(losses['contrastive_loss']), np.mean(losses['cls_loss'])
    
    def __calc_contrastive_loss(self, embedding1, embedding2, logit_scale, focal_loss=False):
        '''
        the contrastive loss for embeddings
        embedding1: (24, 128)
        embedding2: (24, 128)
        '''
        embedding1 = embedding1 / embedding1.norm(dim=1, keepdim=True)
        embedding2 = embedding2 / embedding2.norm(dim=1, keepdim=True)

        similar_matrix = logit_scale*(embedding1@embedding2.t())
        classes = similar_matrix.shape[-1]
        labels = torch.arange(classes).to(self.device)
        if focal_loss:
            l1 = fc.FocalLoss(gamma=2.0)(similar_matrix, labels)*10.0 #F.cross_entropy(similar_matrix, labels)
            l2 = fc.FocalLoss(gamma=2.0)(similar_matrix.t(), labels)*10.0 #F.cross_entropy(similar_matrix.t(), labels)
        else:
            l1 = F.cross_entropy(similar_matrix, labels)
            l2 = F.cross_entropy(similar_matrix.t(), labels)
        return 0.5*(l1+l2)
    
    def __calc_cls_loss(self, pred, labels, focal_loss=False):
        if focal_loss:
            return fc.FocalLoss(gamma=2.0)(pred, labels.long())*10.0
        else:
            return F.cross_entropy(pred, labels.long())*10.0
    
