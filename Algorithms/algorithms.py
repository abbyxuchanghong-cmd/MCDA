# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:33:07 2024

@author: XCH
"""

import torch
import torch.nn as nn
import itertools
import numpy as np 

from Models.models import classifier, Discriminator, ReverseLayerF, AdvSKM_Disc
from Models.loss import KDLoss, MMD_loss, CORAL, ConditionalEntropyLoss
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy

from torchmetrics import Accuracy
CCL_classes = 3

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs, backbone):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

    # update function is common to all algorithms
    def update(self, src_loader, src_test_loader, trg_loader, avg_meter, logger):
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # training loop 
            self.training_epoch(src_loader, src_test_loader, trg_loader, avg_meter, epoch)
            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src Cls Loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src Cls Loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model

class DANN(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)
        
        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Domain Discriminator
        self.domain_classifier = Discriminator(configs)
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99))
        self.lr_scheduler_disc = StepLR(self.optimizer_disc, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, src_test_loader, trg_loader, avg_meter, epoch):
        
        joint_loader = enumerate(zip(src_loader, itertools.cycle(src_test_loader), itertools.cycle(trg_loader)))
        num_batches = max(len(src_loader), len(trg_loader))

        for step, ((src_x, src_y), (src_test_x, src_test_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            src_test_x, src_test_y = src_test_x.to(self.device), src_test_y.to(self.device)
            
            p = float(step + epoch * num_batches) / self.hparams["num_epochs"] + 1 / num_batches
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            self.optimizer.zero_grad()
            self.optimizer_disc.zero_grad()

            domain_label_src = torch.ones(len(src_x)).to(self.device)
            domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            
            src_test_feat = self.feature_extractor(src_test_x)
            src_test_pred = self.classifier(src_test_feat)

            trg_feat = self.feature_extractor(trg_x)

            # Task classification  Loss
            src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

            # Domain classification loss
            # source
            src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
            src_domain_pred = self.domain_classifier(src_feat_reversed)
            src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

            # target
            trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
            trg_domain_pred = self.domain_classifier(trg_feat_reversed)
            trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

            # Total domain loss
            domain_loss = src_domain_loss + trg_domain_loss

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * domain_loss

            loss.backward()
            self.optimizer.step()
            self.optimizer_disc.step()

            ACC = Accuracy(task="multiclass", num_classes=CCL_classes)
            train_acc = ACC(src_pred.squeeze().cpu(), src_y.cpu())
            test_acc = ACC(src_test_pred.squeeze().cpu(), src_test_y.cpu())

            losses =  {'Src Cls Loss': src_cls_loss.item(), 
                       'Src Train Acc': train_acc.item(),
                       'Src Test Acc': test_acc.item(),
                       'Domain Loss': domain_loss.item(),
                       'Total Loss': loss.item()}
           
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
        self.lr_scheduler_disc.step()


class MMDA(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Aligment losses
        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()
        
        # hparams
        self.hparams = hparams
        # device
        self.device = device
    
    def training_epoch(self, src_loader, src_test_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(src_test_loader), itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (src_test_x, src_test_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device)
            src_test_x, src_test_y = src_test_x.to(self.device), src_test_y.to(self.device)

            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            
            src_test_feat = self.feature_extractor(src_test_x)
            src_test_pred = self.classifier(src_test_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y)

            trg_feat = self.feature_extractor(trg_x)

            coral_loss = self.coral(src_feat, trg_feat)
            mmd_loss = self.mmd(src_feat, trg_feat)
            cond_ent_loss = self.cond_ent(trg_feat)

            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                   self.hparams["coral_wt"] * coral_loss + \
                   self.hparams["mmd_wt"] * mmd_loss + \
                   self.hparams["cond_ent_wt"] * cond_ent_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ACC = Accuracy(task="multiclass", num_classes=CCL_classes)
            train_acc = ACC(src_pred.squeeze().cpu(), src_y.cpu())
            test_acc = ACC(src_test_pred.squeeze().cpu(), src_test_y.cpu())

            losses =  {'Src Cls Loss': src_cls_loss.item(),
                       'Src Train Acc': train_acc.item(),
                       'Src Test Acc': test_acc.item(),
                       'Coral Loss': coral_loss.item(),
                       'MMD Loss': mmd_loss.item(),
                       'Cond Ent Loss': cond_ent_loss.item(),
                       'Total Loss': loss.item()}
            
            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler.step()


class AdvSKM(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        # optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # Aligment losses
        self.mmd_loss = MMD_loss()
        self.AdvSKM_embedder = AdvSKM_Disc(configs).to(device)
        self.optimizer_disc = torch.optim.Adam(
            self.AdvSKM_embedder.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        self.lr_scheduler_disc = StepLR(self.optimizer_disc, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def training_epoch(self, src_loader, src_test_loader, trg_loader, avg_meter, epoch):
        joint_loader = enumerate(zip(src_loader, itertools.cycle(src_test_loader), itertools.cycle(trg_loader)))

        for step, ((src_x, src_y), (src_test_x, src_test_y), (trg_x, trg_y)) in joint_loader:
            src_x, src_y, trg_x, trg_y = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device), trg_y.to(self.device)
            src_test_x, src_test_y = src_test_x.to(self.device), src_test_y.to(self.device)
            
            src_feat = self.feature_extractor(src_x)
            src_pred = self.classifier(src_feat)
            
            src_test_feat = self.feature_extractor(src_test_x)
            src_test_pred = self.classifier(src_test_feat)

            # extract target features
            trg_feat = self.feature_extractor(trg_x)

            source_embedding_disc = self.AdvSKM_embedder(src_feat.detach())
            target_embedding_disc = self.AdvSKM_embedder(trg_feat.detach())
            mmd_loss = - self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss.requires_grad = True

            # update discriminator
            self.optimizer_disc.zero_grad()
            mmd_loss.backward()
            self.optimizer_disc.step()

            # calculate source classification loss
            src_cls_loss = self.cross_entropy(src_pred, src_y)

            # domain loss.
            source_embedding_disc = self.AdvSKM_embedder(src_feat)
            target_embedding_disc = self.AdvSKM_embedder(trg_feat)

            mmd_loss_adv = self.mmd_loss(source_embedding_disc, target_embedding_disc)
            mmd_loss_adv.requires_grad = True

            # calculate the total loss
            loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
                self.hparams["domain_loss_wt"] * mmd_loss_adv

            # update optimizer
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            ACC = Accuracy(task="multiclass", num_classes=CCL_classes)
            train_acc = ACC(src_pred.squeeze().cpu(), src_y.cpu())
            test_acc = ACC(src_test_pred.squeeze().cpu(), src_test_y.cpu())
            
            losses =  {'Src Cls Loss': src_cls_loss.item(),
                       'Src Train Acc': train_acc.item(),
                       'Src Test Acc': test_acc.item(),
                       'MMD Loss': mmd_loss_adv.item(),
                       'Total Loss': loss.item()}
            
            for key, val in losses.items():
                    avg_meter[key].update(val, 32)

        self.lr_scheduler.step()
        self.lr_scheduler_disc.step()


class MCDU(Algorithm):
    
    def __init__(self, backbone, configs, hparams, device):
        super().__init__(configs, backbone)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.classifier1 = classifier(configs)
        self.classifier2 = classifier(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer and scheduler
        self.optimizer_fe = torch.optim.Adam(
            self.feature_extractor.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        # optimizer and scheduler
        self.optimizer_c = torch.optim.Adam(
            self.classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        # optimizer and scheduler
        self.optimizer_c1 = torch.optim.Adam(
            self.classifier1.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])
        # optimizer and scheduler
        self.optimizer_c2 = torch.optim.Adam(
            self.classifier2.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"])

        self.lr_scheduler_fe = StepLR(self.optimizer_fe, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c = StepLR(self.optimizer_c, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c1 = StepLR(self.optimizer_c1, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.lr_scheduler_c2 = StepLR(self.optimizer_c2, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        
        self.KD_loss = KDLoss()

        # hparams
        self.hparams = hparams
        # device
        self.device = device

    def update(self, src_train_loader, src_test_loader, trg_train_loader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = None

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # source pretraining loop 
            self.pretrain_epoch(src_train_loader, src_test_loader, avg_meter)
            # training loop 
            self.training_epoch(src_train_loader, trg_train_loader, avg_meter, epoch)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src Cls Loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src Cls Loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug('-------------------------------------')
        
        last_model = self.network.state_dict()

        return last_model, best_model

    def pretrain_epoch(self, src_train_loader, src_test_loader, avg_meter):
        joint_loader = enumerate(zip(src_train_loader, itertools.cycle(src_test_loader)))

        for step, ((src_x, src_y), (src_test_x, src_test_y)) in joint_loader:
            src_x, src_y = src_x.to(self.device), src_y.to(self.device)
            src_test_x, src_test_y = src_test_x.to(self.device), src_test_y.to(self.device)
          
            src_feat = self.feature_extractor(src_x) # feature extractor
            src_pred = self.classifier(src_feat) # source classifier
            
            src_test_feat = self.feature_extractor(src_test_x)
            src_test_pred = self.classifier(src_test_feat)

            src_cls_loss = self.cross_entropy(src_pred, src_y) # source loss
            
            self.optimizer_fe.zero_grad()
            self.optimizer_c.zero_grad()
            
            src_cls_loss = self.hparams["src_cls_loss_wt"] * src_cls_loss
            src_cls_loss.backward()

            self.optimizer_fe.step()
            self.optimizer_c.step()
            
            ACC = Accuracy(task="multiclass", num_classes=CCL_classes)
            train_acc = ACC(src_pred.squeeze().cpu(), src_y.cpu())
            test_acc = ACC(src_test_pred.squeeze().cpu(), src_test_y.cpu())
            
            losses = {'Src Cls Loss': src_cls_loss.item(),
                      'Src Train Acc': train_acc.item(),
                      'Src Test Acc': test_acc.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)
        
        self.lr_scheduler_fe.step()
        self.lr_scheduler_c.step()

    def training_epoch(self, src_train_loader, trg_train_loader, avg_meter, epoch):

        # Construct Joint Loaders 
        joint_loader =enumerate(zip(src_train_loader, itertools.cycle(trg_train_loader)))

        for step, ((src_x, src_y), (trg_x, _)) in joint_loader:
            src_x, src_y, trg_x = src_x.to(self.device), src_y.to(self.device), trg_x.to(self.device) 
            
            # Step B: Freeze the feature extractor
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = False
            # Update C1 and C2 to maximize their difference on target sample
            for k, v in self.classifier1.named_parameters():
                v.requires_grad = True
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = True
            
            # source
            src_feat = self.feature_extractor(src_x)
            src_pred1 = self.classifier1(src_feat)
            src_pred2 = self.classifier2(src_feat)
            # source losses
            src_cls_loss1 = self.cross_entropy(src_pred1, src_y)
            src_cls_loss2 = self.cross_entropy(src_pred2, src_y)
            loss_s =src_cls_loss1 + src_cls_loss2

            # target
            trg_feat = self.feature_extractor(trg_x) 
            trg_pred1 = self.classifier1(trg_feat.detach())
            trg_pred2 = self.classifier2(trg_feat.detach())
            # target loss
            loss_dis = self.KD_loss(trg_pred1, trg_pred2)
            
            loss = self.hparams["src_cls_loss_wt"] * loss_s - loss_dis
            loss.backward()
            self.optimizer_c1.step()
            self.optimizer_c2.step()

            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()
            self.optimizer_fe.zero_grad()

            # Step C: Freeze the classifiers
            for k, v in self.classifier1.named_parameters():
                v.requires_grad = False
            for k, v in self.classifier2.named_parameters():
                v.requires_grad = False
            # Update the feature extractor to minimize the discrepaqncy on target samples
            for k, v in self.feature_extractor.named_parameters():
                v.requires_grad = True
            
            # target
            trg_feat = self.feature_extractor(trg_x)        
            trg_pred1 = self.classifier1(trg_feat)
            trg_pred2 = self.classifier2(trg_feat)
            # target loss
            loss_dis_t = self.KD_loss(trg_pred1, trg_pred2)
            domain_loss = self.hparams["domain_loss_wt"] * loss_dis_t 

            domain_loss.backward()
            self.optimizer_fe.step()

            self.optimizer_fe.zero_grad()
            self.optimizer_c1.zero_grad()
            self.optimizer_c2.zero_grad()

            losses =  {'Total Loss': loss.item(),
                       'Domain Loss': domain_loss.item()}

            for key, val in losses.items():
                avg_meter[key].update(val, 32)

        self.lr_scheduler_fe.step()
        self.lr_scheduler_c1.step()
        self.lr_scheduler_c2.step()
