# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:38:53 2024

@author: XCH
"""

## The current hyper-parameters values are not necessarily the best ones for a specific risk.
def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class KtoC():
    def __init__(self):
        super(KtoC, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'learning_rate': 0.005,
            'weight_decay': 0.0001,
            'step_size': 20.0,
            'lr_decay': 0.0005
            }
        self.alg_hparams = {
            "DANN": {
                "src_cls_loss_wt": 5.466455396319444,
                "domain_loss_wt": 6.984488815135327,
                },
            "MMDA": {
                "src_cls_loss_wt": 2.273807301486671,
                "coral_wt": 8.525181690680212,
                "mmd_wt": 0.7495582524661274,
                "cond_ent_wt": 8.394987387331275
                },
            "AdvSKM": {
                "src_cls_loss_wt": 5.152904627448014,
                "domain_loss_wt": 4.459557921862565
                },
            'MCDU': {
                'src_cls_loss_wt': 5.029537691598133,
                'domain_loss_wt': 5.103745553357777
                }
            }

class RtoH():
    def __init__(self):
        super(RtoH, self).__init__()
        self.train_params = {
            'num_epochs': 4,
            'batch_size': 32,
            'learning_rate': 0.01,
            'weight_decay': 1e-06,
            'step_size': 10.0,
            'lr_decay': 0.0005
            }
        self.alg_hparams = {
            "DANN": {
                "src_cls_loss_wt": 8.046205198685827,
                "domain_loss_wt": 0.02084406143192912,
                },
            "MMDA": {
                "src_cls_loss_wt": 3.758774591871598,
                "coral_wt": 3.847937150193473,
                "mmd_wt": 7.530003859721275,
                "cond_ent_wt": 9.390220727241504
                },
            "AdvSKM": {
                "src_cls_loss_wt": 9.14670342832931,
                "domain_loss_wt": 2.4169506827411182
                },
            'MCDU': {
                'src_cls_loss_wt': 5.029537691598133,
                'domain_loss_wt': 5.103745553357777
                }
            }

class MtoC():
    def __init__(self):
        super(MtoC, self).__init__()
        self.train_params = {
            'num_epochs': 4,
            'batch_size': 32,
            'learning_rate': 0.01,
            'weight_decay': 1e-06,
            'step_size': 10.0,
            'lr_decay': 0.0005
            }
        self.alg_hparams = {
            "DANN": {
                "src_cls_loss_wt": 8.046205198685827,
                "domain_loss_wt": 0.02084406143192912,
                },
            "MMDA": {
                "src_cls_loss_wt": 3.758774591871598,
                "coral_wt": 3.847937150193473,
                "mmd_wt": 7.530003859721275,
                "cond_ent_wt": 9.390220727241504
                },
            "AdvSKM": {
                "src_cls_loss_wt": 9.14670342832931,
                "domain_loss_wt": 2.4169506827411182
                },
            'MCDU': {
                'src_cls_loss_wt': 5.029537691598133,
                'domain_loss_wt': 5.103745553357777
                }
            }
