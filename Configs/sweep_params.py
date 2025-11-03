# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:37:28 2024

@author: XCH
"""

sweep_hparams = {
        'DANN': {
            'num_epochs':       {'values': [40]},
            'batch_size':       {'values': [32]},
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'weight_decay':     {'values': [1e-4, 1e-5, 1e-6]},
            'step_size':        {'values': [5, 10, 20]},
            'lr_decay':         {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
            },
        'MMDA': {
            'num_epochs':       {'values': [40]},
            'batch_size':       {'values': [32]},
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'weight_decay':     {'values': [1e-4, 1e-5, 1e-6]},
            'step_size':        {'values': [5, 10, 20]}, 
            'lr_decay':         {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'coral_wt':         {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'mmd_wt':           {'distribution': 'uniform', 'min': 1e-2, 'max': 10},
            'cond_ent_wt':      {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
            },
        'AdvSKM': {
            'num_epochs':       {'values': [40]},
            'batch_size':       {'values': [32]},
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'weight_decay':     {'values': [1e-4, 1e-5, 1e-6]},
            'step_size':        {'values': [5, 10, 20]}, 
            'lr_decay':         {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
            },
        'MCDU': {
            'num_epochs':       {'values': [40]},
            'batch_size':       {'values': [32]},
            'learning_rate':    {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'weight_decay':     {'values': [1e-4, 1e-5, 1e-6]},
            'step_size':        {'values': [5, 10, 20]}, 
            'lr_decay':         {'values': [1e-2, 5e-3, 1e-3, 5e-4]},
            'src_cls_loss_wt':  {'distribution': 'uniform', 'min': 1e-1, 'max': 10},
            'domain_loss_wt':   {'distribution': 'uniform', 'min': 1e-2, 'max': 10}
            }
}