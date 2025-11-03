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
            'weight_decay': 1e-05,
            'step_size': 10.0,
            'lr_decay': 0.0005
            }
        self.alg_hparams = {
            'MCDU': {
                'src_cls_loss_wt': 9.548912787464536,
                'domain_loss_wt': 5.983865175795871
                }
            }
        