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
            'learning_rate': 0.0005,
            'weight_decay': 0.0001,
            'step_size': 20.0,
            'lr_decay': 0.01
            }
        self.alg_hparams = {
            "AdvSKM": {
                "src_cls_loss_wt": 0.9380716781550786,
                "domain_loss_wt": 3.817488628714772
                }
            }
