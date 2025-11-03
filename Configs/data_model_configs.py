# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:33:34 2024

@author: XCH
"""

def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


class KtoC():
    def __init__(self):
        super(KtoC, self)
        # the names of source and target
        self.scenarios = [("Kansas", "Czech")]
        self.pre_scenarios = [("Czech")]
        # class names
        self.class_names = ['Other', 'Maize', 'Wheat']
        # the length of time series 
        self.sequence_len = 37 
        self.shuffle = True
        self.drop_last = True
        self.normalize = False

        """ model configs """ 
        self.input_channels = 2
        self.kernel_size = 3
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 3

        """ CNN and RESNET features """ 
        self.mid_channels = 32
        self.final_out_channels = 128
        self.features_len = 1

        """ lstm features """
        self.lstm_hid = 64
        self.lstm_n_layers = 1
        self.lstm_bid = False

        """ discriminator """
        self.disc_hid_dim = 32
        self.hidden_dim = 64
        self.DSKN_disc_hid = 32
