# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:31:21 2024

@author: XCH
"""
import time
start = time.time()

from Trainers.train import Trainer
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # ========  Experiments Phase ================
    parser.add_argument('--phase', default='train', type=str, help='train, test, predict')
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='Experiments_MCDU', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='MCDU-KTC', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='MCDU', type=str, help='DANN, MMDA, AdvSKM, MCDU')
    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'.\UDA-Data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='KtoC', type=str, help='Dataset of choice: KtoC')
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: CNN')
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=42, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default= "cuda", type=str, help='cpu or cuda')
    # arguments
    args = parser.parse_args()
    # create trainier object
    trainer = Trainer(args)
    # train and test
    if args.phase == 'train':
        trainer.fit()
    elif args.phase == 'test':
        trainer.test()
    elif args.phase == 'predict':
        trainer.predict()

end = time.time()
runTime = end - start
print("Run Time:", runTime)