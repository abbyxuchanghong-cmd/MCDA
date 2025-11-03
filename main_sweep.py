# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:59:26 2024

@author: XCH
"""
import time
start = time.time()

from Trainers.sweep import Trainer
import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='MCDU', type=str, help='DANN, MMDA, AdvSKM, MCDU')

    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'F:\02-Paper\First Review Comments\Data\01\UDA-Data', type=str, help='Path containing datase2t')
    parser.add_argument('--dataset', default='KtoC', type=str, help='Dataset of choice: KtoC')

    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='CNN', type=str, help='Backbone of choice: CNN')

    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=42, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument('--exp_name', default='Sweep_MCDU_KtoC', type=str, help='experiment name')

    # ======== sweep settings =====================
    parser.add_argument('--num_sweeps', default=50, type=str, help='Number of sweep runs')

    # We run sweeps using wandb plateform, so next parameters are for wandb.
    parser.add_argument('--sweep_project_wandb', default='MCDU_KtoC', type=str, help='Project name in Wandb')
    parser.add_argument('--wandb_entity', type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
    parser.add_argument('--search_strategy', default="random", type=str, help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
    parser.add_argument('--metric_to_minimize', default="dev_risk", type=str, help='select one of: (src_risk - trg_risk - dev_risk)')

    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='Sweep_MCDU_KtoC', type=str, help='Directory containing all experiments')

    args = parser.parse_args()

    trainer = Trainer(args)

    trainer.sweep()

end = time.time()
runTime = end - start
print("Run Time:", runTime)
