# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:48:52 2024

@author: XCH
"""

import os
import wandb
import warnings
import sklearn.exceptions
import collections
import argparse
from Configs.sweep_params import sweep_hparams
from utils import AverageMeter, fix_randomness, starting_logs
from Trainers.abstract_trainer import AbstractTrainer

import time

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """

    def __init__(self, args):
        super(Trainer, self).__init__(args)

        # sweep parameters
        self.num_sweeps = args.num_sweeps
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.search_strategy = args.search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # Logging
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir)
        os.makedirs(self.exp_log_dir, exist_ok=True)

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'name': self.da_method + '_' + self.backbone,
            'method': self.search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'parameters': {**sweep_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)
        
        sweep_start = time.time()

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)
        
        sweep_end = time.time()
        total_sweep_time = sweep_end - sweep_start
        print(f"Total sweep time: {total_sweep_time:.2f} s")
        
        save_path = os.path.join(self.exp_log_dir, f"Sweep_Summary_{self.da_method}_{self.backbone}.txt")
        with open(save_path, "w") as f:
            f.write(f"total_sweep_time_sec: {total_sweep_time:.2f}\n")
        print(f"Sweep summary saved to: {save_path}")

    def train(self):
        
        start_time = time.time()
        
        run = wandb.init(config=self.hparams)
        self.hparams= wandb.config
        
        # create tables for results and risks
        columns = ["Scenario", "Random Seed", "Accuracy", "Precision", "Recall", "F1Score", "Kappa"]
        table_results = wandb.Table(columns=columns, allow_mixed_types=True)
        columns = ["Scenario", "Random Seed", "Source Risk", "DEV Risk", "Target Risk"]
        table_risks = wandb.Table(columns=columns, allow_mixed_types=True)

        for src_id, trg_id in self.dataset_configs.scenarios:
            # set random seed and create logger
            fix_randomness(self.num_runs)
            self.logger, self.scenario_log_dir = starting_logs( self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, self.num_runs)

            # average meters
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # load data and train model
            self.load_data(src_id, trg_id)

            # initiate the domain adaptation algorithm
            self.initialize_algorithm()

            # Train the domain adaptation algorithm
            self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.src_test_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

            # calculate metrics and risks
            metrics = self.calculate_metrics()
            risks = self.calculate_risks()
            print("Accuracy, Precision, Recall, F1Score, Kappa: ", metrics)
            print("Source Risk, DEV Risk, Target Risk: ", risks)

            # append results to tables
            scenario = f"{src_id}_to_{trg_id}"
            table_results.add_data(scenario, self.num_runs, *metrics)
            table_risks.add_data(scenario, self.num_runs, *risks)

        # calculate overall metrics and risks
        total_results, summary_metrics = self.calculate_avg_std_wandb_table(table_results)
        total_risks, summary_risks = self.calculate_avg_std_wandb_table(table_risks)

        # log results to WandB
        self.wandb_logging(total_results, total_risks, summary_metrics, summary_risks)
        
        end_time = time.time()
        run_time = end_time - start_time
        wandb.log({"run_time_sec": run_time})
        print(f"Run finished. Duration: {run_time:.2f} s")

        # finish the run
        run.finish()