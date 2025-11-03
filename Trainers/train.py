# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 22:32:21 2024

@author: XCH
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
import collections
import argparse
from utils import AverageMeter, fix_randomness, starting_logs
from Trainers.abstract_trainer import AbstractTrainer
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()

import time

import seaborn as sns
import matplotlib.pyplot as plt

row = 1559
column = 1559

class Trainer(AbstractTrainer):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        super().__init__(args)
        self.results_columns = ["Scenario", "Random Seed", "Accuracy", "Precision", "Recall", "F1Score", "Kappa"]
        self.risks_columns = ["Scenario", "Random Seed", "Source Risk", "DEV Risk", "Target Risk"]

    def fit(self):
        # table with metrics
        table_results = pd.DataFrame(columns=self.results_columns)
        # table with risks
        table_risks = pd.DataFrame(columns=self.risks_columns)

        # Trainer
        for src_id, trg_id in self.dataset_configs.scenarios:
            # fixing random seed
            fix_randomness(self.num_runs)

            # Logging
            self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                            src_id, trg_id, self.num_runs)
            # Average meters
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

            # Load data
            self.load_data(src_id, trg_id)
            
            # initiate the domain adaptation algorithm
            self.initialize_algorithm()

            # Train the domain adaptation algorithm
            self.last_model, self.best_model = self.algorithm.update(self.src_train_dl, self.src_test_dl, self.trg_train_dl, self.loss_avg_meters, self.logger)

            # Save checkpoint
            self.save_checkpoint(self.home_path, self.scenario_log_dir, self.last_model, self.best_model)

            # Calculate risks and metrics
            metrics = self.calculate_metrics()
            risks = self.calculate_risks()
            print("Accuracy, Precision, Recall, F1Score, Kappa: ", metrics)
            print("Source Risk, DEV Risk, Target Risk: ", risks)
            
            CM = self.confusionmatrix()
            label = ["Other", "Maize", "Winter Wheat"]
            sns.heatmap(CM,fmt='g',cmap='Reds',annot=True,cbar=False,xticklabels=label,yticklabels=label,annot_kws={"fontsize":25}) 
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()

            # Append results to tables
            scenario = f"{src_id}_to_{trg_id}"
            table_results = self.append_results_to_tables(table_results, scenario, self.num_runs, metrics)
            table_risks = self.append_results_to_tables(table_risks, scenario, self.num_runs, risks)
            print(table_results)
            print(table_risks)

        # Save tables to file if needed
        self.save_tables_to_file(table_results, 'results')
        self.save_tables_to_file(table_risks, 'risks')

    def test(self):
        # Results dataframes
        last_results = pd.DataFrame(columns=self.results_columns)
        best_results = pd.DataFrame(columns=self.results_columns)

        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            # fixing random seed
            fix_randomness(self.num_runs)
            # Logging
            self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(self.num_runs))
            print(self.scenario_log_dir)
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            # Load data
            self.load_data(src_id, trg_id)
            # Build model
            self.initialize_algorithm()
            # Load chechpoint 
            last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)
            # Testing the last model
            self.algorithm.network.load_state_dict(last_chk)
            self.evaluate(self.trg_test_dl)
            last_metrics = self.calculate_metrics()
            last_results = self.append_results_to_tables(last_results, f"{src_id}_to_{trg_id}", self.num_runs,
                                                         last_metrics)
            print(last_results)
            print("---")
            # Testing the best model
            self.algorithm.network.load_state_dict(best_chk)
            self.evaluate(self.trg_test_dl)
            best_metrics = self.calculate_metrics()
            # Append results to tables
            best_results = self.append_results_to_tables(best_results, f"{src_id}_to_{trg_id}", self.num_runs,
                                                         best_metrics)
            print(best_results)
            print("---")

        # Save tables to file if needed
        self.save_tables_to_file(last_results, 'last_results')
        self.save_tables_to_file(best_results, 'best_results')

        # printing summary 
        summary_last = {metric: np.mean(last_results[metric]) for metric in self.results_columns[2:]}
        summary_best = {metric: np.mean(best_results[metric]) for metric in self.results_columns[2:]}
        for summary_name, summary in [('Last', summary_last), ('Best', summary_best)]:
            for key, val in summary.items():
                print(f'{summary_name}: {key}\t: {val:2.4f}')

    def predict(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cross-domain scenarios
        for src_id, trg_id in self.dataset_configs.scenarios:
            # fixing random seed
            fix_randomness(self.num_runs)
            # Logging
            self.scenario_log_dir = os.path.join(self.exp_log_dir, src_id + "_to_" + trg_id + "_run_" + str(self.num_runs))
            print(self.scenario_log_dir)
            self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            # Load predict data
            for pre_id in self.dataset_configs.pre_scenarios:
                self.load_predata(pre_id)
                # Build model
                self.initialize_algorithm()
                # Load chechpoint 
                last_chk, best_chk = self.load_checkpoint(self.scenario_log_dir)
                # Testing the last model
                self.algorithm.network.load_state_dict(last_chk)
                
                start_time = time.time()
                self.evaluatepre(self.pre_dl)
                end_time = time.time()
                
                reasoning_time_last = end_time - start_time
                print(f"Reasoning time of last model: {reasoning_time_last:} seconds")
                
                predicted_lastclasses = torch.argmax(self.full_preds, dim=1)
                one_hot_last = torch.eye(self.full_preds.shape[1], device=device)[predicted_lastclasses]
                # print(self.one_hot_predictions)
                predicted_last = torch.argmax(one_hot_last, dim=1)
                # print(predicted_classes)
                
                # image_lastarray = np.array(predicted_last)
                image_lastarray = predicted_last.cpu().numpy()
                imagelast = image_lastarray.reshape((row, column))
                cv2.imwrite(r".\TestL.png", imagelast)
                
                # Testing the best model
                self.algorithm.network.load_state_dict(best_chk)
                
                start_time = time.time()
                self.evaluatepre(self.pre_dl)
                end_time = time.time()
                
                reasoning_time_best = end_time - start_time
                print(f"Reasoning time of best model: {reasoning_time_best:} seconds")
                
                predicted_bestclasses = torch.argmax(self.full_preds, dim=1)
                one_hot_best = torch.eye(self.full_preds.shape[1], device=device)[predicted_bestclasses]
                # print(self.one_hot_predictions)
                predicted_best = torch.argmax(one_hot_best, dim=1)
                # print(predicted_classes)
                
                # image_bestarray = np.array(predicted_best)
                image_bestarray = predicted_best.cpu().numpy()
                imagebest = image_bestarray.reshape((row, column))
                cv2.imwrite(r".\TestB.png", imagebest)
                
                txt_path = r".\reasoning_time.txt"
                with open(txt_path, "w") as f:
                    f.write(f"Reasoning time of last model: {reasoning_time_last:} seconds\n")
                    f.write(f"Reasoning time of best model: {reasoning_time_best:} seconds\n")
        
        print("Predtict Over!")
