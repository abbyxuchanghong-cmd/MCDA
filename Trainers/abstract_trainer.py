# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:47:02 2024

@author: XCH
"""

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score, CohenKappa, ConfusionMatrix
import os
import wandb
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
from Dataloader.dataloader import data_generator
from Dataloader.predataloader import predata_generator
from Configs.data_model_configs import get_dataset_class
from Configs.hparams import get_hparams_class
from Algorithms.algorithms import get_algorithm_class
from Models.models import get_backbone_class

from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch import nn as nn

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# logistic regression model
class simple_MLP(nn.Module):
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()
        self.dense0 = nn.Linear(inp_units, inp_units // 2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units // 2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x

def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu())  # source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu())  # target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()

    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []
    
    # compile the logistic regression model
    for lr in learning_rates:
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units=configs.final_out_channels * configs.features_len,
            max_epochs=40,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable")
        
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]
    
    # source test results
    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    # compute the importance weights
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t

def get_dev_value(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    # weighted cross-entropy loss
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    # compute the DEV risk
    return np.mean(weighted_error) + eta * np.mean(weight) - eta

def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_test_dl, configs, device):
    src_train_feats = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_test_feats = target_model.feature_extractor(src_test_dl.dataset.x_data.float().to(device))
    src_test_pred = target_model.classifier(src_test_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_test_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_test_pred, src_test_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    
    return dev_risk

class AbstractTrainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        # Exp Description
        self.experiment_description = args.dataset 
        self.run_description = f"{args.da_method}_{args.exp_name}"
        # Paths, os.getcwd get Current Working Directory
        self.home_path =  os.getcwd() # os.path.dirname(os.getcwd())
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        # self.create_save_dir(os.path.join(self.home_path,  self.save_dir ))
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")
        os.makedirs(self.exp_log_dir, exist_ok=True)

        # Specify runs
        self.num_runs = args.num_runs
        # get dataset and base model configs
        self.dataset_configs, self.hparams_class = self.get_configs()
        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels
        # Specify number of hparams
        self.hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

        # metrics
        self.num_classes = self.dataset_configs.num_classes
        self.Accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.Precision = Precision(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.Recall = Recall(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.F1Score = F1Score(task="multiclass", num_classes=self.num_classes, average="weighted")
        self.Kappa = CohenKappa(task="multiclass", num_classes=self.num_classes)
        self.ConfusionMatrix = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
    
    def initialize_algorithm(self):
        # get algorithm class
        algorithm_class = get_algorithm_class(self.da_method)
        backbone_fe = get_backbone_class(self.backbone)
        # Initilaize the algorithm
        self.algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)
        self.algorithm.to(self.device)

    def load_checkpoint(self, model_dir):
        checkpoint = torch.load(os.path.join(self.home_path, model_dir, 'checkpoint.pt'))
        last_model = checkpoint['last']
        best_model = checkpoint['best']
        return last_model, best_model

    def evaluate(self, test_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        total_loss, preds_list, labels_list = [], [], []

        with torch.no_grad():
            for data, labels in test_loader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                # forward pass
                features = feature_extractor(data)
                predictions = classifier(features)
                # compute loss
                loss = F.cross_entropy(predictions, labels)
                total_loss.append(loss.item())
                pred = predictions.detach()  # .argmax(dim=1)  # get the index of the max log-probability
                # append predictions and labels
                preds_list.append(pred)
                labels_list.append(labels)

        self.loss = torch.tensor(total_loss).mean()  # average loss
        self.full_preds = torch.cat((preds_list))
        self.full_labels = torch.cat((labels_list))
        
    def evaluatepre(self, pre_loader):
        feature_extractor = self.algorithm.feature_extractor.to(self.device)
        classifier = self.algorithm.classifier.to(self.device)

        feature_extractor.eval()
        classifier.eval()

        preds_list = []

        with torch.no_grad():
            for data in pre_loader:
                data = data.float().to(self.device)
                features = feature_extractor(data)
                predictions = classifier(features)
                pred = predictions.detach()
                preds_list.append(pred)

        self.full_preds = torch.cat((preds_list))

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "train")
        self.src_test_dl = data_generator(self.data_path, src_id, self.dataset_configs, self.hparams, "test")
        self.trg_train_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "train")
        self.trg_test_dl = data_generator(self.data_path, trg_id, self.dataset_configs, self.hparams, "test")
    
    def load_predata(self, pre_id):
        self.pre_dl = predata_generator(self.data_path, pre_id, self.dataset_configs, self.hparams, "predict")

    def create_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    def save_tables_to_file(self,table_results, name):
        # save to file if needed
        table_results.to_csv(os.path.join(self.exp_log_dir,f"{name}.csv"))

    def save_checkpoint(self, home_path, log_dir, last_model, best_model):
        save_dict = {
            "last": last_model,
            "best": best_model
        }
        # save classification report
        save_path = os.path.join(home_path, log_dir, "checkpoint.pt")
        torch.save(save_dict, save_path)
    
    def calculate_avg_std_wandb_table(self, results):
        # avg_metrics = [np.mean(results.get_column(metric)) for metric in results.columns[2:]]
        # std_metrics = [np.std(results.get_column(metric)) for metric in results.columns[2:]]
        summary_metrics = {metric: np.mean(results.get_column(metric)) for metric in results.columns[2:]}
        # results.add_data('mean', '-', *avg_metrics)
        # results.add_data('std', '-', *std_metrics)
        return results, summary_metrics

    def wandb_logging(self, total_results, total_risks, summary_metrics, summary_risks):
        # log wandb
        wandb.log({'results': total_results})
        wandb.log({'risks': total_risks})
        wandb.log({'hparams': wandb.Table(dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']), allow_mixed_types=True)})
        wandb.log(summary_metrics)
        wandb.log(summary_risks)
        print(pd.DataFrame(dict(self.hparams).items())) 
 
    def calculate_metrics(self):
        self.evaluate(self.trg_test_dl)
        # accuracy  
        accuracy = self.Accuracy(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # precision 
        precision = self.Precision(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # recall
        recall = self.Recall(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # f1
        f1score = self.F1Score(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        # kappa
        kappa = self.Kappa(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu()).item()
        
        return accuracy, precision, recall, f1score, kappa
    
    def confusionmatrix(self):
        # ConfusionMatrix
        confusionmatrix = self.ConfusionMatrix(self.full_preds.argmax(dim=1).cpu(), self.full_labels.cpu())
        return confusionmatrix

    def calculate_risks(self):
        # calculation based source test data
        self.evaluate(self.src_test_dl)
        src_risk = self.loss.item()
        # calculation based target test data
        self.evaluate(self.trg_test_dl)
        trg_risk = self.loss.item()
        dev_risk = calc_dev_risk(self.algorithm, self.src_train_dl, self.trg_train_dl, self.src_test_dl, self.dataset_configs, self.device)
        return src_risk, dev_risk, trg_risk
    
    def append_results_to_tables(self, table, scenario, run_id, metrics):
        # Create metrics and risks rows
        results_row = [scenario, run_id, *metrics]
        # Create new dataframes for each row
        results_df = pd.DataFrame([results_row], columns=table.columns)
        # Concatenate new dataframes with original dataframes
        table = pd.concat([table, results_df], ignore_index=True)
        return table