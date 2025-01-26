"""
Author: Jay Shah
Email: jgshah1@asu.edu
Affiliation: Arizona State University
Date: 01/16/2025
Description: A script for testing a deep learning model on a pre-specified dataset. 
It evaluates metrics such as loss, accuracy, and balanced accuracy, 
and generates a confusion matrix for visualization.
"""

import argparse
import os
import torch
import numpy as np
import logging
import time
import librosa
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from misc import load_yaml
from custom_model import CustomModel
from models import resnet
from sklearn.metrics import roc_auc_score, confusion_matrix

CONFIG_FILE = "./configs/inference_config.yaml"

def extract_mfcc(row):
    soundArr, sample_rate = librosa.load(row['audio_path'])
    return librosa.feature.mfcc(y=soundArr, sr=sample_rate)

def setup_logging(logs_dir):
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, "testing.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(data_configs):
    if not os.path.exists(data_configs["name"]):
        raise FileNotFoundError("Dataset not found.")
    if not os.path.exists(data_configs["label_file"]):
        raise FileNotFoundError("Label file not found.")

    data_wlabels = pd.read_csv(data_configs["label_file"])
    test_x = np.array(data_wlabels.apply(extract_mfcc, axis=1).tolist())
    test_y = np.array(data_wlabels.disease).reshape(-1, 1)
    return test_x, test_y

def initialize_model(model_name, model_configs, device):
    model_dict = {
        "custom": CustomModel,
        "resnet18": lambda: resnet.resnet18_2d(num_classes=model_configs["num_classes"], in_channels=1),
        "resnet34": lambda: resnet.resnet34_2d(num_classes=model_configs["num_classes"], in_channels=1),
        "resnet50": lambda: resnet.resnet50_2d(num_classes=model_configs["num_classes"], in_channels=1)
    }
    return model_dict.get(model_name, lambda: Exception("Model not found in config file."))().to(device)

def run_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss, all_preds, all_labels = 0, [], []
    start_time = time.time()

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            epoch_loss += loss.item()

            all_preds.extend(torch.round(y_pred).detach().cpu().numpy().flatten())
            all_labels.extend(y_batch.detach().cpu().numpy().flatten())

    epoch_loss /= len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    time_per_sample = (time.time() - start_time) / len(loader.dataset)

    return epoch_loss, accuracy, balanced_accuracy, precision, recall, f1, all_preds, all_labels, time_per_sample

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", nargs="?", const=CONFIG_FILE, default=CONFIG_FILE)
    args = parser.parse_args()
    configs = load_yaml(args.config_file)
    model_configs = configs["model"]
    data_configs = configs["dataset"]

    device = torch.device(f"cuda:{model_configs['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    model = initialize_model(model_configs["name"], model_configs, device)

    test_x, test_y = load_dataset(data_configs)
    test_x = torch.tensor(test_x).unsqueeze(1) if len(test_x.shape) == 3 else torch.tensor(test_x)
    test_y = torch.tensor(test_y)
    test_loader = DataLoader(TensorDataset(test_x, test_y))

    model_path = os.path.join("_".join(["mfcc", model_configs["name"], configs["save_dir"]]), "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = torch.nn.BCELoss()

    save_dir = "_".join([configs["dataset"]["name"], model_configs["name"], "results"])
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(os.path.join(save_dir, configs["logs_dir"]))

    test_loss, test_accuracy, test_balanced_accuracy, precision, recall, f1, all_preds, all_labels, time_per_sample = run_epoch(
        model, test_loader, criterion, device
    )

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    auc = roc_auc_score(all_labels, all_preds)

    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Bal-Acc: {test_balanced_accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, Specificity: {specificity:.4f}")
    logger.info(f"F1-Score: {f1:.4f}, AUC: {auc:.4f}")
    logger.info(f"Time per sample: {time_per_sample:.4f} seconds in test set")

    cm = confusion_matrix(all_labels, all_preds)
    labels = ['Healthy', 'Sick']
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, save_path)
    logger.info(f"Confusion matrix saved at {save_path}")

if __name__ == "__main__":
    main()
