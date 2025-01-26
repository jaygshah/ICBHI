import argparse
import os
import torch
import numpy as np
import logging
import time
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter

from misc import load_yaml
from custom_model import CustomModel
from models import resnet

CONFIG_FILE = "./configs/train_config.yaml"

def setup_logging(logs_dir):
    """
    to set up logging to file and console.
    """
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, "training.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_dataset(dataset_name):
    """
    Load dataset from .npy files.
    """
    if not os.path.exists(f"./{dataset_name}"):
        raise Exception("Dataset not found.")
    
    def load_data(file_name):
        data = torch.from_numpy(np.load(f"{dataset_name}/{file_name}.npy"))
        return data.unsqueeze(1) if len(data.shape) == 3 else data

    train_x, val_x, test_x = map(load_data, ["x_train", "x_val", "x_test"])
    train_y, val_y, test_y = map(lambda x: torch.from_numpy(np.load(f"{dataset_name}/{x}.npy")).reshape(-1, 1), ["y_train", "y_val", "y_test"])

    return train_x, val_x, test_x, train_y, val_y, test_y

def get_dataloader(dataset, batch_size, oversample=False, y_data=None):
    """
    to create a DataLoader with optional oversampling.
    """
    if oversample:
        class_counts = torch.bincount(y_data.squeeze().long())
        sample_weights = (1.0 / class_counts.float())[y_data.squeeze().long()]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def initialize_model(model_name, model_configs, device):
    """
    Initialize model based on the given name and configurations.
    """
    model_dict = {
        "custom": CustomModel,
        "resnet18": lambda: resnet.resnet18_2d(num_classes=model_configs["num_classes"], in_channels=1),
        "resnet34": lambda: resnet.resnet34_2d(num_classes=model_configs["num_classes"], in_channels=1),
        "resnet50": lambda: resnet.resnet50_2d(num_classes=model_configs["num_classes"], in_channels=1)
    }
    return model_dict.get(model_name, lambda: Exception("Model not found in config file."))().to(device)

def initialize_optimizer(optimizer_name, model, learning_rate):

    optimizers = {"adam": torch.optim.Adam}
    return optimizers.get(optimizer_name, lambda: Exception("Optimizer not found in config file."))(model.parameters(), lr=learning_rate)

def initialize_loss_function(loss_name):

    loss_functions = {"binary_crossentropy": torch.nn.BCELoss}
    return loss_functions.get(loss_name, lambda: Exception("Loss function not found in config file."))()

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    """
    Run one epoch of training or validation (or testing).
    """
    model.train() if train else model.eval()
    epoch_loss, all_preds, all_labels = 0, [], []
    start_time = time.time()

    for x_batch, y_batch in tqdm(loader, desc="Training" if train else "Validation", leave=False):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device).float()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        all_preds.extend(torch.round(y_pred).detach().cpu().numpy().flatten())
        all_labels.extend(y_batch.detach().cpu().numpy().flatten())

    epoch_loss /= len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    end_time = time.time()
    time_per_sample = (end_time - start_time) / len(loader.dataset)

    return epoch_loss, accuracy, balanced_accuracy, all_preds, all_labels, time_per_sample

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

    train_x, val_x, test_x, train_y, val_y, test_y = load_dataset(data_configs["name"])
    train_loader = get_dataloader(TensorDataset(train_x, train_y), model_configs["batch_size"], model_configs["oversample"], train_y)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=model_configs["batch_size"], shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=model_configs["batch_size"], shuffle=False)

    optimizer = initialize_optimizer(model_configs["optimizer"], model, model_configs["learning_rate"])
    criterion = initialize_loss_function(model_configs["loss"])

    save_dir = "_".join([configs["dataset"]["name"], model_configs["name"], configs["save_dir"]])
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, configs["logs_dir"]), exist_ok=True)
    logger = setup_logging(os.path.join(save_dir, configs["logs_dir"]))

    # TensorBoard logging
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard_logs"))

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(model_configs["num_epochs"]):
        epoch_loss, accuracy, balanced_accuracy, _, _, _ = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', accuracy, epoch)
        writer.add_scalar('Balanced_Accuracy/train', balanced_accuracy, epoch)
        
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch}.pth"))

        if epoch % model_configs.get("eval_every", 1) == 0:
            val_loss, val_accuracy, val_balanced_accuracy, _, _, _ = run_epoch(model, val_loader, optimizer, criterion, device, train=False)
            logger.info(f"Epoch: {epoch} | Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, Bal-Acc: {val_balanced_accuracy:.4f}")

            # Log validation metrics to TensorBoard
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            writer.add_scalar('Balanced_Accuracy/val', val_balanced_accuracy, epoch)

            if "early_stopping" in model_configs and model_configs["early_stopping"]:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= model_configs["patience"]:
                    logger.info("Early stopping triggered")
                    break

    #Test evaluation - uncomment if you want to test on unseen set 

    test_loss, test_accuracy, test_balanced_accuracy, all_preds, all_labels, time_per_sample = run_epoch(model, test_loader, optimizer, criterion, device, train=False)
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Test Bal-Acc: {test_balanced_accuracy:.4f}")
    logger.info(f"Time per sample: {time_per_sample:.4f} seconds in test set")

    # Log test metrics to TensorBoard
    writer.add_scalar('Loss/test', test_loss)
    writer.add_scalar('Accuracy/test', test_accuracy)
    writer.add_scalar('Balanced_Accuracy/test', test_balanced_accuracy)

    # Compute and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    labels = ['Healthy', 'Sick']
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, save_path)
    logger.info(f"Confusion matrix saved at {save_path}")

    writer.close()

if __name__ == "__main__":
    main()