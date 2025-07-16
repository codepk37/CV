import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class ModelCheckpoint:
    def __init__(self, model_name="mlp_model"):
        self.best_val_acc = 0
        self.model_name = model_name
        self.best_epoch = 0

    def save_checkpoint(self, model, val_acc, epoch):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch
            }
            torch.save(checkpoint, f'./ckpt/{self.model_name}_best.pth')
            return True
        return False


# Train and evaluate
def train_and_evaluate(model, train_loader, val_loader, num_epochs=5, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_handler = ModelCheckpoint()

    if wandb.run is not None:
        wandb.watch(model, log="all", log_freq=10)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        wandb.log({"Epoch": epoch + 1, "Train Loss": train_loss / len(train_loader), "Train Accuracy": train_accuracy})

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss / len(val_loader), "Validation Accuracy": val_accuracy})

        # Save checkpoint if validation accuracy improved
        is_best = checkpoint_handler.save_checkpoint(model, val_accuracy, epoch)
        if is_best:
            print(f'New best model saved! (Validation Accuracy: {val_accuracy:.2f}%)')

