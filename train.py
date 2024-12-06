import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms

from torchvision.datasets import ImageFolder
from Utils.getData import Data


def main():
    BATCH_SIZE = 8
    EPOCH = 15
    LEARNING_RATE = 0.001
    NUM_CLASSES = 6  

    # dataset
    aug_path = "C:/Users/Admin/Documents/Praktikum IPSD/TUGAS/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/Admin/Documents/Praktikum IPSD/TUGAS/Dataset/Original Images/Original Images/FOLDS/"
    dataset = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
    full_data = dataset.dataset_train + dataset.dataset_aug

    # split train dan valid (80-20)
    train_size = int(0.8 * len(full_data))
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size])

    # data loader
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # model, loss func, optim
    model = models.mobilenet_v3_large(pretrained=True)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []

    # training loop
    for epoch in range(EPOCH):
        model.train()
        loss_train, correct_train, total_train = 0.0, 0, 0

        for src, trg in train_loader:
            src = src.permute(0, 3, 1, 2).float()
            trg = torch.argmax(trg, dim=1)

            pred = model(src)
            loss = loss_fn(pred, trg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            _, predicted = torch.max(pred, 1)
            total_train += trg.size(0)
            correct_train += (predicted == trg).sum().item()

        accuracy_train = 100 * correct_train / total_train
        train_losses.append(loss_train / len(train_loader))

        # valid
        model.eval()
        loss_val, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for src, trg in val_loader:
                src = src.permute(0, 3, 1, 2).float()
                trg = torch.argmax(trg, dim=1)

                pred = model(src)
                loss = loss_fn(pred, trg)

                loss_val += loss.item()
                _, predicted = torch.max(pred, 1)
                total_val += trg.size(0)
                correct_val += (predicted == trg).sum().item()

        accuracy_val = 100 * correct_val / total_val
        val_losses.append(loss_val / len(val_loader))

        print(f"Epoch [{epoch + 1}/{EPOCH}], "
              f"Train Loss: {train_losses[-1]:.4f}, Accuracy: {accuracy_train:.2f}%, "
              f"Val Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy_val:.2f}%")

    # simpan
    torch.save(model.state_dict(), "trained_modelmobilenet.pth")

    plt.plot(range(EPOCH), train_losses, color="#3399e6", label='Train Loss')
    plt.plot(range(EPOCH), val_losses, color="#ff5733", label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig("./train_val_loss.png")
    plt.show()

if __name__ == "__main__":
    main()
