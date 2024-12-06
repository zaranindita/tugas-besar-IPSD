import numpy as np
import cv2 as cv
import os
import torch

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, base_folder_aug, base_folder_orig):
        """
        :param base_folder_aug: path folder augmented
        :param base_folder_orig: path folder original 
        """
        self.dataset_aug = []
        self.dataset_train = []
        self.dataset_test = []
        self.dataset_valid = []
        onehot = np.eye(6) 
        
        # load data aug
        for fold_num in range(1, 6):
            aug_folder = os.path.join(base_folder_aug, f"fold{fold_num}_AUG", "Train")
            if not os.path.exists(aug_folder):
                print(f"augmented folder not found: {aug_folder}")
                continue
            for class_idx, class_name in enumerate(os.listdir(aug_folder)):
                class_folder = os.path.join(aug_folder, class_name)
                if not os.path.isdir(class_folder):
                    continue
                for img_name in os.listdir(class_folder):
                    img_path = os.path.join(class_folder, img_name)
                    image = cv.imread(img_path)
                    if image is None:
                        continue
                    image = cv.resize(image, (32, 32)) / 255.0
                    self.dataset_aug.append([image, onehot[class_idx]])
        
        # data ori
        for fold_num in range(1, 6):
            fold_folder = os.path.join(base_folder_orig, f"fold{fold_num}")
            if not os.path.exists(fold_folder):
                print(f"original fold folder not found: {fold_folder}")
                continue

            # load train
            trainfolder = os.path.join(fold_folder, "train")
            if os.path.exists(trainfolder):
                for class_idx, class_name in enumerate(os.listdir(trainfolder)):
                    class_folder = os.path.join(trainfolder, class_name)
                    if not os.path.isdir(class_folder):
                        continue
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        image = cv.imread(img_path)
                        if image is None:
                            continue
                        image = cv.resize(image, (32, 32)) / 255.0
                        self.dataset_train.append([image, onehot[class_idx]])

            # load test
            testfolder = os.path.join(fold_folder, "test")
            if os.path.exists(testfolder):
                for class_idx, class_name in enumerate(os.listdir(testfolder)):
                    class_folder = os.path.join(testfolder, class_name)
                    if not os.path.isdir(class_folder):
                        continue
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        image = cv.imread(img_path)
                        if image is None:
                            continue
                        image = cv.resize(image, (32, 32)) / 255.0
                        self.dataset_test.append([image, onehot[class_idx]])

            # load valid
            validfolder = os.path.join(fold_folder, "valid")
            if os.path.exists(validfolder):
                for class_idx, class_name in enumerate(os.listdir(validfolder)):
                    class_folder = os.path.join(validfolder, class_name)
                    if not os.path.isdir(class_folder):
                        continue
                    for img_name in os.listdir(class_folder):
                        img_path = os.path.join(class_folder, img_name)
                        image = cv.imread(img_path)
                        if image is None:
                            continue
                        image = cv.resize(image, (32, 32)) / 255.0
                        self.dataset_valid.append([image, onehot[class_idx]])
        
        # debugging 
        print(f"augmented images (train): {len(self.dataset_aug)}")
        print(f"original images (train): {len(self.dataset_train)}")
        print(f"original images (test): {len(self.dataset_test)}")
        print(f"original images (valid): {len(self.dataset_valid)}")

    def __len__(self):
        """Mengembalikan jumlah data di augmented images."""
        return len(self.dataset_aug)

    def __getitem__(self, idx):
        """
        :param idx: index data
        :return: tuple (image, label) dalam format tensor
        """
        features, label = self.dataset_aug[idx]
        return (torch.tensor(features, dtype=torch.float32).permute(2, 0, 1),  
                torch.tensor(label, dtype=torch.float32))


if __name__ == "__main__":
    aug_path = "C:/Users/Admin/Documents/Praktikum IPSD/TUGAS/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/"
    orig_path = "C:/Users/Admin/Documents/Praktikum IPSD/TUGAS/Dataset/Original Images/Original Images/FOLDS/"

    data = Data(base_folder_aug=aug_path, base_folder_orig=orig_path)
