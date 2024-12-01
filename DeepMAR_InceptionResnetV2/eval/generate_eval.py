import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.nn.parallel import DataParallel
import pickle
from torchvision.models import ResNet50_Weights
from PIL import Image  # Import Image for loading images
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import your custom modules
from baseline.model.DeepMAR import DeepMAR_ResNet50
from baseline.model.DeepMAR import DeepMAR_ResNet50_ExtractFeature
from baseline.utils.utils import load_ckpt, may_mkdir

# --- Configuration ---
class Config(object):
    def __init__(self):
        self.sys_device_ids = (0,)
        self.dataset_names = ['peta', 'pa100k']  # Datasets for confusion matrices
        self.split = 'test'
        self.partition_idx = 0
        self.resize = (224, 224)
        self.batch_size = 32
        self.workers = 4
        self.num_att = 35  # Will be updated dynamically
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.ckpt_file = (  # Path to your trained model
            'ckpt_epoch25.pth'
        )

        # Dataset paths (Direct paths)
        self.dataset_paths = {
            'peta': 'X:/PETA/peta_dataset.pkl',
            'pa100k': 'X:/PA100K/Bpa100k_dataset.pkl',
        }
        self.partition_paths = {
            'peta': 'X:/PETA/peta_partition.pkl',
            'pa100k': 'X:/PA100K/Bpa100k_partition.pkl',
        }

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Model kwargs
        self.model_kwargs = {
            'num_att': self.num_att,
            'last_conv_stride': self.last_conv_stride,
            'drop_pool5': self.drop_pool5,
            'drop_pool5_rate': self.drop_pool5_rate,
            'weights': ResNet50_Weights.IMAGENET1K_V1,
        }

    def get_dataset_path(self, dataset_name):
        return self.dataset_paths[dataset_name]

    def get_partition_path(self, dataset_name):
        return self.partition_paths[dataset_name]

# --- Function to Generate Predictions ---
def generate_predictions(model_w, image_paths, cfg, device):
    """Generates predictions for a list of image paths."""
    normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
    transform = transforms.Compose([
        transforms.Resize(cfg.resize),
        transforms.ToTensor(),
        normalize
    ])

    feat_func_att = DeepMAR_ResNet50_ExtractFeature(model=model_w)
    predictions = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            score = feat_func_att(img)
        score = torch.sigmoid(score).cpu().numpy()
        score = (score > 0.5).astype(int)  # Proper thresholding
        predictions.append(score[0])  # Assuming batch size of 1

    return np.array(predictions)

# --- Function to Generate Confusion Matrices ---
import pandas as pd

def generate_confusion_matrix(ground_truth_df, predictions, dataset_name, att_names):
    """Generates and plots the confusion matrix and saves the data to an Excel file."""
    gt_attributes_str = ground_truth_df[ground_truth_df['dataset'] == dataset_name]['attributes'].values

    # Convert the string representation of attributes to a NumPy array
    gt_attributes = np.array([list(map(int, attr_str.split(','))) for attr_str in gt_attributes_str])

    # Align the number of attributes between ground truth and predictions
    if dataset_name == 'peta':
        gt_attributes = gt_attributes[:, :35]  # Use only the first 35 attributes for PETA
        predictions = predictions[:, :35]
    elif dataset_name == 'pa100k':
        gt_attributes = gt_attributes[:, :26]  # Use only the first 26 attributes for PA100K
        predictions = predictions[:, :26]

    cm = multilabel_confusion_matrix(gt_attributes, predictions)
    
    # Save confusion matrix to Excel
    save_confusion_matrix_to_csv(cm, dataset_name, att_names)

    return cm

import pandas as pd
import os

import pandas as pd
import os

import pandas as pd
import os

def save_confusion_matrix_to_csv(cm, dataset_name, att_names):
    """Saves confusion matrices (TP, TN, FP, FN) to a CSV file for each attribute."""
    save_path = f'confusion_matrices/{dataset_name}_confusion_matrix.csv'
    ensure_directory_exists(os.path.dirname(save_path))  # Ensure the directory exists

    # Create a DataFrame to store all confusion matrices for attributes
    rows = []
    for i, matrix in enumerate(cm):
        tn, fp, fn, tp = matrix.ravel()
        # Append a row for each attribute with its confusion matrix values
        rows.append({
            'Attribute': att_names[i],
            'True Positive': tp,
            'True Negative': tn,
            'False Positive': fp,
            'False Negative': fn
        })

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    # Save DataFrame to CSV
    df.to_csv(save_path, index=False)
    print(f"Confusion matrices saved to {save_path}")


# --- Ensure directory creation ---
def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# --- Example Main Code Integration ---
def main():
    cfg = Config()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Loading ---
    model = DeepMAR_ResNet50(**cfg.model_kwargs).to(device)
    
    # Load checkpoint
    ckpt = torch.load(cfg.ckpt_file, map_location=device)

    if 'state_dicts' in ckpt:
        state_dict = ckpt['state_dicts'][0]
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        raise KeyError("Checkpoint file is missing 'state_dicts' or 'state_dict' key.")

    # Load state dict with strict=False to ignore mismatches
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Wrap model with DataParallel
    model_w = DataParallel(model).to(device)

    # --- Ground Truth Data Loading ---
    # Ensure that 'ground_truth.csv' has columns: 'dataset', 'image_path', 'attributes'
    ground_truth_df = pd.read_csv('ground_truth.csv')
    print("Ground truth data loaded.")

    all_predictions = {}  # Store predictions for all datasets

    for dataset_name in cfg.dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Get image paths for the current dataset
        image_paths = ground_truth_df[ground_truth_df['dataset'] == dataset_name]['image_path'].values

        # Update num_att dynamically BEFORE generating predictions
        if dataset_name == 'peta':
            cfg.num_att = 35
        elif dataset_name == 'pa100k':
            cfg.num_att = 26
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Generate predictions
        predictions = generate_predictions(model_w, image_paths, cfg, device)
        all_predictions[dataset_name] = predictions  # Store predictions

        # Get attribute names from the dataset info
        with open(cfg.get_dataset_path(dataset_name), 'rb') as f:
            dataset_info = pickle.load(f)
        att_names = [dataset_info['att_name'][i] for i in dataset_info['selected_attribute']]

        # Generate confusion matrices
        cm = generate_confusion_matrix(
            ground_truth_df, predictions, dataset_name, att_names
        )

        # Save confusion matrices to CSV
        save_confusion_matrix_to_csv(cm, dataset_name, att_names)

        print(f"Confusion matrices for {dataset_name} saved successfully.")

    print("All confusion matrices have been generated and saved.")

# --- Execute Main ---
if __name__ == '__main__':
    main()
