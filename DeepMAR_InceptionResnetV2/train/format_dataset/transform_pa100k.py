import os
import numpy as np
import random
import pickle
from scipy.io import loadmat

np.random.seed(0)
random.seed(0)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def generate_data_description(save_dir):
    dataset = dict()
    dataset['description'] = 'pa100k'
    dataset['root'] = 'X:/PA100K/data/release_data/release_data'
    dataset['image'] = []
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = list(range(26))
    
    # Load ANNOTATION.MAT
    data = loadmat('X:/PA100K/annotation/annotation.mat')
    for idx in range(26):
        dataset['att_name'].append(data['attributes'][idx][0][0])
    
    # Store the first 19k images for balancing
    for idx in range(19000):
        dataset['image'].append(data['train_images_name'][idx][0][0])
        dataset['att'].append(data['train_label'][idx, :].tolist())

    with open(os.path.join(save_dir, 'Bpa100k_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(traintest_split_file):
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    partition['weight_trainval'] = []
    partition['weight_train'] = []
    
    # Load ANNOTATION.MAT
    data = loadmat('X:/PA100K/annotation/annotation.mat')
    
    # Fixed split sizes
    train_size = 9500  # 9500 samples for training
    val_size = 1900    # 1900 samples for validation
    test_size = 7600   # 7600 samples for testing
    total_images = train_size + val_size + test_size  # 19000 total images

    # Shuffling indices
    indices = np.random.permutation(total_images)
    
    # Create new splits based on shuffled indices
    train = indices[:train_size].tolist()
    val = indices[train_size:train_size + val_size].tolist()
    test = indices[train_size + val_size:].tolist()
    trainval = train + val  # Concatenate for trainval

    partition['train'].append(train)
    partition['val'].append(val)
    partition['trainval'].append(trainval)
    partition['test'].append(test)

    # Weight calculation
    train_label = data['train_label'][:total_images].astype('float32')
    trainval_label = np.concatenate((data['train_label'][:total_images], data['val_label']), axis=0).astype('float32')
    
    weight_train = np.mean(train_label[:train_size] == 1, axis=0).tolist()
    weight_trainval = np.mean(trainval_label == 1, axis=0).tolist()

    partition['weight_trainval'].append(weight_trainval)
    partition['weight_train'].append(weight_train)

    with open(traintest_split_file, 'wb') as f:
        pickle.dump(partition, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PA100K dataset")
    parser.add_argument(
        '--save_dir',
        type=str,
        default='X:/PA100K/')
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="X:/PA100K/Bpa100k_partition.pkl")
    args = parser.parse_args()
    
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    make_dir(save_dir)
    generate_data_description(save_dir)
    create_trainvaltest_split(traintest_split_file)
