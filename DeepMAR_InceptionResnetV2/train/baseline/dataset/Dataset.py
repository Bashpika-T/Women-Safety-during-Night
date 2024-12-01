#Dataset.py
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import pickle  # Changed from cPickle to pickle for Python 3

class AttDataset(data.Dataset):
    """
    Person attribute dataset interface
    """
    def __init__(
        self, 
        dataset,
        partition,
        split='train',
        partition_idx=0,
        transform=None,
        target_transform=None,
        **kwargs):
        
        # Load dataset and partition
        if os.path.exists(dataset):
            with open(dataset, 'rb') as f:
                self.dataset = pickle.load(f)
            print("Dataset loaded successfully.")
            print(f"Number of images in dataset: {len(self.dataset['image'])}")
        else:
            raise ValueError(f"{dataset} does not exist.")
        
        if os.path.exists(partition):
            with open(partition, 'rb') as f:
                self.partition = pickle.load(f)
            print("Partition loaded successfully.")
            print(f"Available splits in partition: {list(self.partition.keys())}")
        else:
            raise ValueError(f"{partition} does not exist.")
        
        if split not in self.partition:
            raise ValueError(f"{split} does not exist in partition.")
        
        if partition_idx >= len(self.partition[split]):
            raise ValueError("partition_idx is out of range in partition.")
        
        self.transform = transform
        self.target_transform = target_transform

        # Create image and label lists based on the selected partition and dataset split
        self.root_path = self.dataset['root']
        self.att_name = [self.dataset['att_name'][i] for i in self.dataset['selected_attribute']]
        self.image = []
        self.label = []
        
        # Check number of indices in the selected split
        for idx in self.partition[split][partition_idx]:
            if idx < len(self.dataset['image']):
                self.image.append(self.dataset['image'][idx])
                label_tmp = np.array(self.dataset['att'][idx])[self.dataset['selected_attribute']].tolist()
                self.label.append(label_tmp)
            else:
                print(f"Index {idx} is out of bounds for dataset with length {len(self.dataset['image'])}")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the index of the target class
        """
        imgname, target = self.image[index], self.label[index]
        
        # Load image and labels
        imgname = os.path.join(self.dataset['root'], imgname)
        img = Image.open(imgname).convert('RGB')  # Ensure image is in RGB format
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Process target
        target = np.array(target).astype(np.float32)
        target[target == 0] = -1
        target[target == 2] = 0
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.image)
