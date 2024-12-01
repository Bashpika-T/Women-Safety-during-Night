# --- START OF FILE train_deepmar.py ---
import sys
import os
import numpy as np
import random
import math
import time
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.nn.parallel import DataParallel
import pickle

# Import your custom modules
from baseline.dataset import add_transforms
from baseline.dataset.Dataset import AttDataset
from baseline.model.DeepMAR import DeepMAR_InceptionResNetV2  
from baseline.model.DeepMAR import DeepMAR_InceptionResNetV2_ExtractFeature  
from baseline.utils.evaluate import attribute_evaluate
from baseline.utils.utils import str2bool, transfer_optim_state, save_ckpt, load_ckpt
from baseline.utils.utils import load_state_dict, adjust_lr_staircase, set_devices
from baseline.utils.utils import AverageMeter, to_scalar, may_set_mode, may_mkdir, set_seed

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return time.strftime(fmt)

class Config(object):
    def __init__(self):
        self.sys_device_ids = (0,)
        self.set_seed = False
        self.dataset_names = ['peta', 'pa100k']  # Train on both datasets
        self.split = 'trainval'
        self.test_split = 'test'
        self.partition_idx = 0
        self.resize = (299, 299)  # Input size for Inception-ResNet-v2
        self.mirror = True
        self.batch_size = 8  # Adjust as needed based on your GPU memory
        self.workers = 2
        self.num_att = 61  # Combined attributes (35 PETA + 26 PA100K)
        self.last_conv_stride = 2
        self.drop_pool5 = True
        self.drop_pool5_rate = 0.5
        self.sgd_weight_decay = 0.0005
        self.sgd_momentum = 0.9
        self.new_params_lr = 0.001
        self.finetuned_params_lr = 0.001
        self.staircase_decay_at_epochs = (51,)
        self.staircase_decay_multiple_factor = 0.1
        self.total_epochs = 10
        self.weighted_entropy = False # Weighted entropy removed
        self.resume = False
        self.ckpt_file = ''
        self.load_model_weight = False
        self.model_weight_file = ''
        self.test_only = False
        self.exp_dir = 'C:/Users/Dell/deepmar_Inception-ResNet-v2/exp'  
        self.exp_subpath = 'deepmar_inceptionresnetv2_combined'  
        self.log_to_file = False
        self.steps_per_log = 10
        self.epochs_per_val = 5
        self.epochs_per_save = 2
        self.run = 1

        # Dataset paths 
        self.dataset_paths = {
            'peta': 'X:/PETA/peta_dataset.pkl',   
            'pa100k': "X:/PA100K/Bpa100k_dataset.pkl", 
        }
        self.partition_paths = {
            'peta': 'X:/PETA/peta_partition.pkl',   
            'pa100k': "X:/PA100K/Bpa100k_partition.pkl", 
        }

        self.mean = [0.5, 0.5, 0.5]  
        self.std = [0.5, 0.5, 0.5]  

        # Model kwargs
        self.model_kwargs = {
                    'num_att': self.num_att,
                    'last_conv_stride': self.last_conv_stride,
                    'drop_pool5': self.drop_pool5,
                    'drop_pool5_rate': self.drop_pool5_rate
                }

        # Test kwargs
        self.test_kwargs = {}

    def get_dataset_path(self, dataset_name):
        return self.dataset_paths[dataset_name]
    
    def get_partition_path(self, dataset_name):
        return self.partition_paths[dataset_name]
    
    def setup_logging(self):
        if self.log_to_file:
            log_dir = os.path.join(self.exp_dir, 'log')
            os.makedirs(log_dir, exist_ok=True)
            self.stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
            self.stderr_file = os.path.join(log_dir, f'stderr_{time_str()}.txt')
        else:
            self.stdout_file = None
            self.stderr_file = None

class Combined_Dataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_sizes = [len(d) for d in datasets]
        self.dataset_cumulative_sizes = np.cumsum(self.dataset_sizes)
        self.total_size = sum(self.dataset_sizes)
        self.max_num_att = max([len(datasets[i].dataset['selected_attribute']) for i in range(len(datasets))])

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.dataset_cumulative_sizes, idx, side="right")
        if dataset_idx > 0:
            sample_idx = idx - self.dataset_cumulative_sizes[dataset_idx - 1]
        else:
            sample_idx = idx
        image, label = self.datasets[dataset_idx][sample_idx]

        # Pad the label to the maximum size (61)
        padded_label = np.zeros(61)  # Correct padding size 
        padded_label[:len(label)] = label

        return image, padded_label, dataset_idx 

def main():
    cfg = Config()
    cfg.setup_logging()

    # Print configuration
    print('-' * 60)
    print('cfg.__dict__')
    import pprint
    pprint.pprint(cfg.__dict__)
    print('-' * 60)

    # Set seed for reproducibility
    if cfg.set_seed:
        set_seed(0)

    # Set device
    set_devices(cfg.sys_device_ids)

    # Data Loaders
    datasets = []
    for dataset_name in cfg.dataset_names:
        dataset_path = cfg.get_dataset_path(dataset_name)
        partition_path = cfg.get_partition_path(dataset_name)

        normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)
        transform = transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = AttDataset(
            dataset=dataset_path,
            partition=partition_path,
            split=cfg.split,
            partition_idx=cfg.partition_idx,
            transform=transform)
        datasets.append(train_set)

    combined_train_set = Combined_Dataset(datasets)
    train_loader = torch.utils.data.DataLoader(
        dataset=combined_train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True,
        drop_last=False)

    # Test sets (separate for evaluation)
    test_sets = {}
    for dataset_name in cfg.dataset_names:
        dataset_path = cfg.get_dataset_path(dataset_name)
        partition_path = cfg.get_partition_path(dataset_name)

        test_transform = transforms.Compose([
            transforms.Resize(cfg.resize),
            transforms.ToTensor(),
            normalize,
        ])

        test_set = AttDataset(
            dataset=dataset_path,
            partition=partition_path,
            split=cfg.test_split,
            partition_idx=cfg.partition_idx,
            transform=test_transform)
        test_sets[dataset_name] = test_set

    # Model 
    model = DeepMAR_InceptionResNetV2(**cfg.model_kwargs).cuda()  
    model_w = DataParallel(model)

    # Optimizer
    inception_resnet_params = []
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            inception_resnet_params.append(param)

    param_groups = [
        {'params': inception_resnet_params, 'lr': cfg.finetuned_params_lr},
        {'params': classifier_params, 'lr': cfg.new_params_lr},
    ]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Bind the model and optimizer
    modules_optims = [model, optimizer]

    # Load model weight if necessary
    if cfg.load_model_weight:
        map_location = (lambda storage, loc: storage)
        ckpt = torch.load(cfg.model_weight_file, map_location=map_location)
        model.load_state_dict(ckpt['state_dicts'][0], strict=False)

    # Resume or not
    start_epoch = 0
    if cfg.resume:
        start_epoch, scores = load_ckpt(modules_optims, cfg.ckpt_file)

    model_w = DataParallel(model)
    model_w.cuda()
    transfer_optim_state(state=optimizer.state, device_id=0)

    # For evaluation
    feat_func_att = DeepMAR_InceptionResNetV2_ExtractFeature(model=model_w)

    # Training loop
    for epoch in range(start_epoch, cfg.total_epochs):
        # Adjust learning rate
        if (epoch + 1) in cfg.staircase_decay_at_epochs:
            optimizer.param_groups[0]['lr'] *= cfg.staircase_decay_multiple_factor 

        may_set_mode(modules_optims, 'train')
        loss_meter = AverageMeter()

        ep_st = time.time()
        for step, (imgs, targets, dataset_idx) in enumerate(train_loader):
            step_st = time.time()

            imgs_var = Variable(imgs).cuda()
            targets_var = Variable(targets).cuda()

            score = model_w(imgs_var)

            # --- Compute loss ---
            targets_var[targets_var == -1] = 0

            # --- Separate Losses and Masking ---
            peta_loss = F.binary_cross_entropy_with_logits(score[:, :35], targets_var[:, :35], reduction='none')
            pa100k_loss = F.binary_cross_entropy_with_logits(score[:, 35:61], targets_var[:, 35:61], reduction='none') 

            # --- Apply masks (all ones for both datasets) ---
            peta_mask = torch.ones(len(targets_var), 35).cuda()
            peta_loss = peta_loss * peta_mask 
            pa100k_mask = torch.ones(len(targets_var), 26).cuda() 
            pa100k_loss = pa100k_loss * pa100k_mask  

            # --- Combine Losses ---
            peta_loss = peta_loss.mean() 
            pa100k_loss = pa100k_loss.mean() 
            loss = (peta_loss + pa100k_loss) / 2 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(to_scalar(loss))

            if (step + 1) % cfg.steps_per_log == 0:
                print(f'{time_str()}, Step {step+1}/{len(train_loader)} in Ep {epoch+1}, '
                      f'{time.time() - step_st:.2f}s, loss:{loss_meter.val:.4f}')

        print(f'Ep{epoch+1}, {time.time() - ep_st:.2f}s, loss {loss_meter.avg:.4f}')

        # Save ckpt
        if (epoch + 1) % cfg.epochs_per_save == 0 or epoch + 1 == cfg.total_epochs:
            ckpt_file = os.path.join(cfg.exp_dir, cfg.exp_subpath, f'ckpt_epoch{epoch+1}.pth')
            save_ckpt(modules_optims, epoch + 1, 0, ckpt_file)

        # --- Evaluation ---
        if (epoch + 1) % cfg.epochs_per_val == 0 or epoch + 1 == cfg.total_epochs:
            for dataset_name, test_set in test_sets.items():
                print(f'Evaluating {dataset_name} with feat_func_att')

                result = attribute_evaluate(feat_func_att, test_set, **cfg.test_kwargs)  # Pass mask to evaluation function
                mA = result['instance_acc']
                print(f'mA for {dataset_name}: {mA:.4f}')

if __name__ == '__main__':
    main()
# --- END OF FILE train_deepmar.py ---

