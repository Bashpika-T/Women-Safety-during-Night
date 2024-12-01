import os
import pickle
import datetime
import time
import torch
from torch.autograd import Variable
import random
import numpy as np

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.datetime.today().strftime(fmt)

def str2bool(v):
    return v.lower() in ("yes", "true", "1")

def is_iterable(obj):
    return hasattr(obj, '__len__')

def to_scalar(vt):
    """
    Transform a 1-length PyTorch Variable or Tensor to a scalar.
    """
    if isinstance(vt, Variable):
        return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        return vt.cpu().numpy().flatten()[0]
    raise TypeError('Input should be a Variable or Tensor')

def set_seed(rand_seed):
    np.random.seed(rand_seed)
    random.seed(rand_seed)
    torch.backends.cudnn.enabled = True
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def may_mkdir(fname):
    if not os.path.exists(os.path.dirname(os.path.abspath(fname))):
        os.makedirs(os.path.dirname(os.path.abspath(fname)))

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-10)

class RunningAverageMeter(object):
    """
    Computes and stores the running average and current value
    """
    def __init__(self, hist=0.99):
        self.val = None
        self.avg = None
        self.hist = hist
    
    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val

class RecentAverageMeter(object):
    """
    Stores and computes the average of recent values
    """
    def __init__(self, hist_size=100):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, value):
        self.val = value  # Change 'val' to 'value'
        self.fifo.append(value)  # Also change 'val' to 'value'
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)


class ReDirectSTD(object):
    """
    overwrites the sys.stdout or sys.stderr
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visiable: False
    Usage example:
      ReDirectSTD('stdout.txt', 'stdout', False)
      ReDirectSTD('stderr.txt', 'stderr', False)
    """
    def __init__(self, fpath=None, console='stdout', immediately_visiable=False):
        import sys
        import os
        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == "stdout" else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visiable = immediately_visiable
        if fpath is not None:
            # Remove existing log file
            if os.path.exists(fpath):
                os.remove(fpath)
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, **args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            if not os.path.exists(os.path.dirname(os.path.abspath(self.file))):
                os.mkdir(os.path.dirname(os.path.abspath(self.file)))
            if self.immediately_visiable:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())
    
    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()

def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1

def set_devices(sys_device_ids):
    """
    Set the CUDA_VISIBLE_DEVICES environment variable.

    Args:
        sys_device_ids: A tuple specifying the GPU IDs to use. 
    """
    import os
    visible_devices = ''
    for i in sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

def transfer_optims(optims, device_id=-1):
    for optim in optims:
        if isinstance(optim, torch.optim.Optimizer):
            transfer_optim_state(optim.state, device_id=device_id)

def transfer_optim_state(state, device_id=-1):
    for key, val in state.items():
        if isinstance(val, dict):
            transfer_optim_state(val, device_id=device_id)
        elif isinstance(val, Variable):
            raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
        elif isinstance(val, torch.nn.Parameter):
            raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
        else:
            try:
                if device_id == -1:
                    state[key] = val.cpu()
                else:
                    state[key] = val.cuda(device=device_id)
            except:
                pass

def load_state_dict(model, src_state_dict):
    """
    Copy parameters from src_state_dict to model.

    Args:
        model: A torch.nn.Module object.
        src_state_dict: A dictionary containing parameters and persistent buffers.
    """
    from torch.nn import Parameter
    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception as msg:
            print("Warning: Error occurs when copying '{}': {}".format(name, str(msg)))

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        for n in dest_missing:
            print('\t', n)

def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
    """
    Load the state_dict of modules/optimizers from a file.
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)

    # Check if 'state_dicts' is a list or a single state dictionary
    if isinstance(ckpt['state_dicts'], list):
        # Load the state dictionary for the model (first element in modules_optims)
        modules_optims[0].load_state_dict(ckpt['state_dicts'][0])  # Access the first element of the list 
    else:
        # Load the single state dictionary directly
        modules_optims[0].load_state_dict(ckpt['state_dicts']) 

def save_ckpt(modules_optims, ep, scores, ckpt_file):
    """
    Save the state_dict of modules/optimizers to a file.

    Args:
        modules_optims: A list containing the model and optimizer.
        ep: The current epoch number.
        scores: The performance scores.
        ckpt_file: The checkpoint file path.
    """
    state_dicts = [m.state_dict() for m in modules_optims]
    ckpt = dict(state_dicts=state_dicts,
                ep=ep,
                scores=scores)
    may_mkdir(ckpt_file)
    torch.save(ckpt, ckpt_file)

def adjust_lr_staircase(param_groups, base_lrs, ep, decay_at_epochs, factor):
    """
    Adjust learning rate according to a staircase schedule.
    """
    assert len(base_lrs) == len(param_groups), \
        "You should specify a base learning rate for each parameter group."
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)
    for i, (g, base_lr) in enumerate(zip(param_groups, base_lrs)):
        g['lr'] = base_lr * factor ** (ind + 1)
        print('=====> Parameter group {}: Learning rate adjusted to {:.10f}'.format(i, g['lr']).rstrip('0'))

def may_set_mode(maybe_modules, mode):
    """
    Set the mode (train or eval) for the given modules.
    """
    assert mode in ['train', 'eval']
    if not is_iterable(maybe_modules):
        maybe_modules = [maybe_modules]
    for m in maybe_modules:
        if isinstance(m, torch.nn.Module):
            if mode == 'train':
                m.train()
            else:
                m.eval()