# --- START OF FILE DeepMAR.py ---
import torch
import torch.nn as nn
from .inceptionresnetv2 import inceptionresnetv2

class DeepMAR_InceptionResNetV2(nn.Module):
    def __init__(self, num_att, last_conv_stride, drop_pool5, drop_pool5_rate):  
        super(DeepMAR_InceptionResNetV2, self).__init__()
        self.num_att = num_att
        self.last_conv_stride = last_conv_stride
        self.drop_pool5 = drop_pool5
        self.drop_pool5_rate = drop_pool5_rate

        # Initialize InceptionResNetV2 (NO pretrained weights)
        self.base = inceptionresnetv2(num_classes=1000, pretrained=None) 

        # Replace last_linear with a new layer for attribute prediction
        self.base.last_linear = nn.Linear(1536, self.num_att)

    def forward(self, x):
        x = self.base(x)
        return x

class DeepMAR_InceptionResNetV2_ExtractFeature(object):
    """
    A feature extraction function for InceptionResNetV2
    """
    def __init__(self, model, **kwargs):
        self.model = model

    def __call__(self, imgs):
        old_train_eval_model = self.model.training

        self.model.eval()

        # imgs should be Tensor
        if not isinstance(imgs, torch.Tensor):
            raise ValueError('imgs should be of type: torch.Tensor')
        score = self.model(imgs) 

        self.model.train(old_train_eval_model)

        return score  

# --- END OF FILE DeepMAR.py ---

