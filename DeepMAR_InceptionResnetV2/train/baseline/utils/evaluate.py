# --- START OF FILE evaluate.py ---
import torch
from torch.autograd import Variable
import numpy as np
import time

def extract_feat(feat_func, dataset, **kwargs):
    """
    Extract features for images.
    """
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32,
        num_workers=2, pin_memory=True)

    start_time = time.time()
    total_eps = len(test_loader)
    N = len(dataset.image)
    start = 0

    for ep, (imgs, labels) in enumerate(test_loader):
        with torch.no_grad(): 
            imgs_var = Variable(imgs).cuda()
            feat_tmp = feat_func( imgs_var )
        batch_size = feat_tmp.shape[0]
        if ep == 0:
            # Change this line:
            feat = np.zeros((N, feat_tmp.shape[1]))  # Use feat_tmp.shape[1]
        feat[start:start+batch_size, :] = feat_tmp.cpu().numpy().reshape((batch_size, -1))
        start += batch_size

    end_time = time.time()
    print('{} batches done, total {:.2f}s'.format(total_eps, end_time - start_time))
    return feat 


def attribute_evaluate(feat_func, dataset, mask=None, **kwargs):
    print("Extracting features for attribute recognition")
    pt_result = extract_feat(feat_func, dataset)
    
    # Obtain the attributes from the attribute dictionary
    print("Computing attribute recognition result")
    N = pt_result.shape[0]
    L = pt_result.shape[1]
    gt_result = np.zeros(pt_result.shape)

    # Get the groundtruth attributes
    for idx, label in enumerate(dataset.label):
        gt_result[idx, :len(label)] = label  # Fill only the valid label positions

    pt_result[pt_result >= 0] = 1
    pt_result[pt_result < 0] = 0

    return attribute_evaluate_lidw(gt_result, pt_result, mask=mask)  # Pass the mask to attribute_evaluate_lidw

def attribute_evaluate_lidw(gt_result, pt_result, mask=None):
    """
    Input:
    gt_result, pt_result, N*L, with 0/1
    Output:
    result
    a dictionary, including label-based and instance-based evaluation
    label-based: label_pos_acc, label_neg_acc, label_acc
    instance-based: instance_acc, instance_precision, instance_recall, instance_F1
    """
    # Apply the mask if it's provided
    if mask is not None:
        gt_result = gt_result * mask  # Element-wise multiplication with the mask
        pt_result = pt_result * mask
    if gt_result.shape != pt_result.shape:
        raise ValueError('Shape between groundtruth and predicted results are different')

    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    
    label_pos_acc = pt_pos / gt_pos
    label_neg_acc = pt_neg / gt_neg
    label_acc = (label_pos_acc + label_neg_acc) / 2
    
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc

    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    intersect_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)

    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff -= 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    
    instance_acc = np.sum(intersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(intersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(intersect_pos / gt_pos) / cnt_eff
    instance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = instance_F1

    return result

# Debugging helper function
def run_evaluation(feat_func, dataset):
    try:
        result = attribute_evaluate(feat_func, dataset)
        print(result)
    except ValueError as e:
        print(f"Error occurred: {e}")