import numpy as np
import cv2
import collections
import yaml
import pandas as pd
import requests
import pydicom
import torch
import math
import os

from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef, accuracy_score

def get_acc(y, prob, keys):
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    acc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            acc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            acc_dict[k] = 0.0
        else:
            # Convert probabilities to binary predictions
            prob_cls_binary = (prob_cls >= 0.5).astype(int)
            # Calculate accuracy
            acc_dict[k] = accuracy_score(y_cls, prob_cls_binary)
    values = list(acc_dict.values())
    acc_dict["mean"] = float(np.mean(values))
    return acc_dict

def get_auroc(y, prob, keys):
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    auroc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auroc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            auroc_dict[k] = 0.0
        else:
            auroc_dict[k] = roc_auc_score(y_cls, prob_cls)
    values = list(auroc_dict.values())
    auroc_dict["mean"] = float(np.mean(values))
    return auroc_dict

def get_mcc(y, prob, keys):
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    mcc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            mcc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            mcc_dict[k] = 0.0
        else:
            # Convert probabilities to binary predictions
            prob_cls_binary = (prob_cls >= 0.5).astype(int)
            # Calculate Matthews correlation coefficient
            mcc_dict[k] = matthews_corrcoef(y_cls, prob_cls_binary)
    values = list(mcc_dict.values())
    mcc_dict["mean"] = float(np.mean(values))
    return mcc_dict



def get_auprc(y, prob, keys):
    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    if len(y.shape) == 1:
        y = np.expand_dims(y, -1)
    if len(prob.shape) == 1:
        prob = np.expand_dims(prob, -1)

    auprc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auprc_dict[k] = 0.0
        elif len(set(y_cls)) == 1:
            auprc_dict[k] = 0.0
        else:
            auprc_dict[k] = average_precision_score(y_cls, prob_cls)
    values = list(auprc_dict.values())
    auprc_dict["mean"] = float(np.mean(values))
    return auprc_dict


import tarfile
import io


def read_tar_dicom(tar_file_path):
    tar_contents = {}
    try:
        # Open the tar file as a binary stream
        with tarfile.open(tar_file_path, "r") as tar:
            # Iterate through the files in the tar archive
            for tar_info in tar:
                # Check if the tar entry is a regular file (not a directory or a symlink)
                if tar_info.isfile():
                    # Read the content of the file into a variable
                    content = tar.extractfile(tar_info).read()

                    # Store the content in the dictionary with the file name as the key
                    tar_contents[tar_info.name] = content

    except tarfile.TarError as e:
        print(f"Error while processing the tar file: {e}")

    return tar_contents


def has_parameter(cls, name):
    import inspect
    return name in inspect.signature(cls.__init__).parameters

def get_latest_ckpt(config):
    config_ckpt, dataset_target = config.ckpt, config.dataset.target
    assert os.path.isdir(config_ckpt), f"{config_ckpt} is not a directory"
    if config_ckpt.endswith(".ckpt"):  # and not os.path.isfile(config_ckpt):
        latest_ckpt = config_ckpt
    else:
        task_paths = [
            os.path.join(config_ckpt, task_path)
            for task_path in os.listdir(config_ckpt)
            if dataset_target in task_path
        ]
        ckpt_paths = [
            os.path.join(task_path, ckpt_path)
            for task_path in task_paths
            for ckpt_path in os.listdir(task_path)
            if ckpt_path.endswith(".ckpt")
        ]
        latest_ckpt = max(ckpt_paths, key=os.path.getctime)
        # while not os.path.isfile(latest_ckpt):
        #     ckpt_paths.remove(latest_ckpt)
        #     latest_ckpt = max(ckpt_paths, key=os.path.getctime)
    print(f"Loading latest checkpoint: {latest_ckpt}")
    return latest_ckpt

def mask_to_bbox(mask):
    """
    Extracts the bounding box from a binary mask.
    Returns (x_min, y_min, w, h) normalized by mask dimensions.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return np.array([0, 0, 0, 0], dtype=np.float32)
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    h_img, w_img = mask.shape
    
    x = cmin / w_img
    y = rmin / h_img
    w = (cmax - cmin) / w_img
    h = (rmax - rmin) / h_img
    
    return np.array([x, y, w, h], dtype=np.float32)

def calculate_iou(boxA, boxB):
    # boxA, boxB: [x, y, w, h] - normalized
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

def draw_bboxes_on_slice(image, bbox_pred, bbox_gt, output_path):
    """
    image: numpy array (H, W, 3) or (H, W)
    bbox: [x, y, w, h] normalized
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    h_img, w_img = image.shape[:2]
    
    # Scale bboxes
    x_min_pred = int(bbox_pred[0] * w_img)
    y_min_pred = int(bbox_pred[1] * h_img)
    w_pred = int(bbox_pred[2] * w_img)
    h_pred = int(bbox_pred[3] * h_img)
    
    x_min_gt = int(bbox_gt[0] * w_img)
    y_min_gt = int(bbox_gt[1] * h_img)
    w_gt = int(bbox_gt[2] * w_img)
    h_gt = int(bbox_gt[3] * h_img)
    
    # Draw Pred (Green)
    cv2.rectangle(image, (x_min_pred, y_min_pred), 
                  (x_min_pred + w_pred, y_min_pred + h_pred), (0, 255, 0), 2)
    
    # Draw GT (Red)
    cv2.rectangle(image, (x_min_gt, y_min_gt), 
                  (x_min_gt + w_gt, y_min_gt + h_gt), (0, 0, 255), 2)
    
    cv2.imwrite(output_path, image)

def calculate_map(preds, gts, thresholds=None):
    """
    preds: list of [x, y, w, h, score] or list of [x, y, w, h] if scores are separate
    gts: list of [x, y, w, h]
    thresholds: list of IoU thresholds, e.g., [0.5, 0.75, 0.95]
    """
    if thresholds is None:
        thresholds = np.linspace(0.5, 0.95, 10) # COCO mAP 50-95
    
    # This is a simplified version for a single class (nodule)
    # in medical imaging we often use mAP@0.5 or mAP@0.1-0.5 depending on clinical relevance
    
    aps = []
    for th in thresholds:
        tp = 0
        fp = 0
        fn = 0
        # For simplicity, if we have one detection per slice:
        # (This should be refined for multi-nodule cases)
        for p, g in zip(preds, gts):
            iou = calculate_iou(p[:4], g[:4])
            if iou >= th:
                tp += 1
            else:
                fp += 1
                fn += 1 # assuming one GT exists
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        aps.append(precision) # Simplified AP
        
    return np.mean(aps)

def calculate_sensitivity(preds, gts):
    tp = 0
    fn = 0
    for p, g in zip(preds, gts):
        iou = calculate_iou(p[:4], g[:4])
        if iou >= 0.5:
            tp += 1
        else:
            fn += 1
    return tp / (tp + fn + 1e-8)

def calculate_specificity(preds, gts):
    tn = 0
    fp = 0
    for p, g in zip(preds, gts):
        iou = calculate_iou(p[:4], g[:4])
        if iou < 0.5:
            tn += 1
        else:
            fp += 1
    return tn / (tn + fp + 1e-8)

def calculate_accuracy(preds, gts):
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for p, g in zip(preds, gts):
        iou = calculate_iou(p[:4], g[:4])
        if iou >= 0.5:
            if g[4] >= 0.5:
                tp += 1
            else:
                fp += 1
        else:
            if g[4] >= 0.5:
                fn += 1
            else:
                tn += 1
    return (tp + tn) / (tp + fp + fn + tn + 1e-8)

def calculate_F1(preds, gts):
    tp = 0
    fp = 0
    fn = 0
    for p, g in zip(preds, gts):
        iou = calculate_iou(p[:4], g[:4])
        if iou >= 0.5:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp / (tp + fp + fn + 1e-8)

def calculate_FROC(preds, gts):
    tp = 0
    fp = 0
    fn = 0
    for p, g in zip(preds, gts):
        iou = calculate_iou(p[:4], g[:4])
        if iou >= 0.5:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp / (tp + fp + fn + 1e-8)