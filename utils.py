# src/utils.py
import os
import random
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, confusion_matrix
import math
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        pass

def autocrop_background(img):
    # img: BGR or RGB numpy image
    # convert to gray and threshold to remove black border
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(th)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]
    return cropped



def compute_metrics(y_true, y_pred, threshold=0.5):
    y_bin = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_bin, labels=[0, 1])
    
    # Handle case where only one class exists
    if cm.shape == (1, 1):
        tn = cm[0, 0] if y_true[0] == 0 else 0
        tp = cm[0, 0] if y_true[0] == 1 else 0
        fp = fn = 0
    else:
        tn, fp, fn, tp = cm.ravel()

    # Avoid divide-by-zero
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = 0.0

    return {"sensitivity": sensitivity, "specificity": specificity, "auc": auc}


def bootstrap_ci(metric_func, y_true, y_scores, iters=10000, alpha=0.05):
    # metric_func(y_true, y_scores) should return a scalar
    import numpy as np
    n = len(y_true)
    idx = np.arange(n)
    stats = []
    for _ in range(iters):
        s = np.random.choice(idx, size=n, replace=True)
        stats.append(metric_func(np.array(y_true)[s], np.array(y_scores)[s]))
    lower = np.percentile(stats, 100*(alpha/2))
    upper = np.percentile(stats, 100*(1-alpha/2))
    return lower, upper

def mcnemar_test(y_true, preds_a, preds_b):
    # returns chi2 and p-value
    # Build contingency table
    import numpy as np
    a = np.array(preds_a)
    b = np.array(preds_b)
    t = np.array(y_true)
    # Cases where one correct and other incorrect:
    correct_a = (a == t)
    correct_b = (b == t)
    n10 = np.sum((correct_a==True) & (correct_b==False))
    n01 = np.sum((correct_a==False) & (correct_b==True))
    # McNemar's test with continuity correction
    from scipy.stats import chi2
    stat = (abs(n10 - n01) - 1)**2 / (n10 + n01 + 1e-9)
    p = 1 - chi2.cdf(stat, df=1)
    return stat, p
