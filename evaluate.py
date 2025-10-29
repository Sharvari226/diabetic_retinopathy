# src/evaluate.py
import torch
import yaml
import os
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import FundusDataset
from src.transforms import get_transforms
from src.model import DRMultiTaskModel
from src.utils import compute_metrics, bootstrap_ci, mcnemar_test
import numpy as np
from tqdm import tqdm
import argparse

def load_model_ckpt(ckpt_path, device='cpu'):
    # create model and load weights
    model = DRMultiTaskModel(backbone_name='efficientnet_b0', pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    # if saved from lightning checkpoint, extract state_dict
    if 'state_dict' in state:
        sd = state['state_dict']
        # adjust keys if lightning adds 'model.' prefix
        new_sd = {}
        for k,v in sd.items():
            if k.startswith('model.'):
                new_sd[k.replace('model.','')] = v
            else:
                new_sd[k] = v
        model.load_state_dict(new_sd, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def infer_on_loader(model, loader, device='cpu'):
    preds = []
    labels = []
    image_ids = []
    with torch.no_grad():
        for batch in tqdm(loader):
            imgs = batch['image'].to(device)
            y = batch['label'].numpy()
            ids = batch['image_id']
            dr_logits, iqa_logits = model(imgs)
            probs = torch.sigmoid(dr_logits).cpu().numpy()
            preds.extend(probs.tolist())
            labels.extend(y.tolist())
            image_ids.extend(ids)
    return labels, preds, image_ids

def evaluate(cfg_path='config.yaml', ckpt=None):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() and cfg['training']['gpus']>0 else 'cpu'
    # load messidor csv
    mess_csv = os.path.join(cfg['data']['messidor_dir'], 'labels.csv')
    df_m = pd.read_csv(mess_csv)
    ds = FundusDataset(df_m, img_dir=cfg['data']['messidor_dir'], transforms=get_transforms(cfg['data']['image_size'], train=False), image_size=cfg['data']['image_size'])
    loader = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])
    model = load_model_ckpt(ckpt if ckpt else cfg['logging']['model_ckpt'], device=device)
    labels, preds, image_ids = infer_on_loader(model, loader, device=device)
    metrics = compute_metrics(labels, preds)
    print("External test metrics:", metrics)
    # bootstrap CI for sensitivity and specificity by thresholding at 0.5
    def sens(y_true, y_scores):
        import numpy as np
        y_pred = (np.array(y_scores)>=0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn) if (tp+fn)>0 else 0.0
    def spec(y_true, y_scores):
        import numpy as np
        y_pred = (np.array(y_scores)>=0.5).astype(int)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp) if (tn+fp)>0 else 0.0
    sens_ci = bootstrap_ci(sens, labels, preds, iters=2000)
    spec_ci = bootstrap_ci(spec, labels, preds, iters=2000)
    print(f"Sensitivity CI (95%): {sens_ci}")
    print(f"Specificity CI (95%): {spec_ci}")
    return labels, preds, image_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config.yaml")
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    evaluate(args.cfg, args.ckpt)
