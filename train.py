import os
import yaml
from argparse import ArgumentParser
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.dataset import FundusDataset
from src.transforms import get_transforms
from src.model import DRMultiTaskModel
from src.utils import seed_everything
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import BinaryAUROC
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        bce_loss = self.bce(logits, targets.float())
        pt = probs * targets + (1 - probs) * (1 - targets)
        loss = (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class LitDR(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DRMultiTaskModel(backbone_name='efficientnet_b0', pretrained=True)
        self.alpha = cfg['training']['alpha_iqa']
        self.lr = cfg['training']['lr']
        self.weight_decay = cfg['training']['weight_decay']
        self.focal = FocalLoss(gamma=cfg['training']['focal_gamma'])
        self.ce = nn.CrossEntropyLoss()
        self.auroc = BinaryAUROC()
        self.save_hyperparameters(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs = batch['image']
        y = batch['label'].float()
        y_iqa = batch['iqa'].long()
        dr_logits, iqa_logits = self(imgs)
        l_dr = self.focal(dr_logits, y)
        l_iqa = self.ce(iqa_logits, y_iqa)
        loss = l_dr + self.alpha * l_iqa
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch['image']
        y = batch['label'].float()
        y_iqa = batch['iqa'].long()
        dr_logits, iqa_logits = self(imgs)
        l_dr = self.focal(dr_logits, y)
        l_iqa = self.ce(iqa_logits, y_iqa)
        loss = l_dr + self.alpha * l_iqa

        probs = torch.sigmoid(dr_logits)
        self.log('val/loss', loss, prog_bar=True, on_epoch=True)
        self.log('val/auroc', self.auroc(probs, y.int()), prog_bar=True, on_epoch=True)

        # Store outputs for epoch-end computation
        if not hasattr(self, "_validation_outputs"):
            self._validation_outputs = []
        self._validation_outputs.append({
            'probs': probs.detach().cpu().numpy(),
            'labels': y.detach().cpu().numpy()
        })
        return loss

    def on_validation_epoch_end(self):
        if not hasattr(self, "_validation_outputs"):
            return

        probs = np.concatenate([x['probs'] for x in self._validation_outputs])
        labels = np.concatenate([x['labels'] for x in self._validation_outputs])
        from src.utils import compute_metrics
        metrics = compute_metrics(labels, probs, threshold=0.5)

        self.log('val/sensitivity', metrics['sensitivity'], prog_bar=True)
        self.log('val/specificity', metrics['specificity'], prog_bar=True)
        self.log('val/auc_full', metrics['auc'], prog_bar=False)

        # clear cache to avoid memory leaks
        self._validation_outputs.clear()


    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.cfg['training']['max_epochs']), eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/auc_full"}}


def load_df_from_csv(csv_path):
    return pd.read_csv(csv_path)


def main(cfg_path='config.yaml'):
    cfg = yaml.safe_load(open(cfg_path))
    seed_everything(cfg['seed'])

    aptos_csv = os.path.join(cfg['data']['aptos_dir'], 'labels.csv')
    df = load_df_from_csv(aptos_csv)

    from sklearn.model_selection import StratifiedShuffleSplit
    df['referable'] = (df['label'] >= 2).astype(int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=cfg['seed'])
    train_idx, val_idx = next(sss.split(df, df['referable']))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_ds = FundusDataset(df_train, img_dir=cfg['data']['aptos_dir'], transforms=get_transforms(cfg['data']['image_size'], train=True), image_size=cfg['data']['image_size'])
    val_ds = FundusDataset(df_val, img_dir=cfg['data']['aptos_dir'], transforms=get_transforms(cfg['data']['image_size'], train=False), image_size=cfg['data']['image_size'])

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'])

    model = LitDR(cfg)
    ckpt_cb = ModelCheckpoint(dirpath=os.path.dirname(cfg['logging']['model_ckpt']), filename='best', monitor='val/auroc', mode='max', save_top_k=1)
    early_stop = EarlyStopping(monitor='val/auroc', patience=cfg['training']['early_stop_patience'], mode='max')

    # Safe device configuration
    use_gpu = cfg['training']['gpus'] > 0 and torch.cuda.is_available()
    accelerator = 'gpu' if use_gpu else 'cpu'
    devices = cfg['training']['gpus'] if use_gpu else max(1, cfg['training'].get('devices', 1))

    print(f"Using accelerator: {accelerator}, devices: {devices}")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=cfg['training']['precision'],
        max_epochs=cfg['training']['max_epochs'],
        accumulate_grad_batches=cfg['training']['accumulate_grad_batches'],
        callbacks=[ckpt_cb, early_stop],
        enable_model_summary=False
    )

    trainer.fit(model, train_loader, val_loader)
    print("Training finished. Best checkpoint:", ckpt_cb.best_model_path)


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    main(cfg)
