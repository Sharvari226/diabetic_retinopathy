# src/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np

class FundusDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None, image_size=512, synthetic_iqa=True):
        """
        df: pandas DataFrame with columns: 'image_id', 'label' (0-4 ideally)
        label mapping to referable: label >=2 -> 1 else 0
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self.image_size = image_size
        self.synthetic_iqa = synthetic_iqa

    def __len__(self):
        return len(self.df)

    def read_image(self, image_id):
      for ext in (".png", ".jpg", ".jpeg", ".tif"):
        p = os.path.join(self.img_dir, image_id + ext)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ⚠️ Instead of crashing, print a warning and return a blank image
      print(f"[Warning] Image {image_id} not found in {self.img_dir}. Skipping.")
      return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)


    def compute_synthetic_iqa(self, img):
        # simple heuristics:
        # 1) blur: variance of laplacian
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # 2) brightness: mean intensity
        bright = gray.mean()
        # heuristics thresholds (tune as necessary)
        if blur_var < 80 or bright < 20 or bright > 240:
            return 0  # unusable
        return 1  # usable

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # ✅ Handle different dataset column names safely
        if 'image_id' in row:
            image_id = str(row['image_id'])
        elif 'id_code' in row:
            image_id = str(row['id_code'])
        elif 'image' in row:
            image_id = str(row['image'])
        else:
            raise KeyError("No valid image ID column found (expected one of ['image_id', 'id_code', 'image']).")

        img = self.read_image(image_id)

        # crop borders
        from src.utils import autocrop_background
        img = autocrop_background(img)

        if self.synthetic_iqa:
            iqa_label = self.compute_synthetic_iqa(img)
        else:
            iqa_label = 1

        # referable label
        label_raw = int(row.get('label', 0))
        referable = 1 if label_raw >= 2 else 0

        # apply transforms
        if self.transforms:
            augmented = self.transforms(image=img)
            img_t = augmented['image']
        else:
            # basic resize + ToTensor
            import torchvision.transforms as T
            t = T.Compose([
                T.ToPILImage(),
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor()
            ])
            img_t = t(img)

        return {
            'image': img_t,
            'label': referable,
            'iqa': int(iqa_label),
            'image_id': image_id
        }

