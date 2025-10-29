# src/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(image_size=512, train=True):
    if train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(),
            A.Rotate(limit=15),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=0, p=0.5),
            A.OneOf([A.CLAHE(clip_limit=2), A.Equalize(), A.RandomGamma()], p=0.5),
            A.CoarseDropout(max_holes=1, max_height=int(0.08*image_size), max_width=int(0.08*image_size), p=0.3),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(),
            ToTensorV2()
        ])
