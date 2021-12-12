import os
from albumentations.augmentations.transforms import HueSaturationValue, RGBShift
import cv2
import glob
import torch
import numpy as np
import albumentations as albu
from natsort import natsorted
from albumentations.pytorch.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class LG_Train_dataset(Dataset):
    def __init__(self, img_folder, target_folder, transform=None):
        self.images = natsorted(glob.glob(img_folder + '/*'))
        self.targets = natsorted(glob.glob(target_folder + '/*'))
        self.transform = transform

        print(f'Train Dataset size:{len(self.images)}')

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx]).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        target = cv2.imread(self.targets[idx]).astype(np.float32)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB) / 255.0


        if self.transform is not None:
            augmented = self.transform(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']

        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(target.transpose(2, 0, 1))

    def __len__(self):
        return len(self.images)


class LG_Test_dataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.images = natsorted(glob.glob(img_folder + '/*'))
        self.transform = transform

    def __getitem__(self, idx):
        img_name = os.path.basename(self.images[idx])
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']

        return torch.from_numpy(image.transpose(2, 0, 1)), img_name

    def __len__(self):
        return len(self.images)


def get_loader(img_folder, tar_folder, phase: str, batch_size, shuffle,
               num_workers, transform, seed=None):
    if phase == 'test':
        dataset = LG_Test_dataset(img_folder, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = LG_Train_dataset(img_folder, tar_folder, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                 drop_last=False)
    return data_loader

def get_train_augmentation(img_size, ver):
    '''
    ver1 (Base): Only Resize
    ver2 (Normal): HorizontalFlip, RandomRotate90, Resize
    ver3 (Hard): HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, ColorJitter, Resize
    '''
    
    if ver==1:
        transforms = albu.Compose([
                albu.Resize(img_size, img_size, always_apply=True),
                ])

    if ver==2:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.RandomRotate90(),
                ], p=0.5),
                albu.Resize(img_size, img_size, always_apply=True),
            ])

    if ver==3:
        transforms = albu.Compose([
            albu.OneOf([
                albu.HorizontalFlip(),
                albu.VerticalFlip(),
                albu.RandomRotate90(),
                albu.ShiftScaleRotate(),
                ], p=0.5),
            albu.ColorJitter(p=0.3),
            albu.Resize(img_size, img_size, always_apply=True),
            ])

    return transforms


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
    ])
    return transforms



# img_size=128
# transforms = albu.Compose([
#     albu.OneOf([
#         albu.HorizontalFlip(),
#         albu.VerticalFlip(),
#         albu.RandomRotate90(),
#         albu.ShiftScaleRotate(),
#         ], p=0.5),
#     albu.ColorJitter(p=0.3),
#     albu.RGBShift(p=0.3),
#     albu.Resize(img_size, img_size, always_apply=True),
#     ],
#     additional_targets={"image1": "mask",})


# x = img2
# y = img2

# aug = transforms(image=x, mask=y)

# print(aug['image'].shape, aug['mask'].shape)

# import matplotlib.pyplot as plt
# fig,ax= plt.subplots(1,2)
# ax[0].imshow(aug['image'])
# ax[1].imshow(aug['mask'])
