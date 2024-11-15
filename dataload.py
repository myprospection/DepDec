import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import scipy.misc as m
from skimage.transform import resize
import glob
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler


#数据
class Satellitedata(data.Dataset):
    def __init__(self,imgs_path,labels_path,transform):
        self.imgs=imgs_path
        self.labels=labels_path
        self.transform=transform

    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]   // 输出他的图片名
        pil_img=Image.open(img)
        pil_label=Image.open(label)

        img=self.transform(pil_img)
        label=np.array(pil_label)/100
        label=label.astype(np.int64)
        label=torch.from_numpy(label)
        label=label-1

        return img,label

    def __len__(self):
        return len(self.imgs)


class Yxdata(data.Dataset):
    def __init__(self,imgs_path,labels_path,transform):
        self.imgs=imgs_path
        self.labels=labels_path
        self.transform=transform
        self.img_size=(160,240)

    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]

        pil_img=Image.open(img)
        pil_label=Image.open(label)

        img=self.transform(pil_img)
        label=np.array(pil_label)
        label[label==4]=1
        label=label.astype(float)
        label = resize(label,output_shape=(self.img_size[0], self.img_size[1]))
        label=label.astype(np.int64)
        label=torch.from_numpy(label)

        return img,label

    def __len__(self):
        return len(self.imgs)

#数据
class Citydata(data.Dataset):
    def __init__(self,imgs_path,labels_path,transform):
        self.imgs=imgs_path
        self.labels=labels_path
        self.transform=transform

    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]

        pil_img=Image.open(img)
        pil_label=Image.open(label)

        img=self.transform(pil_img)
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
        ])
        label=test_transform(pil_label)
        label=np.array(label)
        label=torch.from_numpy(label).long()

        return img,label

    def __len__(self):
        return len(self.imgs)

#无标签数据
class Unlabeleddata(data.Dataset):
    def __init__(self,imgs_path,transform):
        self.imgs=imgs_path
        self.transform=transform
    def __getitem__(self, index):
        img=self.imgs[index]
        pil_img=Image.open(img)
        img=self.transform(pil_img)
        return img

    def __len__(self):
        return len(self.imgs)
