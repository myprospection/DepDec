import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from dataload import Satellitedata
from dataload import Citydata
from dataload import Yxdata
from nnmodel import Unet_model
from deeplabv3plus import deeplabv3plus_resnet
from unetplusplus import get_unetplusplus

model=deeplabv3plus_resnet(2)
model.to('cuda')


model_path = 'deeplab.pth'

def get_accuracy(sr,gt):
    corr=torch.sum(sr==gt)
    tensor_size=sr.size(0)*sr.size(1)
    acc=float(corr)/float(tensor_size)
    return acc


def get_meaniou(segmentation_result, y, n_classes = 34):
    iou = []
    iou_sum = 0
    segmentation_result = segmentation_result.view(-1)
    y = y.view(-1)
    classes=torch.unique(y)
    #print(classes)

    for cls in range(1, n_classes):
        if cls not in classes:
            n_classes-=1
            continue
        result_inds = segmentation_result == cls
        y_inds = y == cls
        intersection = (result_inds[y_inds]).long().sum().data.cpu().item()
        union = result_inds.long().sum().data.cpu().item() + y_inds.long().sum().data.cpu().item() - intersection
        if union == 0:
            iou.append(float('nan'))
        else:
            iou.append(float(intersection) / float(max(union, 1)))
            iou_sum += float(intersection) / float(max(union, 1))
    #print(iou)
    return iou_sum/n_classes


#画图
def plot_img(imgs,labels):
    plt.figure(figsize=(10, 5*len(imgs)))
    for i, (img, label) in enumerate(zip(imgs,labels)):
        img = img.permute(1, 2, 0)  # 交换顺序
        img = img.cpu().numpy()
        plt.subplot(len(imgs), 2, i * 2 + 1)
        plt.imshow(img)
        plt.subplot(len(imgs), 2, i * 2 + 2)
        plt.imshow(label.cpu().numpy())
    plt.show()

def plot_img_gt(imgs,annos,labels,count):
    plt.figure(figsize=(15, 5 * len(imgs)))
    for i, (img, anno,label) in enumerate(zip(imgs, annos,labels)):
        img = img.permute(1, 2, 0)  # 交换顺序
        img = img.cpu().numpy()
        plt.subplot(len(imgs), 3, i * 3 + 1)
        plt.imshow(img)
        plt.subplot(len(imgs), 3, i * 3 + 2)
        plt.imshow(anno.cpu().numpy())
        plt.subplot(len(imgs), 3, i * 3 + 3)
        plt.imshow(label.cpu().numpy())
    plt.savefig('./结果/'+str(count)+'.jpg')
    #plt.show()
    count+=1


def testing():
    model.load_state_dict(torch.load(model_path))  # 加载模型
    all_imgs_path = glob.glob('E:\\data\\src\\*\\*.jpg')
    all_labels_path = glob.glob('E:\\data\\src\\*\\*_label.png')
    mask = []
    for i in all_imgs_path:
        for j in all_labels_path:
            if i.split('\\')[-2] == j.split('\\')[-2] and i.split('\\')[-1].split('.')[0] == \
                    j.split('\\')[-1].split('_')[0]:
                mask.append(j)

    all_labels_path = mask

    transform = transforms.Compose([
        transforms.Resize((160, 240)),
        transforms.ToTensor()
    ])
    dataset=Yxdata(all_imgs_path,all_labels_path,transform)
    model_data = data.DataLoader(dataset, batch_size=2, shuffle=False)

    total=0

    for x,y in model_data:
        x=x.to('cuda')
        y=y.to('cuda')
        segmentation_result=model(x)
        segmentation_result=torch.argmax(segmentation_result,dim=1)
        plot_img_gt(x,y,segmentation_result,total)
        total+=1

    # model.eval()
    # epoch_loss = 0
    # acc = 0
    # meaniou = 0
    # test_total = 0
    # with torch.no_grad():
    #     for i, (x, y) in enumerate(test_dl):
    #         x = x.to('cuda')
    #         y = y.to('cuda')
    #         segmentation_result = model(x)
    #         segmentation_result = torch.argmax(segmentation_result, dim=1)
    #         acc += get_accuracy(segmentation_result, y)
    #         meaniou += get_meaniou(segmentation_result, y)
    #
    #         test_total += 1
    #
    #     epcoh_loss = epoch_loss / test_total
    #     acc = acc / test_total
    #     meaniou = meaniou / test_total
    #
    #     print('[Validation] Acc: %.4f, MeanIou: %.4f' % (acc, meaniou))






if __name__ == '__main__':
    testing()

