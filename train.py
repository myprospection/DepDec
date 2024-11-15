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
import datetime
from apex import amp
from  unetplusplus import UNet
from dataload import Satellitedata
from dataload import Citydata
from dataload import Yxdata
from nnmodel import Unet_model
from deeplabv3plus import deeplabv3plus_resnet
from deeplabv3plus import correction_resnet
from unetplusplus import get_unetplusplus

transform = transforms.Compose([
    transforms.Resize((160, 240)),
    transforms.ToTensor()
])

num_classes = 2
batch_size = 4
epochs = 60

train_loss = []
train_acc = []
train_meaniou = []
test_loss = []
test_acc = []
test_meaniou = []
cross_entropy_loss_values = []  # 新增列表保存交叉熵损失

model_path = 'my.pth'


def soft_dice_loss(y_pred, y_true, epsilon=1e-5):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred[:, 1, :, :]
    numerator = 2. * torch.sum(y_pred * y_true)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true))
    return 1 - torch.mean(numerator / (denominator + epsilon))


def calculate_cross_entropy(y_pred, y_true):
    y_true = y_true.long()  # 移除单维度并转换为长整型
    return F.cross_entropy(y_pred, y_true)


def plot_img(imgs, labels):
    plt.figure(figsize=(10, 5 * len(imgs)))
    for i, (img, label) in enumerate(zip(imgs, labels)):
        img = img.permute(1, 2, 0)  # 交换顺序
        img = img.numpy()
        plt.subplot(len(imgs), 2, i * 2 + 1)
        plt.imshow(img)
        plt.subplot(len(imgs), 2, i * 2 + 2)
        plt.imshow(label.numpy())
    plt.show()


def get_accuracy(sr, gt):
    corr = torch.sum(sr == gt)
    tensor_size = sr.size(0) * sr.size(1)
    acc = float(corr) / float(tensor_size)
    return acc


def get_meaniou(segmentation_result, y, n_classes=2):
    iou = []
    iou_sum = 0
    segmentation_result = segmentation_result.view(-1)
    y = y.view(-1)
    classes = torch.unique(y)

    for cls in range(1, n_classes):
        if cls not in classes:
            n_classes -= 1
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
    return iou_sum / n_classes


def main():
    train_imgs = glob.glob('E:\\data\\src\\*\\*.jpg')
    train_labels = glob.glob('E:\\data\\src\\*\\*_label.png')

    mask = []
    for i in train_imgs:
        for j in train_labels:
            if i.split('\\')[-2] == j.split('\\')[-2] and i.split('\\')[-1].split('.')[0] == \
                    j.split('\\')[-1].split('_')[0]:
                mask.append(j)

    train_labels = mask

    test_imgs = glob.glob('E:\\data\\src\\*\\*.jpg')
    test_labels = glob.glob('E:\\data\\src\\*\\*_label.png')

    mask = []
    for i in test_imgs:
        for j in test_labels:
            if i.split('\\')[-2] == j.split('\\')[-2] and i.split('\\')[-1].split('.')[0] == \
                    j.split('\\')[-1].split('_')[0]:
                mask.append(j)

    test_labels = mask

    # 训练数据和测试数据的划分
    index = np.random.permutation(len(train_imgs))
    train_imgs = np.array(train_imgs)[index]
    train_labels = np.array(train_labels)[index]

    train_ds = Yxdata(train_imgs, train_labels, transform)
    test_ds = Yxdata(test_imgs, test_labels, transform)
    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(test_ds, batch_size=batch_size)

    segmentation_model =deeplabv3plus_resnet(num_classes)
    segmentation_model.to('cuda')

    optimizer_segmentation = torch.optim.Adam(segmentation_model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_segmentation, step_size=20, gamma=0.1)

    segmentation_model, optimizer_segmentation = amp.initialize(segmentation_model, optimizer_segmentation,
                                                                opt_level='O0')

    for epoch in range(epochs):
        epoch_loss = 0
        total = 0
        acc = 0
        meaniou = 0
        epoch_ce_loss = 0  # 新增用于计算交叉熵损失的累加器
        start_epoch = datetime.datetime.now()
        segmentation_model.train()

        for i, (x, y) in enumerate(train_dl):
            x = x.to('cuda')
            y = y.to('cuda')

            optimizer_segmentation.zero_grad()
            segmentation_result = segmentation_model(x)

            # 计算 Soft Dice 损失
            loss = soft_dice_loss(segmentation_result, y)

            # 计算交叉熵损失

            ce_loss = calculate_cross_entropy(segmentation_result, y)
            epoch_ce_loss += ce_loss.item()  # 累加当前批次的交叉熵损失


            # 累加损失
            epoch_loss += loss.item()  # 累加当前批次的损失

            # 反向传播
            with amp.scale_loss(loss, optimizer_segmentation) as scaled_loss:
                scaled_loss.backward()

            # 优化器更新
            optimizer_segmentation.step()

            # 计算准确率和 MeanIoU
            segmentation_result = torch.argmax(segmentation_result, dim=1)
            for j in range(len(segmentation_result)):
                acc += get_accuracy(segmentation_result[j], y[j])
                meaniou += get_meaniou(segmentation_result[j] + 1, y[j] + 1)
                total += 1

            # 每个 batch 计算运行时间
            end = datetime.datetime.now()
            running_time = (end - start_epoch) * ((len(train_imgs) // batch_size) - i - 1)
            running_time = str(running_time)[:7]
            p = (total / len(train_imgs)) * 100
            show_str = ('[%%-%ds]' % 100) % (int(100 * p / 100) * ">")
            print('\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss: %.4f, CE Loss: %.4f, Acc: %.4f, MeanIou: %.4f' % (
                epoch + 1, epochs, total, len(train_imgs), show_str, p, running_time,
                epoch_loss / (total // batch_size), epoch_ce_loss / (total // batch_size), acc / total, meaniou / total), end='')

        # 计算平均损失和指标
        acc = acc / total
        meaniou = meaniou / total
        epoch_loss = epoch_loss / (total // batch_size)
        epoch_ce_loss = epoch_ce_loss / (total // batch_size)
        end_epoch = datetime.datetime.now()
        running_time = end_epoch - start_epoch
        running_time = str(running_time)[:7]
        p = (total / len(train_imgs)) * 100
        show_str = ('[%%-%ds]' % 100) % (int(100 * p / 100) * ">")
        print('\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss: %.4f, CE Loss: %.4f, Acc: %.4f, MeanIou: %.4f' % (
            epoch + 1, epochs, total, len(train_imgs), show_str, p, running_time, epoch_loss, epoch_ce_loss, acc, meaniou), end='')

        # 记录训练损失和准确率
        train_loss.append(epoch_loss)
        train_acc.append(acc)
        train_meaniou.append(meaniou)
        cross_entropy_loss_values.append(epoch_ce_loss)
        exp_lr_scheduler.step()

        torch.save(segmentation_model.state_dict(), model_path)

def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
        file = open(filename, 'a')
        for i in range(len(data)):
            s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
            s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
            file.write(s)
        file.close()
        print("保存成功")
    # 保存
if __name__ == '__main__':
    main()
    text_save('result/train/train_loss.txt', train_loss)
    text_save('result/train/train_acc.txt', train_acc)
    text_save('result/train/train_eloss.txt', cross_entropy_loss_values)
    text_save('result/train/train_meaniou.txt', train_meaniou)
    text_save('result/test/test_loss.txt', test_loss)
    text_save('result/test/test_acc.txt', test_acc)
    text_save('result/test/test_meaniou.txt', test_meaniou)