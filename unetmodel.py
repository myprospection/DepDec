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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from unetplusplus import get_unetplusplus
from apex import amp
import numpy as np
from dataload import Yxdata
# 定义数据转换
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
train_ce_loss = []  # 初始化交叉熵损失列表

model_path = 'unetmodel.pth'


class MyDataLoader(data.Dataset):
    def __init__(self, img_paths, label_paths, transform):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        label = Image.open(self.label_paths[index])
        if self.transform:
            img = self.transform(img)
            label = self.transform(label)
        return img, label


def soft_dice_loss(y_pred, y_true, epsilon=1e-5):
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = y_pred[:, 1, :, :]
    numerator = 2. * torch.sum(y_pred * y_true)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true))
    return 1 - torch.mean(numerator / (denominator + epsilon))


def calculate_cross_entropy(y_pred, y_true):
    y_true = y_true.squeeze(1).long()  # 移除单维度并转换为长整型
    return F.cross_entropy(y_pred, y_true)


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

    # 匹配图片和标签
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
    # all_imgs_path=all_imgs_path[:int(len(all_imgs_path)*0.1)]
    # all_labels_path=all_labels_path[:int(len(all_labels_path)*0.1)]
    # s = int(len(all_imgs_path) * 0.95)
    # train_imgs = all_imgs_path[:s]
    # train_labels = all_labels_path[:s]
    # test_imgs = all_imgs_path[s:]
    # test_labels = all_labels_path[s:]

    train_ds = Yxdata(train_imgs, train_labels, transform)
    test_ds = Yxdata(test_imgs, test_labels, transform)
    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(test_ds, batch_size=batch_size)
    best_score = 0
    best_epoch = 0

    segmentation_model = get_unetplusplus(num_classes)
    segmentation_model.to('cuda')

    optimizer_segmentation = torch.optim.Adam(segmentation_model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_segmentation, step_size=20, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    segmentation_model, optimizer_segmentation = amp.initialize(segmentation_model, optimizer_segmentation,
                                                                opt_level='O0')

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_ce_loss = 0  # 初始化每个epoch的交叉熵损失
        total = 0
        acc = 0
        meaniou = 0
        start_epoch = datetime.datetime.now()
        segmentation_model.train()
        for i, (x, y) in enumerate(train_dl):
            x = x.to('cuda')
            y = y.to('cuda')
            start = datetime.datetime.now()

            segmentation_result = segmentation_model(x)

            # 计算 Soft Dice 损失
            loss = soft_dice_loss(segmentation_result, y)
            epoch_loss += loss.item()

            # 计算交叉熵损失
            ce_loss = calculate_cross_entropy(segmentation_result, y)
            epoch_ce_loss += ce_loss.item()  # 累加交叉熵损失

            optimizer_segmentation.zero_grad()
            with amp.scale_loss(loss, optimizer_segmentation) as scaled_loss:
                scaled_loss.backward()

            optimizer_segmentation.step()

            segmentation_result = torch.argmax(segmentation_result, dim=1)

            for j in range(len(segmentation_result)):
                acc += get_accuracy(segmentation_result[j], y[j])
                meaniou += get_meaniou(segmentation_result[j] + 1, y[j] + 1)
                total += 1

            end = datetime.datetime.now()
            running_time = (end - start) * ((len(train_imgs) // batch_size) - i - 1)
            running_time = str(running_time)[:7]
            p = (total / len(train_imgs)) * 100
            show_str = ('[%%-%ds]' % 100) % (int(100 * p / 100) * ">")
            print(
                '\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss: %.4f, CE Loss: %.4f, Acc: %.4f, MeanIou: %.4f' % (
                    epoch + 1, epochs, total, len(train_imgs), show_str, p, running_time,
                    epoch_loss / (total // batch_size), epoch_ce_loss / (total // batch_size), acc / total,
                    meaniou / total), end='')

        acc = acc / total
        meaniou = meaniou / total
        epoch_loss = epoch_loss / (total // batch_size)
        epoch_ce_loss = epoch_ce_loss / (total // batch_size)  # 计算平均交叉熵损失
        end_epoch = datetime.datetime.now()
        running_time = end_epoch - start_epoch
        running_time = str(running_time)[:7]
        p = (total / len(train_imgs)) * 100
        show_str = ('[%%-%ds]' % 100) % (int(100 * p / 100) * ">")
        print(
            '\rEpoch [%d/%d] %d/%d %s %d%% Running time: %s [Training] Loss: %.4f, CE Loss: %.4f, Acc: %.4f, MeanIou: %.4f' % (
                epoch + 1, epochs, total, len(train_imgs), show_str, p, running_time, epoch_loss, epoch_ce_loss, acc,
                meaniou), end='')



        train_loss.append(epoch_loss)
        train_acc.append(acc)
        train_meaniou.append(meaniou)
        train_ce_loss.append(epoch_ce_loss)
        exp_lr_scheduler.step()

        # 根据需要更新最佳模型
        current_score = acc  # 可以根据具体需求调整评估指标
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save(segmentation_model.state_dict(), model_path)

    print('best_epoch:' + str(best_epoch) + '   ' + 'best_score:' + str(best_score))
    with open('unetresult/train/train_loss.txt', 'w') as f:
        for loss in train_loss:
            f.write(str(loss) + '\n')
    # 保存交叉熵损失到文件
    with open('unetresult/train/train_ce_loss.txt', 'a') as f:
        for ce_loss in train_ce_loss:
            f.write(str(ce_loss) + '\n')
    with open('unetresult/train/train_acc.txt', 'w') as f:
        for acc in train_acc:
            f.write(str(acc) + '\n')
    with open('unetresult/train/train_meaniou.txt', 'w') as f:
        for meaniou in train_meaniou:
            f.write(str(meaniou) + '\n')

    print("Training completed and results saved.")

if __name__ == '__main__':
    main()
