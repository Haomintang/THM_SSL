import os
from torchvision import transforms
from torch import nn,optim
import torch
from dataset import MyDataset
from datasetoftest import MytestDataset
from net import *
from torch.utils.data import DataLoader
from network import VNet3d
from awingloss import *
import monai
from monai.transforms import Compose, RandHistogramShiftD, Flipd, Rotate90d
import numpy as np
import math
from torch.utils.data.sampler import Sampler
import itertools

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        # 有标签的索引
        self.primary_indices = primary_indices
        # 无标签的索引
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        # 随机打乱索引顺序
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    # print('shuffle labeled_idxs')
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    # print('shuffle unlabeled_idxs')
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def create_model(ema=False):
    # Network definition
    net = VNet3d(1, 18, 24)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()  # 切断反向传播
    return model



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def distance(p0, p1, digits=3):
    a = map(lambda x: (x[0] - x[1]) ** 2, zip(p0, p1))
    return round(math.sqrt(sum(a)), digits)

def D3_point(a,b):

    o = b[0] + a[0]
    p = b[1] + b[1]
    q = b[2] + b[2]
    n = (o,p,q)
    return n

def cut(obj, sec):
    return [obj[i:i+sec] for i in range(0,len(obj),sec)]

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = create_model().to(device)
    ema_model = create_model(ema=True).to(device)

    loss_val = []

    error_val = []

    epochs = 100

    opt = optim.AdamW(model.parameters(), lr=0.0001)

    testimages = os.listdir('E:/fei EZF CT/3Dpoint regress/code/semi/semdataoftest/images')
    test_image_list = []

    for i in testimages:
        file_path = 'E:/fei EZF CT/3Dpoint regress/code/semi/semdataoftest/images/{}'.format(i)
        # print(file_path)
        test_image_list.append(file_path)

    images = os.listdir('E:/fei EZF CT/3Dpoint regress/code/datasem')
    image_list = []

    for i in images:
        file_path = 'E:/fei EZF CT/3Dpoint regress/code/datasem/{}'.format(i)
        # print(file_path)
        image_list.append(file_path)

    labelimage_list = list(range(24))  # (10-20)
    unlabelimage_list = list(range(24,120))
    print(unlabelimage_list)

    batch_sampler = TwoStreamBatchSampler(labelimage_list, unlabelimage_list, 3,2)

    dataset = MyDataset(image_list,'E:/fei EZF CT/3Dpoint regress/code/semi/data18.txt')
    data_loader=DataLoader(dataset, batch_sampler=batch_sampler)

    testdataset = MytestDataset(test_image_list, 'E:/fei EZF CT/3Dpoint regress/code/semi/datasemi.txt')
    test_data_loader = DataLoader(testdataset, batch_size=2, shuffle=True)

    iter_num = 0
    for epoch in range(epochs):
        for i,(volume_batch,label_batch) in enumerate(test_data_loader):
            print(volume_batch.shape)

            volume_batch,label_batch=volume_batch.to(device),label_batch.to(device)

            unlabeled_volume_batch = volume_batch[1:]  # 无标签是后面的
            print(unlabeled_volume_batch.shape)
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)  # 加入噪声
            ema_inputs = unlabeled_volume_batch + noise#无标签
            outputs = model(volume_batch)

            with torch.no_grad():
                ema_output = ema_model(ema_inputs)  # 加噪结果

            #supervised_train_loss = loss_fun(outputs[:1], label_batch[:1])
            train_loss = AWing(outputs[:1], label_batch[:1])
            supervised_train_loss = train_loss.mean()#监督

            volume_batch_r1 =  unlabeled_volume_batch[:1, :, :, :, :, ]
            volume_batch_r1 = volume_batch_r1.repeat(6, 1, 1, 1, 1)#6ci
            print(volume_batch_r1.shape)

            list = []
            for i in range(6):

                volume_batchR = volume_batch_r1[i:i+1, :, :, :, :, ]
                ema_inputs = volume_batchR + torch.clamp(torch.randn_like(volume_batchR) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    ema_inputsR = ema_model(ema_inputs)

                    ema_inputsR = ema_inputsR.detach().cpu().numpy()
                    ema_inputsR = ema_inputsR.squeeze()

                    for i in range(18):  # 点数
                        d, w, h = np.where(ema_inputsR[i] == ema_inputsR[i].max())
                        list.append([d[0],w[0],h[0]])

            list1 = cut([i for i in list], 7)#14duan

            lists_dict = {}
            for i in range(18):
                lists_dict[i] = []  # 4段
            c = 0

            for i in list1:
                for a in range(18):#3点

                    lists_dict[a].append(i[a])
            dis = 0
            var = 0
            for p in range(18):#4段
                avar = np.var(lists_dict[p], axis=0)

                var = np.sum(avar) + var
            var = var / (18*160)
            v = np.log(var)/2.51
            print(v)
            consistency_loss = torch.mean((outputs[1:] - ema_output) ** 2) + v


            print('supervised_train_loss',supervised_train_loss)

            print('consistency_loss', consistency_loss)
            train_loss = supervised_train_loss + consistency_loss

            print(train_loss)

            print(f'train_loss:{train_loss.item()}')

            opt.zero_grad()#清空梯度
            train_loss.backward()
            opt.step()#更新
            update_ema_variables(model, ema_model, 0.99, iter_num)  # 指数滑动平均给教师模型更新EMA平滑
            iter_num = iter_num + 1

        if epoch%2==0:
            torch.save(model.state_dict(),'E:/fei EZF CT/3Dpoint regress/code/semi/net.pth')

            print('save successfully')

        with torch.no_grad():
            n = 0
            m = 0
            valloss = 0
            valerror = 0
            for i, (image_batch, label_batch) in enumerate(data_loader):

                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                label = label_batch[:1]#有标签
                out = model(image_batch[:1])

                val_loss = AWing(out, label)

                val_loss = torch.mean(val_loss)
                val_loss = val_loss.item()

                valloss = val_loss + valloss

                n = n + 1
                #error_val.append(val_loss)
                zhongwu = 0

                for i in range(1):
                    i = 1 + i
                    out = out[:i, :, :, :, :, ]
                    #print(out.shape)
                    label = label[:i, :, :, :, :, ]

                    out = out.detach().cpu().numpy()
                    out = out.squeeze()

                    label = label.detach().cpu().numpy()
                    label = label.squeeze()

                    for i in range(18):#点数


                        d, w, h = np.where(out[i] == out[i].max())
                        ld, lw, lh = np.where(label[i] == label[i].max())

                        p0 = (d[0], w[0], h[0])
                        p1 = (ld[0], lw[0], lh[0])
                        # print(h, w, d)
                        # print(lh[0], lw[0], ld[0])
                        wucha = distance(p0, p1)
                        zhongwu = zhongwu + wucha


                val_error = zhongwu / 18

                m = m + 1
                valerror = val_error + valerror
            print(n)
            print(m)
            val_loss = valloss / n
            print(val_loss)
            val_error = valerror / m


            loss_val.append(val_loss)
            print('val_loss', val_loss)
            print('val_error', val_error)
            error_val.append(val_error)

            with open("E:/fei EZF CT/3Dpoint regress/code/semi/val_loss10mean.txt", 'w') as val_los:
                val_los.write(str(loss_val))
            with open("E:/fei EZF CT/3Dpoint regress/code/semi/val_error10mean.txt", 'w') as val_error:
                val_error.write(str(error_val))
            # with open("E:/fei EZF CT/3Dpoint regress/code/semi/val_loss7our.txt", 'w') as val_los:
            #     val_los.write(str(loss_val))
            # with open("E:/fei EZF CT/3Dpoint regress/code/semi/val_error7our.txt", 'w') as val_error:
            #     val_error.write(str(error_val))


