import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import SimpleITK as sitk
import numpy as np
from CenterLabelHeatMap import *
from Semdataset import *
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
# transformer=transforms.Compose([
#     transforms.Resize((80, 80)),
# ])#缩小8倍

images = os.listdir('E:/fei EZF CT/3Dpoint regress/code/datasem')
image_list = []

for i in images:
    file_path = 'E:/fei EZF CT/3Dpoint regress/code/datasem/{}'.format(i)

    image_list.append(file_path)


class MyDataset(Dataset):

    def __init__(self, image_list, root):

        self.image_list = image_list
        #self.transformer = transformer
        f=open(root,'r')
        self.label=f.readlines()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        #print(index)

        def resize_image_itk(itkimage, newSize=(160, 160, 132), resamplemethod=sitk.sitkNearestNeighbor):
            resampler = sitk.ResampleImageFilter()
            originSize = itkimage.GetSize()  # 原来的体素块尺寸
            originSpacing = itkimage.GetSpacing()
            newSize = np.array(newSize, float)
            factor = originSize / newSize
            newSpacing = originSpacing * factor
            #print(newSpacing)
            newSize = newSize.astype(np.int_)  # spacing肯定不能是整数
            resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
            resampler.SetSize(newSize.tolist())
            resampler.SetOutputSpacing(newSpacing.tolist())
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(resamplemethod)
            itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像

            return itkimgResampled

        image = self.image_list[index]


        image_ct = sitk.ReadImage(image, sitk.sitkInt16)
        image_ct = resize_image_itk(image_ct)
        ct_array = sitk.GetArrayFromImage(image_ct)#[0:527].

        def uniform(data):#归一化image
            max_value, min_value = np.max(data), np.min(data)
            # 根据最大最小值进行归一化放缩到0-1
            data = (data - min_value) / (max_value - min_value)
            return data

        ct_array = uniform(ct_array)

        #print(image_list[index])
        # print(ct_array.shape)

        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)


        #ct_array = self.transformer(ct_array)
        if index < 24:
            labeldata = self.label[index]
            #print(labeldata)

            points = labeldata.split(' ')[1:]

            points = [int(i.rstrip("\n")) * 0.25 for i in points]  # 将数字或字符串转化为列表

            label = []
            for i in range(0, len(points), 3):  # 2是步
                heatmap = CenterLabelHeatMap(160, 160, 132, points[i], points[i + 1], points[i + 2], 8)
                label.append(heatmap)

            label = np.stack(label)
            label = label.transpose(0, 3, 1, 2)
            label = torch.Tensor(label)

        else:
            label = np.zeros((7, 132, 160, 160))#点数18
            label = torch.Tensor(label)


        return ct_array,label


if __name__ == '__main__':

    # data = MyDataset(image_list, 'E:/fei EZF CT/3Dpoint regress/code/semi/data.txt')
    #
    # for i in data:
    #     print(i[0].shape)
    #     print(i[1].shape)

    images = os.listdir('E:/fei EZF CT/3Dpoint regress/code/datasem')
    image_list = []

    for i in images:
        file_path = 'E:/fei EZF CT/3Dpoint regress/code/datasem/{}'.format(i)
        # print(file_path)
        image_list.append(file_path)
    print(len(image_list))

    labelimage_list = list(range(24))  # (10-20)
    unlabelimage_list = list(range(24, 34))

    print(unlabelimage_list)

    batch_sampler = TwoStreamBatchSampler(labelimage_list, unlabelimage_list, 3, 2)

    dataset = MyDataset(image_list, 'E:/fei EZF CT/3Dpoint regress/code/semi/data10.txt')

    # train_ds = monai.data.Dataset(dataset, transform= transfoems)

    data_loader = DataLoader(dataset, batch_sampler=batch_sampler)

    labeled_bs = 1
    batch_sampler = TwoStreamBatchSampler(labelimage_list, unlabelimage_list, 3, 2)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler)

    for i in range(10):
        print(i)
        for i_batch, sampled_batch in enumerate(train_loader):
            # print('fetch data cost {}'.format(time2-time1))
            # volume_batch, label_batch = sampled_batch
            volume_batch, label_batch = sampled_batch
            print(volume_batch.shape, label_batch.shape)
            unlabeled_volume_batch = volume_batch[labeled_bs:]  # 后面的
            labeled_volume_batch = volume_batch[:labeled_bs]
