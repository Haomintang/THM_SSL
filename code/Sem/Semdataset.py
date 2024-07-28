import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL
import itertools
import numpy as np
from torch.utils.data.sampler import Sampler
import os
from tqdm import tqdm


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


class D3VnetData(Dataset):

    def __init__(self, image_list, label_list):

        self.image_list = image_list
        self.label_list = label_list
        self.sample_list = []
        #self.image_list = [item.strip() for item in self.image_list]


    def __getitem__(self, index):
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
        ct_array = sitk.GetArrayFromImage(image_ct)

        ct_array = ct_array[:, ::-1, :]

        def uniform(data):#归一化image
            max_value, min_value = np.max(data), np.min(data)
            # 根据最大最小值进行归一化放缩到0-1
            data = (data - min_value) / (max_value - min_value)
            return data
        ct_array = uniform(ct_array)
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)  ###[bz, 1, 50, 512, 512]

        #print(ct_array.shape)
        if index < 10:
            label = self.label_list[index]
            label_ct = sitk.ReadImage(label, sitk.sitkInt8)
            label_ct = resize_image_itk(label_ct)
            label_array = sitk.GetArrayFromImage(label_ct)
            label_array[label_array > 0] = 1
            label_array = torch.LongTensor(label_array)  ###[50, 512, 512]


        else:
            t = np.zeros((132, 160, 160))
            label_array = torch.tensor(t)


        return ct_array, label_array


    def __len__(self):

        return len(self.image_list)

class  D3VnetData_test(Dataset):

    def __init__(self, image_list, label_list):

        self.image_list = image_list
        self.label_list = label_list



    def __getitem__(self, index):
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
        ct_array = sitk.GetArrayFromImage(image_ct)

        ct_array = ct_array[:, ::-1, :]

        def uniform(data):#归一化image
            max_value, min_value = np.max(data), np.min(data)
            # 根据最大最小值进行归一化放缩到0-1
            data = (data - min_value) / (max_value - min_value)
            return data
        ct_array = uniform(ct_array)
        ct_array = ct_array.astype(np.float32)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)  ###[bz, 1, 50, 512, 512]

        #print(ct_array.shape)

        label = self.label_list[index]
        label_ct = sitk.ReadImage(label, sitk.sitkInt8)
        label_ct = resize_image_itk(label_ct)
        label_array = sitk.GetArrayFromImage(label_ct)
        label_array[label_array > 0] = 1
        label_array = torch.LongTensor(label_array)  ###[50, 512, 512]

        return ct_array, label_array


    def __len__(self):

        return len(self.image_list)


# if __name__ == '__main__':
    images = os.listdir('E:/fei EZF CT/3DUnet\dataset_shang/train/images')
    labels = os.listdir('E:/fei EZF CT/3DUnet\dataset_shang/train/labels')

    image_list = []
    label_list = []

    for i in images:
        file_path = 'E:/fei EZF CT/3DUnet\dataset_shang/train/images/{}'.format(i)
        # print(file_path)
        image_list.append(file_path)

    for i in labels:
        file_path = 'E:/fei EZF CT/3DUnet\dataset_shang/train/labels/{}'.format(i)
        # print(file_path)
        label_list.append(file_path)
#
#     print(len(image_list))
#
#     #train_set = D3VnetData_test(image_list, label_list)
#     # labelimage_list = list(range(10))  # (10-20)
#     # unlabelimage_list = list(range(10, 40))
#     #
#     # labeled_bs = 1
#
#     test_ds = D3VnetData_test(image_list, label_list)  # 看dataset里写的image_list
#     testloader = DataLoader(test_ds, batch_size=2,shuffle=False)
#
#     for i in range(10):
#         print(i)
#         for i_batch, sampled_batch in enumerate(testloader):
#             # print('fetch data cost {}'.format(time2-time1))
#             # volume_batch, label_batch = sampled_batch
#             volume_batch, label_batch = sampled_batch
#             print(volume_batch.shape, label_batch.shape)
#             # unlabeled_volume_batch = volume_batch[labeled_bs:]  # 后面的
#             # labeled_volume_batch = volume_batch[:labeled_bs]






