import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
import nibabel as nib
from torch.utils.data import Sampler
from collections import defaultdict
import random


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]

        elif self.split == 'val':
            with open(self._base_dir + '/val.list', 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '') for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        sample['case'] = case
        return sample

class AMOS2D(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = os.path.join(base_dir, split)
        self.split = split
        self.transform = transform
        self.slice_infos = []  # (case_id, slice_idx) or just case_id list

        if split == 'train':
            for fname in sorted(os.listdir(self._base_dir)):
                if fname.endswith('_image.npy'):
                    case_id = fname.replace('_image.npy', '')
                    image_path = os.path.join(self._base_dir, f"{case_id}_image.npy")
                    label_path = os.path.join(self._base_dir, f"{case_id}_label.npy")
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        image = np.load(image_path)
                        D = image.shape[0]
                        self.slice_infos.extend([(case_id, i) for i in range(D)])
            if num is not None:
                self.slice_infos = self.slice_infos[:num]
        else:  # val or test
            self._base_dir = os.path.join(base_dir, split)
            for fname in sorted(os.listdir(self._base_dir)):
                if fname.endswith('_image.npy'):
                    case_id = fname.replace('_image.npy', '')
                    image_path = os.path.join(self._base_dir, f"{case_id}_image.npy")
                    label_path = os.path.join(self._base_dir, f"{case_id}_label.npy")
                    if os.path.exists(image_path) and os.path.exists(label_path):
                        self.slice_infos.append(case_id)

        print(f"Loaded {len(self.slice_infos)} samples for {split} split.")

    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, idx):
        if self.split == 'train':
            case_id, slice_idx = self.slice_infos[idx]
            image = np.load(os.path.join(self._base_dir, f"{case_id}_image.npy"))[slice_idx]
            label = np.load(os.path.join(self._base_dir, f"{case_id}_label.npy"))[slice_idx]
            sample = {'image': image, 'label': label}
            if self.split == "train":
                sample = self.transform(sample)
            sample['case'] = case_id
            sample['slice'] = slice_idx
        else:
            case_id = self.slice_infos[idx]
            image = np.load(os.path.join(self._base_dir, f"{case_id}_image.npy"))
            label = np.load(os.path.join(self._base_dir, f"{case_id}_label.npy"))
            sample = {'image': image, 'label': label}
            sample['case'] = case_id
        return sample
class AMOS3D(Dataset):
    def __init__(self, base_dir, split='train', num=None, transform=None):
        self._base_dir = os.path.join(base_dir, split)
        self.split = split
        self.transform = transform
        self.slice_infos = []

        for fname in sorted(os.listdir(self._base_dir)):
            if fname.endswith('_image.npy'):
                cid = fname.replace('_image.npy', '')
                if os.path.exists(os.path.join(self._base_dir, f'{cid}_label.npy')):
                    self.slice_infos.append(cid)
        if num is not None and split == 'train':
            self.slice_infos = self.slice_infos[:num]
        print(f'Loaded {len(self.slice_infos)} samples for {split} split.')

    def __len__(self):
        return len(self.slice_infos)

    @staticmethod
    def _ensure_112_112_80(arr: np.ndarray) -> np.ndarray:
        """
        把任何 (80,112,112) / (112,80,112) / (112,112,80) 统一转到 (112,112,80)
        """
        if arr.shape == (112, 112, 80):
            return arr
        # 找到长度为 80 的那个轴
        axis_80 = np.where(np.array(arr.shape) == 80)[0]
        assert len(axis_80) == 1, f"Unexpected shape {arr.shape},无法确定深度维"
        # 把该维度移到最后
        arr = np.moveaxis(arr, axis_80[0], -1)
        return arr   # 现在一定是 (112,112,80)

    def __getitem__(self, idx):
        cid = self.slice_infos[idx]
        img = np.load(os.path.join(self._base_dir, f'{cid}_image.npy'))
        lbl = np.load(os.path.join(self._base_dir, f'{cid}_label.npy'))

        img = self._ensure_112_112_80(img)
        lbl = self._ensure_112_112_80(lbl)

        sample = {'image': img, 'label': lbl, 'case': cid}
        if self.split == 'train' and self.transform is not None:
            sample = self.transform(sample)
        return sample



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'].squeeze(), sample['label'].squeeze()
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class LAHeart(Dataset):
    """ LA Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = self._base_dir + '/train.list'
        test_path = self._base_dir + '/test.list'

        if split == 'train':
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/2018LA_Seg_Training Set/" + image_name + "/mri_norm2.h5", 'r')
        # h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class Parse2022(Dataset):
    def __init__(self, base_dir, split='train', out_size=256, transform=None, slice_axis=2):
        """
        Args:
            base_dir: 数据集根目录
            split: 数据集划分 (train/val/test)
            out_size: 输出尺寸（正方形）
            transform: 数据增强变换
            slice_axis: 切片轴向 (0:矢状面, 1:冠状面, 2:轴向)
        """
        self.base_dir = base_dir
        self.out_size = out_size
        self.split = split
        self.transform = transform
        self.slice_axis = slice_axis

        # 读取文件列表
        self.sample_list = self._load_sample_list(split)

        # 数据路径
        self.image_dir = os.path.join(self.base_dir, 'image')
        self.label_dir = os.path.join(self.base_dir, 'label')

        # 预加载所有病例并计算总slice数
        self.slices_metadata = self._preload_slices()

    def _load_sample_list(self, split):
        sample_list_path = os.path.join(self.base_dir, f'{split}.txt')
        with open(sample_list_path, 'r') as f:
            sample_list = f.readlines()
        return [item.strip() for item in sample_list]

    def _preload_slices(self):
        """预计算每个case的slice数量和全局索引"""
        slices_meta = []
        global_idx = 0

        for case in self.sample_list:
            img_path = os.path.join(self.image_dir, f'{case}.nii.gz')
            img_data = nib.load(img_path).get_fdata()

            # 获取该病例的slice数量
            num_slices = img_data.shape[self.slice_axis]

            for slice_idx in range(num_slices):
                slices_meta.append({
                    'case': case,
                    'slice_idx': slice_idx,
                    'global_idx': global_idx
                })
                global_idx += 1

        return slices_meta

    def __len__(self):
        return len(self.slices_metadata)

    def __getitem__(self, idx):
        meta = self.slices_metadata[idx]
        case = meta['case']
        slice_idx = meta['slice_idx']

        # 加载原始3D数据
        img_path = os.path.join(self.image_dir, f'{case}.nii.gz')
        label_path = os.path.join(self.label_dir, f'{case}.nii.gz')

        img_3d = nib.load(img_path).get_fdata()
        label_3d = nib.load(label_path).get_fdata()

        # 提取指定slice (处理不同轴向)
        if self.slice_axis == 0:
            img_slice = img_3d[slice_idx, :, :]
            label_slice = label_3d[slice_idx, :, :]
        elif self.slice_axis == 1:
            img_slice = img_3d[:, slice_idx, :]
            label_slice = label_3d[:, slice_idx, :]
        else:  # axial
            img_slice = img_3d[:, :, slice_idx]
            label_slice = label_3d[:, :, slice_idx]

        # 调整尺寸 (2D)
        img_resized = zoom(img_slice, (self.out_size / img_slice.shape[0], self.out_size / img_slice.shape[1]), order=1)
        label_resized = zoom(label_slice, (self.out_size / label_slice.shape[0], self.out_size / label_slice.shape[1]),
                             order=0)

        # 增加通道维度 (C,H,W)
        img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0)  # 1×H×W
        label_tensor = torch.from_numpy(label_resized).long().unsqueeze(0)  # 1×H×W

        sample = {
            'image': img_tensor,
            'label': label_tensor
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        """自定义batch组装方式"""
        images = torch.stack([item['image'] for item in batch], dim=0)
        labels = torch.stack([item['label'] for item in batch], dim=0)
        case_ids = [item['case_id'] for item in batch]
        slice_indices = [item['slice_idx'] for item in batch]

        return {
            'image': images,
            'label': labels,
            'case_id': case_ids,
            'slice_idx': slice_indices
        }


class Parse3D(Dataset):
    """ 3D Parse Dataset from NIfTI files """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        """
        Args:
            base_dir: 数据集根目录
            split: 数据集划分 (train/test)
            num: 限制加载的样本数量
            transform: 数据增强变换
        """
        self.base_dir = base_dir
        self.transform = transform
        self.image_list = []

        # 文件路径配置
        self.image_dir = os.path.join(self.base_dir, 'image')
        self.label_dir = os.path.join(self.base_dir, 'label')
        self.file_suffix = '.nii.gz'  # 根据实际文件命名修改

        # 加载文件列表
        list_path = os.path.join(self.base_dir, f'{split}.txt')
        with open(list_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.image_list = self.image_list[:num]

        print(f"Total {len(self.image_list)} 3D samples loaded")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        case_id = self.image_list[idx]

        # 加载NIfTI文件 (修改为您的实际文件路径结构)
        img_path = os.path.join(self.image_dir, f'{case_id}.nii.gz')
        label_path = os.path.join(self.label_dir, f'{case_id}{self.file_suffix}')

        # 读取3D数据
        img = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.uint8)

        # 转换为torch tensor
        sample = {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(label),
            'case_id': case_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        """自定义3D数据batch组装"""
        images = torch.cat([item['image'].unsqueeze(0) for item in batch], dim=0)
        labels = torch.cat([item['label'].unsqueeze(0) for item in batch], dim=0)
        case_ids = [item['case_id'] for item in batch]

        return {
            'image': images,  # B×C×D×H×W
            'label': labels,  # B×C×D×H×W
            'case_id': case_ids
        }


class ImageCAS3D(Dataset):
    """ 3D LA Heart Dataset from NIfTI files """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        """
        Args:
            base_dir: 数据集根目录
            split: 数据集划分 (train/test)
            num: 限制加载的样本数量
            transform: 数据增强变换
        """
        self.base_dir = base_dir
        self.transform = transform
        self.image_list = []

        # 文件路径配置
        self.file_label = '.label.nii.gz'  # 根据实际文件命名修改
        self.file_img = '.img.nii.gz'  # 根据实际文件命名修改

        # 加载文件列表
        list_path = os.path.join(self.base_dir, f'{split}.txt')
        with open(list_path, 'r') as f:
            self.image_list = [line.strip() for line in f.readlines()]

        if num is not None:
            self.image_list = self.image_list[:num]

        print(f"Total {len(self.image_list)} 3D samples loaded")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        case_id = self.image_list[idx]

        # 加载NIfTI文件 (修改为您的实际文件路径结构)
        img_path = os.path.join(self.base_dir, f'{case_id}{self.file_img}')
        label_path = os.path.join(self.base_dir, f'{case_id}{self.file_label}')

        # 读取3D数据
        img = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(label_path).get_fdata().astype(np.uint8)

        # 转换为torch tensor
        sample = {
            'image': torch.from_numpy(img),
            'label': torch.from_numpy(label),
            'case_id': case_id
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def collate_fn(batch):
        """自定义3D数据batch组装"""
        images = torch.cat([item['image'].unsqueeze(0) for item in batch], dim=0)
        labels = torch.cat([item['label'].unsqueeze(0) for item in batch], dim=0)
        case_ids = [item['case_id'] for item in batch]

        return {
            'image': images,  # B×C×D×H×W
            'label': labels,  # B×C×D×H×W
            'case_id': case_ids
        }


class Resize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        (w, h, d) = image.shape
        label = label.astype(np.bool)
        image = sk_trans.resize(image, self.output_size, order=1, mode='constant', cval=0)
        label = sk_trans.resize(label, self.output_size, order=0)
        assert (np.max(label) == 1 and np.min(label) == 0)
        assert (np.unique(label).shape[0] == 2)

        return {'image': image, 'label': label}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image': image, 'label': label, 'sdf': sdf}
        else:
            return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rot_flip(image, label)

        return {'image': image, 'label': label}


class RandomRot(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = random_rotate(image, label)

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


def case_to_slices(dataset, labeled_case):
        """
        Args:
            dataset: 具有 .slice_infos 属性，格式为 (case_id, slice_idx)
            labeled_case: 有标签 case_id 的 list
        """


        # 构建 labeled / unlabeled 的切片索引列表
        labeled_idxs = []
        unlabeled_idxs = []

        for idx, case_id in enumerate(dataset.slice_infos):
        # for idx, (case_id, _) in enumerate(dataset.slice_infos):
            if case_id in labeled_case:
                labeled_idxs.append(idx)
                # labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
                # unlabeled_idxs.append(idx)

        return labeled_idxs, unlabeled_idxs

import numpy as np
from torch.utils.data import Sampler
from itertools import cycle, islice
# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices
#
#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.
#     """
#
#     def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size, seed=3407):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.secondary_batch_size = secondary_batch_size
#         self.primary_batch_size = batch_size - secondary_batch_size
#
#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0
#
#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in zip(grouper(primary_iter, self.primary_batch_size),
#                    grouper(secondary_iter, self.secondary_batch_size))
#         )
#
#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size

class TwoStreamBatchSampler(Sampler):
    """
    以两个索引集(primary/secondary)组成 batch：
      - 每个 batch: primary_batch_size + secondary_batch_size
      - primary 一轮(epoch)只遍历一次
      - secondary 循环使用，直至对齐 primary 的步数

    可控打乱：shuffle_primary / shuffle_secondary
    可复现：seed + set_epoch
    """

    def __init__(self,
                 primary_indices,
                 secondary_indices,
                 batch_size,
                 secondary_batch_size,
                 seed=3407,
                 shuffle_primary=True,
                 shuffle_secondary=True,
                 drop_last=True):
        self.primary_indices   = list(primary_indices)
        self.secondary_indices = list(secondary_indices)
        self.secondary_batch_size = int(secondary_batch_size)
        self.primary_batch_size   = int(batch_size - secondary_batch_size)
        assert len(self.primary_indices)   >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

        self.shuffle_primary   = bool(shuffle_primary)
        self.shuffle_secondary = bool(shuffle_secondary)
        self.drop_last         = bool(drop_last)

        self._base_seed = int(seed)
        self._epoch = 0  # 用于分布式/多轮训练的确定性打乱

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _rng(self):
        # 每个 epoch 使用不同但可复现的随机序列
        return np.random.RandomState(self._base_seed ^ self._epoch)

    def _iterate_primary_once(self):
        idxs = list(self.primary_indices)  # 直接用原顺序，不调用 rng
        step = self.primary_batch_size
        n = len(idxs) // step if self.drop_last else (len(idxs) + step - 1) // step
        for i in range(n):
            batch = idxs[i * step:(i + 1) * step]
            if len(batch) == step or not self.drop_last:
                yield batch

    def _iterate_secondary_cycle(self, n_batches):
        idxs = list(self.secondary_indices)  # 直接用原顺序
        step = self.secondary_batch_size
        cycled = cycle(idxs)  # 从索引 0 开始循环
        for _ in range(n_batches):
            yield list(islice(cycled, step))

    def __iter__(self):
        # 先生成 primary 的分批
        primary_batches = list(self._iterate_primary_once())
        n_batches = len(primary_batches)
        # 再生成 secondary 的分批（循环补足）
        secondary_batches = list(self._iterate_secondary_cycle(n_batches))
        # 合并
        for pb, sb in zip(primary_batches, secondary_batches):
            yield pb + sb

    def __len__(self):
        # 与 primary 受 drop_last 的批数一致
        if self.drop_last:
            return len(self.primary_indices) // self.primary_batch_size
        else:
            from math import ceil
            return ceil(len(self.primary_indices) / self.primary_batch_size)
class ThreeStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch + primary_batch
            for (primary_batch, secondary_batch, primary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size),
                   grouper(primary_iter, self.primary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
