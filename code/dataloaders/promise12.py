import os
import cv2
import torch
import numpy as np
from skimage.exposure import equalize_adapthist
from torch.utils.data import Dataset
import h5py
import itertools
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
from skimage import transform as sk_trans
from scipy.ndimage import rotate, zoom
import pdb
import SimpleITK as sitk
class Promise12(Dataset):
    def __init__(self, base_dir=None, split='train', out_size=256, transform=None):
        self._base_dir = base_dir
        self.out_size = out_size
        self.sample_list = []
        self.split = split
        self.transform = transform
        np_data_path = os.path.join(base_dir+ "/Promise12", 'npy_image')
        if not os.path.exists(np_data_path):
            os.makedirs(np_data_path)
            data_to_array(base_dir, np_data_path, self.out_size, self.out_size)
        else:
            print('read the data from: {}'.format(np_data_path))

        if self.split == 'train':
            self.X_train = np.load(os.path.join(np_data_path, 'X_train.npy'))
            self.y_train = np.load(os.path.join(np_data_path, 'y_train.npy'))

        elif self.split == 'val':
            with open(base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

    def __len__(self):
        if self.split == 'train':
             return self.X_train.shape[0]
        elif self.split == 'val':
             return len(self.sample_list)
    def __getitem__(self, idx):
        np_data_path = os.path.join(self._base_dir + "/Promise12", 'npy_image')
        if self.split == "train":
            img, mask = self.X_train[idx], self.y_train[idx]  # [224,224] [224,224]
        else:
            case = self.sample_list[idx]
            img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
            mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
        img_tensor = torch.from_numpy(img)
        mask_tensor = torch.from_numpy(mask)
        sample = {'image': img_tensor, 'label': mask_tensor}
        if self.split == "train":
            sample = self.transform(sample)
        # sample["idx"] = idx
        return sample

def data_to_array(base_path, store_path, img_rows, img_cols):
    global min_val, max_val
    base_path = base_path + "/Promise12"
    fileList = os.listdir(base_path)
    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [35, 36, 37, 38, 39]
    test_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    train_list = list(set(range(50)) - set(val_list) - set(test_list))

    for the_list in [train_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                print(imgs.shape)
                masks.append(imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images.append(imgs_norm)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)  # (1250,256,256)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        np.save(os.path.join(store_path, 'X_train.npy'), images)
        np.save(os.path.join(store_path, 'y_train.npy'), masks)
    for the_list in [val_list, test_list]:
        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs = imgs.astype(np.uint8)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images = smooth_images(imgs_norm)
                images = images.astype(np.float32)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), images)
def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs
def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                 timeStep=t_step,
                                 numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)

    return imgs
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
        image, label = sample['image'], sample['label']
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


class TwoStreamBatchSampler(Sampler):
    """从两个不同的索引集（即主索引集和次索引集）中按批次（batch）迭代数据。
    其中主索引集（primary indices）按批次一次性加载，而次索引集（secondary indices）则会反复使用，直到加载完主索引集的所有数据
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
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
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
