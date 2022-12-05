##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
import os, sys, torch, json
import os.path as osp
import numpy as np
import torchvision.datasets as dset
import torchvision.transforms as transforms
from copy import deepcopy
from PIL import Image

from xautodl.config_utils import load_config

from .DownsampledImageNet import ImageNet16
from .SearchDatasetWrap import SearchDataset
from .UTD_MHAD_Dataset import UTDMHADDataset  # new
from .NTU_60_Dataset import NTUDataset  # new
from .NTU_120_Dataset import NTU120Dataset  # new


Dataset2Class = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet-1k-s": 1000,
    "imagenet-1k": 1000,
    "ImageNet16": 1000,
    "ImageNet16-150": 150,
    "ImageNet16-120": 120,
    "ImageNet16-200": 200,
    'utdmhad': 27,  # new
    'ntu60_cross_subject': 60,  # new
    'ntu60_cross_view': 60,  # new
    'ntu120_cross_subject': 120,  # new
}


class CUTOUT(object):
    def __init__(self, length):
        self.length = length

    def __repr__(self):
        return "{name}(length={length})".format(
            name=self.__class__.__name__, **self.__dict__
        )

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


imagenet_pca = {
    "eigval": np.asarray([0.2175, 0.0188, 0.0045]),
    "eigvec": np.asarray(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    ),
}


class Lighting(object):
    def __init__(
        self, alphastd, eigval=imagenet_pca["eigval"], eigvec=imagenet_pca["eigvec"]
    ):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.0:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype("float32")
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), "RGB")
        return img

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_datasets_no_augmentation(name, root, cutout):
    if name == 'utdmhad':  # new
        mean = [x / 255 for x in [60.345893612665485, 48.04306844547564, 241.82972737819026]]  # new
        std = [x / 255 for x in [8.877062809530207, 28.94719770153745, 11.07850571439358]]  # new
    elif name == 'ntu60_cross_subject':  # new
        mean = [x / 255 for x in [53.071580996606215, 43.60475026543529, 233.74303839488056]]  # new
        std = [x / 255 for x in [23.32199342771104, 24.374298535860184, 14.255224539298803]]  # new
    elif name == 'ntu60_cross_view':  # new
        mean = [x / 255 for x in [49.801968992877924, 44.162030007093946, 233.35067068761307]]  # new
        std = [x / 255 for x in [22.79544660829292, 24.454556351066618, 15.176378038626176]]  # new
    elif name == 'ntu120_cross_subject':  # new
        mean = [x / 255 for x in [55.57351470186907, 43.290841130115616, 233.58642493785632]]  # new
        std = [x / 255 for x in [23.626959978758112, 23.9743121178058, 14.321376694674635]]  # new
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    # Data Argumentation
    if name == 'utdmhad':  # new start
        lists = [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((20, 68), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 20, 68)  # new end
    elif name == 'ntu60_cross_subject':  # new start
        lists = [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((20, 68), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 28, 84)  # new end
    elif name == 'ntu60_cross_view':  # new start
        lists = [
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop((20, 68), padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 28, 88)  # new end
    elif name == 'ntu120_cross_subject':  # new start
        lists = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        xshape = (1, 3, 28, 76)  # new end
    elif name == "tiered":
        lists = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(80, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        if cutout > 0:
            lists += [CUTOUT(cutout)]
        train_transform = transforms.Compose(lists)
        test_transform = transforms.Compose(
            [
                transforms.CenterCrop(80),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        xshape = (1, 3, 32, 32)
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    if name == 'utdmhad':  # new start
        train_data = UTDMHADDataset(root, True, train_transform, None)
        test_data = UTDMHADDataset(root, False, test_transform, None)  # new end
    elif name == 'ntu60_cross_subject':  # new start
        train_data = NTUDataset(root, True, train_transform, None, use_cross_subject=True, desired_width=84)
        test_data = NTUDataset(root, False, test_transform, None, use_cross_subject=True, desired_width=84)  # new end
    elif name == 'ntu60_cross_view':  # new start
        train_data = NTUDataset(root, True, train_transform, None, use_cross_subject=False, desired_width=88)
        test_data = NTUDataset(root, False, test_transform, None, use_cross_subject=False, desired_width=88)  # new end
    elif name == 'ntu120_cross_subject':  # new start
        train_data = NTU120Dataset(root, True, train_transform, None, use_cross_subject=True, desired_width=76)
        test_data = NTU120Dataset(root, False, test_transform, None, use_cross_subject=False, desired_width=76)  # new end
    else:
        raise TypeError("Unknow dataset : {:}".format(name))

    class_num = Dataset2Class[name]
    return train_data, test_data, xshape, class_num


def get_nas_search_loaders(
    train_data, valid_data, dataset, config_root, batch_size, workers
):
    if isinstance(batch_size, (list, tuple)):
        batch, test_batch = batch_size
    else:
        batch, test_batch = batch_size, batch_size
    if dataset == 'utdmhad':    # new start
        # split train data
        num_images = 431  # split the train_data
        # valid_split = S7 (subject 7)
        valid_split = [x for x in range(0, 112) if x % 16 > 11] + [x - 1 for x in range(113, 432) if x % 16 > 11]
        # train split = S1, S3, S5
        train_split = [x for x in range(0, 431) if x not in valid_split]
        split_info = {'train': train_split, 'valid': valid_split}  # new end
        # copy from cifar10
        # To split data
        xvalid_data = deepcopy(train_data)
        if hasattr(xvalid_data, "transforms"):  # to avoid a print issue
            xvalid_data.transforms = valid_data.transform
        xvalid_data.transform = deepcopy(valid_data.transform)
        search_data = SearchDataset(dataset, train_data, train_split, valid_split)
        # data loader
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(train_split),
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            xvalid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_split),
            num_workers=workers,
            pin_memory=True,
        )  # new end
    elif dataset in ['ntu60_cross_subject', 'ntu60_cross_view', 'ntu120_cross_subject']:    # new start
        if dataset == 'ntu60_cross_subject':  # new start
            split_filename = 'ntu_60_coss_subject-subject_split.json'
            num_images = 40091  # split the test_data
        elif dataset == 'ntu60_cross_view':
            split_filename = 'ntu_60_coss_view-subject_split.json'
            num_images = 37646  # split the test_data
        else:
            split_filename = 'ntu_120_coss_view-subject_split.json'
            num_images = 63026
        with open("{:}/{:}".format(config_root, split_filename), "r") as f:
            ntu_splits = json.load(f)
        # copy from cifar100
        search_train_data = train_data
        search_valid_data = deepcopy(valid_data)
        search_valid_data.transform = train_data.transform
        search_data = SearchDataset(
            dataset,
            [search_train_data, search_valid_data],
            list(range(len(search_train_data))),
            ntu_splits["valid"],
        )
        search_loader = torch.utils.data.DataLoader(
            search_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_data,
            batch_size=test_batch,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                ntu_splits["valid"]
            ),
            num_workers=workers,
            pin_memory=True,
        )  # new end
    else:
        raise ValueError("invalid dataset : {:}".format(dataset))
    return search_loader, train_loader, valid_loader


# if __name__ == '__main__':
#  train_data, test_data, xshape, class_num = dataset = get_datasets('cifar10', '/data02/dongxuanyi/.torch/cifar.python/', -1)
#  import pdb; pdb.set_trace()
