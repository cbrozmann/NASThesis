import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from .get_and_transform_dataset import get_all_data


class UTDMHADDataset(VisionDataset):
    """`UTD-MHAD Dataset`


    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "utdmhad_skeleton"

    # url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    # filename = "cifar-10-python.tar.gz"
    # tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    # train_list = [
    #     ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
    #     ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
    #     ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
    #     ["data_batch_4", "634d18415352ddfa80567beed471001a"],
    #     ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    # ]
    train_folder_name = "train"

    # test_list = [
    #     ["test_batch", "40351d587109b95175f43aff81a1287e"],
    # ]
    test_folder_name = "test"
    meta = {
        "filename": "meta_classes.npy",
        "key": "label_names"
    }
    desired_width = 68
    # fill with zeros or stretch/resize existing image
    fill_with_zeros = False

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            data_path = os.path.join(self.root, self.base_folder, self.train_folder_name)
        else:
            data_path = os.path.join(self.root, self.base_folder, self.test_folder_name)

        # if self.train:
        #     downloaded_list = self.train_list
        # else:
        #     downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []
        all_data = get_all_data(data_path, self.desired_width, self.fill_with_zeros, True)
        self.data = all_data['data']
        self.targets = all_data['targets']

        self._load_meta()

    def _load_meta(self) -> None:
        # path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        path = os.path.join(self.root, self.meta["filename"])
        with open(path, "rb") as infile:
            self.classes = np.load(infile)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"

