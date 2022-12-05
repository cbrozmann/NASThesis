import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from .ntu_60_get_and_transform_dataset import ntu60_get_all_data


class NTU120Dataset(VisionDataset):
    """`NTU Dataset`
    Args:
        root (string): Root directory of dataset where directory ``ntu_120`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """

    base_folder = "ntu_120"
    train_folder_name = 'train'
    cross_subject_folder_name = 'cross_subject'
    cross_setup_folder_name = 'cross_setup'

    test_folder_name = "test"
    meta = {
        "filename": "meta_classes.npy",
        "key": "label_names"
    }
    # fill with zeros or stretch/resize existing image
    fill_with_zeros = False

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        use_cross_subject: bool = True,
        # use 76 for cross_subject True and 72 if False
        desired_width: int = 76
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.use_cross_subject = use_cross_subject
        self.desired_width = desired_width

        if self.use_cross_subject:
            data_root_path = os.path.join(self.root, self.base_folder, self.cross_subject_folder_name)
        else:
            data_root_path = os.path.join(self.root, self.base_folder, self.cross_setup_folder_name)

        if self.train:
            data_path = os.path.join(data_root_path, self.train_folder_name)
        else:
            data_path = os.path.join(data_root_path, self.test_folder_name)

        self.data: Any = []
        self.targets = []
        all_data = ntu60_get_all_data(data_path, self.desired_width, self.fill_with_zeros, True, True)
        self.data = all_data['data']
        self.targets = all_data['targets']

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
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

