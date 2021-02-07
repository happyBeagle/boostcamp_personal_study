from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
import numpy as np
import torch 
import codecs
import string
import tarfile
import gzip
import lzma
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, verify_str_arg


import pandas as pd
import torch.nn.functional as F


from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class NotMNIST(VisionDataset):

    resources = [("http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz", "notMNIST_small.tar.gz"),
                 ("http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz", "notMNIST_large.tar.gz")]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['A','B','C','D','E','F','G','H','I','J']

    @property
    def train_labels(self):
        return self.target
    
    @property
    def test_labels(self):
        return self.target

    @property
    def train_data(self):
        return self.data
    
    @property
    def test_data(self):
        return self.data  


    def __init__(
        self,
        root: str,
        train: bool,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super(NotMNIST, self).__init__(
                                        root, 
                                        transform= transform,
                                        target_transform=target_transform)
        
        self.train = train

        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not find. You can use download=True to downlaod it')

        
        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    
    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    
    @property
    def class_to_index(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and 
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))


    def download(self) -> None:
        if self._check_exists():
            return
        
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        for url, file_name in self.resources:
            download_and_extract_archive(url, download_root=self.raw_folder, filename=file_name, md5=None)

        
        print("processing.......")

        training_set = (
            read_image_and_label(os.path.join(self.raw_folder, 'notMNIST_small'))
        )
        test_set = (
            read_image_and_label(os.path.join(self.raw_folder, 'notMNIST_large'))
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

def read_image_and_label(path: str) -> torch.Tensor:
    data = []
    label = []

    class_list = os.listdir(path)
    for temp_class in class_list:
        class_folder_path = os.path.join(path, temp_class)
        for temp_file in os.listdir(class_folder_path):
            try:
                img = Image.open(os.path.join(class_folder_path,temp_file))
                arr_image = np.array(img)
                
                data.append(arr_image)
                label.append(ord(temp_class) - ord('A'))
            except:
                continue

    data = torch.from_numpy(np.array(data))
    label = torch.from_numpy(np.array(label))
    
    assert(label.ndimension() == 1)
    assert(data.ndimension() == 3)

    return data, label