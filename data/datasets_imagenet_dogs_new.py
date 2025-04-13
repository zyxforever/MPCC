import torchvision
from PIL import Image
import os
import os.path
import numpy as np
from skimage import io, color

import torchvision.datasets as datasets

from torch.utils.data.dataset import Dataset

class ImageNetDogs(Dataset):
    """`ImageNet10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'imagenet-10'
    class_names_file = 'class_names.txt'
    train_list = [
        ['ImageNet10_112.h5', '918c2871b30a85fa023e0c44e0bee87f'],
    ]

    splits = ('train', 'test')

    def __init__(self, split='train',
                 transform=None, target_transform=None, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))

        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # train/test/unlabeled set
        self.classes = ['Maltese dog','Blenheim spaniel','basset','elkhound','giant schnauzer',
                        'golden retriever','Brittany spaniel','clumber','Welsh springer spaniel','groenendael',
                        'kelpie','Shetland','Doberman','pug','chow']
        self.data, self.targets = self.__loadfile()
        print("Dataset Loaded.")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(np.uint8(img)).convert('RGB')
        # class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index, 'class_name': 'unlabeled'}}

        return out

    def __len__(self):
        return len(self.data)

    def __loadfile(self):
        datas,labels = [],[]
        source_dataset = torchvision.datasets.ImageFolder(root='/root/data/datasets/imagenet-dogs')

        for line,tar in zip(source_dataset.imgs,source_dataset.targets):
            try:
                img = io.imread(line[0])
                # img = color.gray2rgb(img)
            except:
                print(line[0])
                continue
            else:
                datas.append(img)
                labels.append(tar)

        return datas, labels

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
