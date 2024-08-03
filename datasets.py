import pathlib
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image

class Custom_dataset(torch.utils.data.Dataset):
    def __init__(self, kind, transform):
        assert kind in ['train', 'valid', 'test']
        
        if kind == 'train':
            self.dirs = [str(i) for i in pathlib.Path('./dataset/train/').rglob('*.png')]
        elif kind == 'valid':
            self.dirs = [str(i) for i in pathlib.Path('./dataset/valid/').rglob('*.png')]
        elif kind == 'test':
            self.dirs = [str(i) for i in pathlib.Path('./dataset/test/').rglob('*.png')]

        self.transform = transform

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, index):
        image_path = self.dirs[index]
        image = Image.open(image_path)
        image = self.transform(image)

        label = int(image_path.split('/')[-1].split('_')[-1].replace('.png', ''))

        return image, torch.tensor(label).to(torch.long)


def set_datasets(image_size=224):

    # train transform
    train_transform = create_transform(
        input_size = image_size,
        is_training = True,
        auto_augment = 'rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        mean = IMAGENET_DEFAULT_MEAN,
        std = IMAGENET_DEFAULT_STD,
    )

    valid_transform = []
    valid_transform.append(
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    # valid_transform.append(transforms.CenterCrop(image_size))
    valid_transform.append(transforms.ToTensor())
    valid_transform.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    valid_transform = transforms.Compose(valid_transform)


    train_dataset = Custom_dataset('train', train_transform)
    valid_dataset = Custom_dataset('valid', valid_transform)
    test_dataset = Custom_dataset('test', valid_transform)

    return train_dataset, valid_dataset, test_dataset



if __name__ == '__main__':
    train, valid, test = set_datasets(100)
    print(train.__len__(), valid.__len__(), test.__len__())
    print(train.__getitem__(0)[0].shape)