import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = path
        if train:
            self.airplane_path = path + 'train/airplane'
            self.automobile_path = path + 'train/automobile'
            self.bird_path = path + 'train/bird'
            self.cat_path = path + 'train/cat'
            self.deer_path = path + 'train/deer'
            self.dog_path = path + 'train/dog'
            self.frog_path = path + 'train/frog'
            self.horse_path = path + 'train/horse'
            self.ship_path = path + 'train/ship'
            self.truck_path = path + 'train/truck'
        else:
            self.airplane_path = path + 'test/airplane'
            self.automobile_path = path + 'test/automobile'
            self.bird_path = path + 'test/bird'
            self.cat_path = path + 'test/cat'
            self.deer_path = path + 'test/deer'
            self.dog_path = path + 'test/dog'
            self.frog_path = path + 'test/frog'
            self.horse_path = path + 'test/horse'
            self.ship_path = path + 'test/ship'
            self.truck_path = path + 'test/truck'

        self.airplane_img_list = glob.glob(self.airplane_path + '/*.png')
        self.automobile_img_list = glob.glob(self.automobile_path + '/*.png')
        self.bird_img_list = glob.glob(self.bird_path + '/*.png')
        self.cat_img_list = glob.glob(self.cat_path + '/*.png')
        self.deer_img_list = glob.glob(self.deer_path + '/*.png')
        self.dog_img_list = glob.glob(self.dog_path + '/*.png')
        self.frog_img_list = glob.glob(self.frog_path + '/*.png')
        self.horse_img_list = glob.glob(self.horse_path + '/*.png')
        self.ship_img_list = glob.glob(self.ship_path + '/*.png')
        self.truck_img_list = glob.glob(self.truck_path + '/*.png')

        self.transform = transform

        self.img_list = self.airplane_img_list + self.automobile_img_list + \
            self.bird_img_list + self.cat_img_list + \
                self.deer_img_list + self.dog_img_list + \
                    self.frog_img_list + self.horse_img_list + \
                        self.ship_img_list + self.truck_img_list

        self.Image_list = []
        for img_path in self.img_list:
            self.Image_list.append(Image.open(img_path))

        self.class_list = [0] * len(self.airplane_img_list) + [1] * len(self.automobile_img_list) + \
            [2] * len(self.bird_img_list) + [3] * len(self.cat_img_list) + \
                [4] * len(self.deer_img_list) + [5] * len(self.dog_img_list) + \
                    [6] * len(self.frog_img_list) + [7] * len(self.horse_img_list) + \
                        [8] * len(self.ship_img_list) + [9] * len(self.truck_img_list)
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.Image_list[idx]
        label = self.class_list[idx]

        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])
        if self.transform is None:
            transform = transforms.Compose([
                # you can add other transformations in this list
                transforms.ToTensor()
            ])
            img = transform(img)
        else:
            img = self.transform(img)

        return img, label
