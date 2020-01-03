import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import random
from mytransforms import *
from mytransforms import mytransforms

from lib import perturbator

import time

def myimshow(i, m):
    r = torch.empty(i.shape[0], i.shape[1], 3)
    r[:,:,0]=i; r[:,:,1]=i; r[:,:,2]=i;
    for k in range(0, 19):
        r[:,:,1][m[k] > .95] = 1
        r[:, :, 0][m[k] > .95] = 0
        r[:, :, 2][m[k] > .95] = 0
    plt.imshow(r);plt.show()

class MD(Dataset):
    def __init__(self,  path='train', H=600, W=480, pow_n=3, aug=True, use_M = False, binary = False):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])

        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
        self.mask_num = len(self.datainfo.classes)-1
        self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
        self.aug=aug
        self.H = H
        self.W = W
        self.pow_n = pow_n
        self.use_M = use_M
        self.binary = binary

        if self.use_M == True:
            self.images = []

            print("load image")
            t = time.time()

            for i in range(len(self.datainfo)):
                self.images.append(self.datainfo.__getitem__(i)[0])

            print("datainfo : ",  self.datainfo)
            print("data_num : ", self.data_num)
            print("images : ", len(self.images))
            print("load time : ", time.time() - t)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.use_M == True:
            image = self.images[idx]
        else:
            image, _ = self.datainfo.__getitem__(idx)

        if self.aug == True:
            self.rv = random.random()
        else:
            self.rv = 1

        if self.rv < 0.9:
            # augmenation of img and masks
            angle = random.randrange(-15, 15)

            # trans img with masks
            self.data_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                    mytransforms.Rotation(angle),
                                                    mytransforms.ColorJitter(brightness=random.random(),
                                                                             contrast=random.random(),
                                                                             saturation=random.random(),
                                                                             hue=random.random() / 2),
                                                    mytransforms.ToTensor(),
                                                    ])
            self.mask_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                    mytransforms.Rotation(angle),
                                                    mytransforms.ToTensor(),
                                                    ])

            if self.binary == True:
                image = perturbator.comb_binary_rec(image, [self.H, self.W])
                #image = comb_black_rec(image, [self.H, self.W])
            image = self.data_trans(image)

            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            if self.use_M == True:
                for k in range(0, self.mask_num):
                    X = self.images[idx + (self.data_num * (1 + k))]
                    mask[k] = self.mask_trans(X)
            else:
                for k in range(0, self.mask_num):
                    X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                    mask[k] = self.mask_trans(X)
        else:
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            if self.use_M == True:
                for k in range(0, self.mask_num):
                    X = self.images[idx + (self.data_num * (1 + k))]
                    mask[k] = X
            else:
                for k in range(0, self.mask_num):
                    X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
                    mask[k] = X
        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()

        return [image, mask]