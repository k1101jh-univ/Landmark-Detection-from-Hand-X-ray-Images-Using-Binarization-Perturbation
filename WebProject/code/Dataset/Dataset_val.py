import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
import random
from mytransforms import *
from mytransforms import mytransforms
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
    def __init__(self,  path='train', H=600,W=480, pow_n = 7, aug=True):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])

        init_trans2 = transforms.Compose([transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])

        self.datainfo2 = torchvision.datasets.ImageFolder(root=path, transform=init_trans2)

        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
        self.mask_num = len(self.datainfo.classes)-1
        self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
        self.pow_n = pow_n
        self.aug=aug

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        image, _ = self.datainfo.__getitem__(idx)
        image2, _ = self.datainfo2.__getitem__(idx)
        ori_size = (image2[0].size()[0], image2[0].size()[1])
        mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
        for k in range(0, self.mask_num):
            # X = self.images[idx + (self.data_num * (1 + k))]
            X, _ = self.datainfo.__getitem__(idx + (self.data_num * (1 + k)))
            mask[k] = X

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()

        return [image, mask, ori_size]