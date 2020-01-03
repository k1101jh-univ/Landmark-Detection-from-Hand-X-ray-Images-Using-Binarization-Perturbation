import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, '..')

import numpy as np
import torch
from torchvision import transforms
from Dataset import Dataset_val as DS
from model import unet


import cv2 as cv
import scipy.io
from skimage.transform import resize
from scipy.spatial import distance

cuda_device = 'cuda:2'
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

num_class = 37
H=600; W=480;
model_folder = '600_480_pow_5_binary_unet_setup_1'
mat_name = 'mat500'
setup = model_folder[-1]

model = unet.UNet(1, num_class, [64, 128, 256, 512]).to(device)


# 저장된 model 로드
model.load_state_dict(torch.load('../../saved_models/Laplace/setup' + setup + '/' + model_folder + '/loss_4.5879849302392e-05_E_759.pth', map_location=cuda_device)['model'])
model.eval()


###################
def angle(v1, v2):
    v1 = np.array(v1);v2 = np.array(v2);
    v1=v1-[H/2, W/2]; v2=v2-[H/2, W/2];
    r =np.arccos(np.dot(v1, v2.transpose()) / (np.linalg.norm(v1, axis=1).reshape(v1.shape[0],1)\
                                       * np.linalg.norm(v2, axis=1)))
    r[np.isnan(r)]=0
    return r


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target)).to(device)
    return loss


def gray_to_rgb(gray):
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = gray
    rgb[:, :, 1] = gray
    rgb[:, :, 2] = gray
    return rgb

#########################################


# testset을 모두 돌려서 성능을 측정하는 함수
def test_testset(setup_num):
    # 결과 matrix 파일을 저장할 폴더가 존재하지 않을 시 생성
    if not os.path.exists('../../mats/' + model_folder):
        os.makedirs('../../mats/' + model_folder)

    data = DS.MD(path='../../images/Laplace/setup' + setup_num + '/val', H=H, W=W, pow_n=3, aug=False)

    mat = scipy.io.loadmat('../../GT/GT' + setup_num + '.mat')
    mat = np.array(mat['GT'])
    test_gt = mat
    print("testset size: ", test_gt.shape)
    num_land = test_gt.shape[0]

    count = 0
    ed = []
    for k in range(0, num_land):
        print('=== num :', k + 1)
        x, masks, ori_size = data.__getitem__(k)

        inputs = x
        inputs = inputs.unsqueeze(0)

        inputs = inputs.to(device)
        outputs = model(inputs.data)
        output = outputs.data

        oW = ori_size[1]
        oH = ori_size[0]

        wrist_width = distance.euclidean(test_gt[k, 0, :], test_gt[k, 4, :])
        print("wrist_width : ", wrist_width)

        for jj in range(0, num_class):

            A = output[0][jj]
            A = A.cpu()

            amax = np.array(np.where(A > A.max() * .85))
            amax = amax.mean(axis=1)

            bmax = test_gt[k, jj, :]

            dst = distance.euclidean([amax[0] * (oH / H), amax[1] * (oW / W)], [bmax[1], bmax[0]])
            dst_ori = dst
            dst = dst * (50 / wrist_width)

            ed.append(dst)
            print("x : {0:2.0f}".format(jj + 1), "dst : {0:0.10f}".format(dst), "ori : {0:0.10f}".format(dst_ori))

            if dst > 2:
                count = count + 1
                print(count)

    mm = 2
    mtx = np.array(ed).reshape(num_land, num_class)

    mm2 = np.mean(mtx <= mm)
    mm2_5 = np.mean(mtx <= mm * 1.25)
    mm3 = np.mean(mtx <= mm * 1.5)
    mm4 = np.mean(mtx <= mm * 2)

    print("2mm: %.4f  " % (mm2), "2.5mm: %.4f  " % (mm2_5),\
          "3mm: %.4f  " % (mm3), "4mm: %.4f  " % (mm4), "ave: %.4f" % np.mean(mtx), "std: %.4f" % np.std(mtx[:, :]))

    mtx_dic = {}
    mtx_dic['vect'] = mtx

    scipy.io.savemat('../../mats/' + model_folder + '/' + mat_name + '.mat', mtx_dic)

    result = np.zeros((num_class, 4))
    for i in range(num_class):
        result[i][0] = np.mean(mtx[:, i] < mm)
        result[i][1] = np.mean(mtx[:, i] < mm * 1.25)
        result[i][2] = np.mean(mtx[:, i] < mm * 1.5)
        result[i][3] = np.mean(mtx[:, i] < mm * 2)

    for i in range(num_class):
        print("point num :%2d " % (i + 1), end=' ')
        for j in range(4):
            print('%.3f ' % (result[i][j]), end=' ')
        print('\t\t', end='')

        point_mean = np.mean(mtx[:, i])
        point_std = np.std(mtx[:, i])
        print('%.3f ' % point_mean, end=' ')
        print('%.3f ' % point_std, end=' ')
        print('')


if __name__ == '__main__':
    test_testset(setup)