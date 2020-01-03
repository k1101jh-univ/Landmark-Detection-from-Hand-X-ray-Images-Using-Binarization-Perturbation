import os
import sys
import io

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, '..')

import numpy as np
import torch
from torchvision import transforms
from Dataset import Dataset_val as DS
from model import unet


import cv2 as cv
import scipy.io
import imageio
from skimage.transform import resize
from scipy.spatial import distance
from PIL import Image

cuda_device = 'cuda:0'
device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

debug = True

num_class = 37
H = 600
W = 480
model_folder = '600_480_pow_5_binary_unet_setup_1'
mat_name = 'mat500'
setup = model_folder[-1]

model = unet.UNet(1, num_class, [64, 128, 256, 512]).to(device)


# 저장된 model 로드
path = os.path.dirname(os.path.abspath(__file__))
model_path = path + '/../../saved_models/Laplace/setup' + setup + '/' + model_folder + '/loss_4.5879849302392e-05_E_759.pth'
print(model_path)
model_path = os.path.abspath(model_path)
model.load_state_dict(torch.load(model_path, map_location=cuda_device)['model'])
model.eval()


###################

def gray_to_rgb(gray):
    h, w = gray.shape
    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = gray
    rgb[:, :, 1] = gray
    rgb[:, :, 2] = gray
    return rgb

#########################################


# 이미지 경로(1장)을 받아 output(랜드마크 찍힌 사진, 랜드마크 위치)를 반환하는 함수
# 입력 이미지 경로, 결과를 저장할 폴더를 입력으로 받음
# 결과로 landmark가 찍힌 이미지와 landmark 포인트가 저장된 txt 파일 생성
def get_output(image_path, save_path):
    image = Image.open(image_path)
    original_w, original_h = image.size

    if debug == True:
        print('original image size: ', original_h, ',', original_w)

    init_trans = transforms.Compose([transforms.Resize((H, W)),
                                     transforms.Grayscale(1),
                                     transforms.ToTensor(),
                                     ])

    input_img = init_trans(image)
    input_img = input_img.unsqueeze(0)

    if debug == True:
        print('resized image size: ', input_img.shape)

    input_img = input_img.to(device)
    output = model(input_img.data).data

    mtx = np.asarray(image, dtype=np.int)
    mtx = gray_to_rgb(mtx)

    landmark_points = []

    for k in range(0, num_class):
        heatmap = output[0][k].cpu()

        heatmap_max_point = np.array(np.where(heatmap > heatmap.max() * .85))
        heatmap_max_point = heatmap_max_point.mean(axis=1)

        heatmap_max_point[0] = (heatmap_max_point[0] * (original_h / H)).astype(int)
        heatmap_max_point[1] = (heatmap_max_point[1] * (original_w / W)).astype(int)

        cv.circle(mtx, (int(heatmap_max_point[1]), int(heatmap_max_point[0])), 10, (0, 0, 1), -1)

        heatmap_max_point = np.ndarray.tolist(heatmap_max_point)

        landmark_points.append(heatmap_max_point)

    if debug == True:
        print("output shape: ", output.shape)

    # 결과 image 저장
    imageio.imwrite(save_path + '/result.png', mtx)

    # landmark 위치 txt 파일로 저장
    f = open(save_path + '/result.txt', 'w')
    for landmark_point in landmark_points:
        f.write(str(landmark_point[0]) + ', ' + str(landmark_point[1]) + '\n')
    f.close()


if __name__ == '__main__':
    get_output('../../images/input/3147.jpg', 'images/result')
