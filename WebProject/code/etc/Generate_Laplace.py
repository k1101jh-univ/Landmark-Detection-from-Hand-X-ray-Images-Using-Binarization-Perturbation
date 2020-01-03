import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import scipy.misc

scale = 15

annotation_path = '../../Dataset/annotation.idl'

file = open(annotation_path, 'r')
lines = file.readlines()
file.close()

image_name = []
points = []
for i in range(len(lines)):
    image_name.append(lines[i][1:5])
    points.append(lines[i][13:].split('('))
    for j in range(len(points[i])):
        points[i][j] = points[i][j][0:-2]
        points[i][j] = points[i][j].split(',')
        points[i][j] = points[i][j][0:2]
        #print(points[i][j])
        for k in range(len(points[i][j])):
            points[i][j][k] = int(points[i][j][k])

# for i in range(len(image_name)):
#     print(points[i])

validation_setup = 1

validation_setup_path = '../../Dataset/x-validation-setup/' + str(validation_setup)
img_folder = ['ASIF', 'ASIM', 'BLKF', 'BLKM', 'CAUF', 'CAUM', 'HISF', 'HISM']
folder_path = '../../images/Laplace/setup' + str(validation_setup) + '/'
image_path = '../../Dataset/Digital Hand Atlas/JPEGimages'

file = open(validation_setup_path + '/test.txt', 'r')
test_set = file.readlines()
file.close()
file = open(validation_setup_path + '/train.txt', 'r')
train_set = file.readlines()
file.close()

for i in range(len(test_set)):
    test_set[i] = test_set[i][:-1]

for i in range(len(train_set)):
    train_set[i] = train_set[i][:-1]

print("test set len : ", len(test_set))
print("train set len : ", len(train_set))

count = 0

if os.path.exists(folder_path + 'train/0images') == False:
    os.makedirs(folder_path + 'train/0images')

if os.path.exists(folder_path + 'val/0images') == False:
    os.makedirs(folder_path + 'val/0images')


for i in range(len(points)):

    img_name = image_name[i]
    #print(img_name)

    if img_name in test_set:
        path = folder_path + 'val'

    elif img_name in train_set:
        path = folder_path + 'train'

    temp_path = path + '/' + '0images'

    ## save Image ##

    get_img = False

    for j in range(8):
        if(get_img == False):
            for k in range(19):
                if(get_img == False):
                    for file in os.listdir(image_path + '/' + img_folder[j] + '/' + img_folder[j] + str(k).zfill(2)):
                        if(file[0:-4] == img_name):
                            img = Image.open(image_path + '/' + img_folder[j] + '/' + img_folder[j] + str(k).zfill(2) + '/' + file)
                            get_img = True
                            W = img.size[0]
                            H = img.size[1]
                            count += 1
                            print(file[0:-4])
                            break

    img.save(temp_path + '/' + img_name + '.jpg')


    ## make groundtruth

    #
    for j in range(len(points[i])):
        Xmtx = np.arange(W); Xmtx = np.matlib.repmat(Xmtx, H, 1)
        Ymtx = np.arange(H); Ymtx = np.matlib.repmat(Ymtx, W, 1).transpose()
        cwmtx = np.ones((H, W)) * points[i][j][0]; chmtx = np.ones((H, W)) * points[i][j][1]
        Xdist = (Xmtx - cwmtx); Ydist = (Ymtx - chmtx)
        #g = (1/(scale * np.sqrt(2 * np.pi)) * np.exp(-(((Xdist ** 2) + (Ydist ** 2)) / (2 * scale ** 3))))
        g = (1/(2 * scale)) * np.exp(-((abs(Xdist) + abs(Ydist)) / (scale ** 2)))
        g = g / g.max()
        g *= 255

        #g.save(temp_path + '/' + img_name + '.jpg')
        #scipy.misc.imsave(temp_path + '/' + img_name + '.jpg')
        temp_path = path + '/' + str(j + 1 + 100)
        if os.path.exists(temp_path) == False:
            os.makedirs(temp_path)
        g2 = Image.fromarray(g)
        g2 = g2.convert('RGB')
        g2.save(temp_path + '/' + img_name + '.png')



        # print("x : ", points[i][j][0], " y : ", points[i][j][1])
        # print(g[points[i][j][0]][points[i][j][1]])
        # plt.imshow(g, cmap='jet')
        # plt.show()

print(count)
#
# print(Xmtx.shape)
# print(cwmtx.shape)

