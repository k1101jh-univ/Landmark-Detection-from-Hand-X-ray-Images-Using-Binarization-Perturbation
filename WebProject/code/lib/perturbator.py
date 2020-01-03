from skimage.filters import threshold_otsu
from skimage.filters import threshold_sauvola
from skimage.transform import resize
import random
import torch

def comb_binary_rec(image, img_size):
    # image = image.numpy()

    min_rec_size = 75
    min_binary = 0.1
    max_binary = 0.2

    while True:
        try:
            random_P = [random.randrange(0, img_size[0] - min_rec_size),
                        random.randrange(0, img_size[1] - min_rec_size)]

            p1 = random_P

            y = random.randrange(random_P[0] + min_rec_size, img_size[0])
            x = random.randrange(random_P[1] + min_rec_size, img_size[1])

            p2 = [y, x]

            crop_img = image[:, p1[0]:p2[0], p1[1]:p2[1]]
            crop_img = crop_img.numpy()

            thresh = threshold_otsu(crop_img)
            break

        except ValueError:
            print("Value Error occured. regenerate rectangle..")

    binary = crop_img > (thresh + random.uniform(min_binary, max_binary))
    binary = binary * 1
    binary = torch.from_numpy(binary)

    image[:, p1[0]:p2[0], p1[1]:p2[1]] = binary
    # plt.imshow(image[0], cmap='gray')
    # plt.show()

    # image = torch.from_numpy(image)

    return image

def comb_sauvola_rec(image, img_size):
    # image = image.numpy()

    window_size = 25
    min_rec_size = 75
    min_binary = 0.1
    max_binary = 0.2

    while True:
        try:
            random_P = [random.randrange(0, img_size[0] - min_rec_size),
                        random.randrange(0, img_size[1] - min_rec_size)]

            p1 = random_P

            y = random.randrange(random_P[0] + min_rec_size, img_size[0])
            x = random.randrange(random_P[1] + min_rec_size, img_size[1])

            p2 = [y, x]

            crop_img = image[:, p1[0]:p2[0], p1[1]:p2[1]]
            crop_img = crop_img.numpy()

            thresh = threshold_sauvola(crop_img, window_size=window_size)
            break

        except ValueError:
            print("Value Error occured. regenerate rectangle..")

    binary = crop_img > (thresh + random.uniform(min_binary, max_binary))
    binary = binary * 1
    binary = torch.from_numpy(binary)

    image[:, p1[0]:p2[0], p1[1]:p2[1]] = binary
    # plt.imshow(image[0], cmap='gray')
    # plt.show()

    # image = torch.from_numpy(image)

    return image

def comb_resize_rec(image, img_size):

    #image = image.numpy()
    mrs = 60  # half of 200
    minv= 20

    min_rec_size = 75

    random_P = [random.randrange(0, img_size[0] - min_rec_size),
                random.randrange(0, img_size[1] - min_rec_size)]

    p1 = random_P

    y = random.randrange(random_P[0] + min_rec_size, img_size[0])
    x = random.randrange(random_P[1] + min_rec_size, img_size[1])

    p2 = [y, x]

    crop_img = image[:, p1[0]:p2[0], p1[1]:p2[1]]

    r_img = resize(crop_img, (1, round((p2[0] - p1[0]) * random.uniform(.05, .1)), round((p2[1] - p1[1]) * random.uniform(.05, .1))))
    r_img = resize(r_img, (1, p2[0] - p1[0], p2[1] - p1[1]))
    r_img = torch.from_numpy(r_img)

    image[:, p1[0]:p2[0], p1[1]:p2[1]] = r_img
    image.double()
    #print(image.dtype)

    return image


def resize_binary(image, img_size):
    oH = img_size[0]
    oW = img_size[1]

    H = oH // 20 * 4    # 1 / 20 sclae
    W = oW // 20 * 4

    image = image.numpy()

    image = resize(image, (3, H, W), anti_aliasing=True)

    thresh = threshold_otsu(image)
    binary = image > thresh
    binary = binary * 1

    image = binary

    image = resize(image, (3, oH, oW), anti_aliasing=True)

    return image