import os,sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, '..')

import torch
from Dataset import Dataset as DS
from lib import network
from collections import defaultdict
import numpy as np

import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time


#### path_setup ####

## codes0
p_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
## Digital Hand Atlas
gp_path = os.path.dirname(p_path)


#### device setup ####
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)


#### model & file setup ####
model_folder = '600_480_pow_5_test_'
file_name = 'loss_{}_E_{}.pth'


Attu = False
use_M = False
binary = False
use_parallel = False

batch_size = 1
setup = 1
mask_type = 'L'

H = 600
W = 480
pow_n = 5
num_epochs = 800
lr = 1e-4

save_epoch = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 149, 199, 299, 399, 499, 599, 699, 799, 899, 999, 1499, 1999]

## Load model
load_model = False
model_name = 'L1_600480_pow3_loss_0.0017326861899658638_E_499.pth'


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    print("{}: {}".format(phase, ", ".join(outputs)))


def L1_loss(pred, target):
    loss = torch.mean(torch.abs(pred - target)).to(device)
    return loss


def L2_loss(pred, target):
    loss = torch.mean(torch.pow((pred - target), 2)).to(device)
    return loss


def angle_mat(v1, v2):
    v1 = np.array(v1);
    v2 = np.array(v2)
    v1 = v1 - [H / 2, W / 2];
    v2 = v2 - [H / 2, W / 2];
    r = np.arccos(np.dot(v1, v2.transpose()) / (np.linalg.norm(v1, axis=1).reshape(v1.shape[0], 1) \
                                                * np.linalg.norm(v2, axis=1)))
    r[np.isnan(r)] = 0
    return r


def dist_mat(v1, v2):
    v1 = np.array(v1);
    v2 = np.array(v2);
    y1, y2 = np.meshgrid(v1[:, 0], v2[:, 0])
    x1, x2 = np.meshgrid(v1[:, 1], v2[:, 1])
    dist_ = np.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2);
    # dist_ = dist_ / dist_.max()
    return dist_


def s_to_m(output, label):  # tensor(1,19,600,480)
    pred_mtx = [];
    gt_mtx = [];
    for k in range(0, 19):
        A = output[0][k];
        A = A.cpu()
        B = label[0][k];
        B = B.cpu()
        amax = np.array(np.where(A == A.max()))
        bmax = np.array(np.where(B == B.max()))
        pred_mtx.append(amax[:, 0])
        gt_mtx.append(bmax[:, 0])
    return pred_mtx, gt_mtx


def ACloss(pred, target):
    # L1loss = torch.mean(torch.abs(pred - target))  # L1 loss
    L2_loss = torch.mean(torch.pow((pred - target), 2))

    # plt.imshow(target[0][0]>.9, cmap='jet');
    # plt.show()
    if batch_size > 1:
        ang_loss = 0
        dist_loss = 0
        for ii in range(0, target.size(0)):
            p, gt = s_to_m(pred[ii].data.unsqueeze(0), target[ii].unsqueeze(0))  # comput XY point matrix
            angle_pred = np.array(angle_mat(p, p))
            angle_gt = np.array(angle_mat(gt, gt))  # angles
            ang_loss = ang_loss + np.mean(np.abs(angle_pred - angle_gt))

            dist_pred = np.array(dist_mat(p, p))
            dist_gt = np.array(dist_mat(gt, gt))  # angles
            dist_loss = dist_loss + np.mean(np.abs(dist_pred - dist_gt))
    else:
        p, gt = s_to_m(pred.data, target)  # comput XY point matrix
        angle_pred = np.array(angle_mat(p, p))
        angle_gt = np.array(angle_mat(gt, gt))  # angles
        ang_loss = np.mean(np.abs(angle_pred - angle_gt))
        dist_pred = np.array(dist_mat(p, p))
        dist_gt = np.array(dist_mat(gt, gt))  # angles
        dist_loss = np.mean(np.abs(dist_pred - dist_gt))

    w_loss = (1 + ang_loss) + np.log(dist_loss)
    loss = torch.mul(L2_loss, w_loss)
    # loss = (L1loss * 1e-10)  + w_loss

    return loss, L2_loss, w_loss


def train():
    for epoch in range(start_epoch, num_epochs):
        print('===' * 30)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('---' * 30)

        now = time.time()

        if (epoch + 1) % valtest == 0:
            uu = ['train', 'val']
        else:
            uu = ['train']

        for phase in uu:
            # since = time.time()
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # print("in shape : ", inputs.shape)

                # plt.imshow(inputs[0][1]);
                # plt.show()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Forward computation
                    outputs = model(inputs).to(device)
                    # plt.imshow(outputs[0][0].cpu().detach().numpy(), cmap='jet')
                    # plt.show()
                    loss, L2_loss, w_loss = ACloss(outputs, labels)
                    metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
                    metrics['L2_loss'] += loss.data.cpu().numpy() * labels.size(0)
                    metrics['w_loss'] += loss.data.cpu().numpy() * labels.size(0)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                # statistics
                epoch_samples += inputs.size(0)

                # Each epoch has a training and validation phase

            # imageio.imwrite(gp_path + '/images/arg_image/main_model/{}.png'.format(img_num),
            #                 (outputs[0][0].cpu() * 255).type(torch.uint8))
            # img_num += 1

            # epoch_loss = metrics['loss'] / epoch_samples
            # print(phase, "loss : ", loss.data.cpu().numpy(), "epoch loss : ", epoch_loss)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_L2_loss = metrics['L2_loss'] / epoch_samples
            epoch_ang_loss = metrics['w_loss'] / epoch_samples
            print(phase, loss, "epoch loss : ", epoch_loss, \
                  "epoch L2 loss: ", epoch_L2_loss, "w loss:", epoch_ang_loss)

            # deep copy the model
            if phase == 'val' and ((epoch in save_epoch) or epoch_loss < best_loss):
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                state = {
                    'model': model.state_dict(),
                    'epoch': epoch + 1,
                    'optim': optimizer,
                    'scheduler': scheduler
                }

                save_file_path = save_path + '/' + file_name.format(best_loss, epoch)

                # torch.save(state, 'model_600480_L1/L1_s600480_pow3_e_{}_loss_{}.pth'.format(epoch, best_loss))
                torch.save(state, save_file_path)

        print(time.time() - now)

        # if (epoch+1) % valtest == 0:
        #     out = outputs.data
        #     for k in range(0,19):
        #         out1 = out[0, k, :, :];
        #         out1 = out1 / out1.max()
        #         plt.imshow(out1,cmap='jet');
        #         plt.show()
        #
        #         l=labels[0][k]
        #         #l[out1 > out1.max()*.0 ]=0
        #         plt.imshow(l,cmap='jet');
        #         plt.show()


if __name__ == '__main__':
    if mask_type == 'G':
        mask_folder = 'Gaussian'
    else:
        mask_folder = 'Laplace'

    save_path = gp_path + '/models/' + mask_folder + '/setup' + str(setup) + '/' + model_folder

    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    else:
        print("model Exists!!")
        # sys.exit(1)

    train_MD = DS.MD(path = gp_path + '/images/' + mask_folder + '/setup' + str(setup) + '/train', H=H, W=W, pow_n=pow_n, aug=True, use_M = use_M, binary = binary)
    val_MD = DS.MD(path = gp_path + '/images/' + mask_folder + '/setup' + str(setup) + '/val', H=H, W=W, pow_n=pow_n, aug=False, use_M = use_M, binary = False)

    dataloaders = {
        'train': DS.DataLoader(train_MD, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory= True),
        'val': DS.DataLoader(val_MD, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory= True)
    }

    num_class = 37; n_input=1;
    if Attu == True:
        model = network.attention_unet.AttU_Net(n_input, num_class).to(device)
    else:
        model = network.unet.UNet(n_input, num_class, [64, 128, 256, 512]).to(device)

    #model = DataParallelModel(model)

    start_epoch = 0

    #### Transfer learning ################################
    if load_model == True:
        loaded_model = torch.load(save_path + '/' + model_name)
        model.load_state_dict(loaded_model['model'])
        start_epoch = loaded_model['epoch']
        optimizer = loaded_model['optim']
        scheduler = loaded_model['scheduler']


    # Observe that all parameters are being optimized

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [900, 1200, 1300, 1400], gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    print("GPU : ", device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    valtest = 10
    img_num = 0

    train()

    state = {
        'model': model.state_dict(),
        'epoch': num_epochs,
        'optim': optimizer,
        'scheduler': scheduler
    }

    # save_file_path = save_path + '/' + file_name.format(best_loss, num_epochs)
    # torch.save(state, save_file_path)

    print('Best val loss: {:4f}'.format(best_loss))

    # # load best model weights
    # model.load_state_dict(best_model_wts)