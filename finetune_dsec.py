from __future__ import print_function
from pprint import pprint
import argparse
import copy
import math
import os
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import shape
import skimage
import skimage.io
import skimage.transform
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import cv2
#from scripts import dataloading_dsec as dsec
from torch.autograd import Variable
from dataset1.visualization import show_image
from dataloader import KITTILoader as DA
from dataloader import KITTIloader2015 as ls
from models import *
import gc
# from torch.cuda.amp import GradScaler, autocast




parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./trained/submission_model.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

    
args = parser.parse_args()
#dsec.args = args
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'dsec':
    from dataloader import DSECloader as ls


all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
    args.datapath)

#print(all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=2, shuffle=True, num_workers=3, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
    batch_size=2, shuffle=False, num_workers=0, drop_last=False)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(
    sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))


def train(batch_idx, imgL, imgR, disp_L):
    model.train()
    # print(disp_L, '\n')
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = disp_L * 1000
    # show (disp_L[0])
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()
        # disp_true= torch.reshape(disp_true, imgL.shape)
        # print (imgL.shape, '\n', imgR.shape, '\n', disp_true.shape, '\n')

    # imgL = imgL/imgL.amax()
    # imgR = imgR/imgR.amax()
    # disp_true = disp_true/disp_true.amax()

    # print(max(imgL), '\n')
    # print((disp_true/disp_true.amax()).tolist(), '\n')
    # show(disp_L)
    # # ---------
    mask = (disp_true > 0)
    #print(mask)
    mask.detach_()
    #print(mask.shape, '\n')

    # ----

    optimizer.zero_grad()

    # with autocast():
    #     if args.model == 'stackhourglass':
    #         output1, output2, output3 = model(imgL,imgR)
    #         output1 = torch.squeeze(output1,1)
    #         output2 = torch.squeeze(output2,1)
    #         output3 = torch.squeeze(output3,1)
    #         loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
    #     elif args.model == 'basic':
    #         output = model(imgL,imgR)
    #         output = torch.squeeze(output,1)
    #         loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    #         ## visualizing model output(prediction)
    #         if batch_idx == 0:
    #             output = output.cpu().detach().numpy()
    #             show(output[0])
    #             # show(output[1])


    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        #print('output.type', output1.type(), '\n')

        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        #print('output.size', output1.size(), '\n')
        loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)
    elif args.model == 'basic':
        output = model(imgL, imgR)
        output = torch.squeeze(output, 1)
        loss = F.smooth_l1_loss(
            output[mask], disp_true[mask], size_average=True)
        # print("disp_true: ",disp_true,'\n')
        # print("output: ", output,'\n')
        # print("loss: ", loss.tolist(),'\n')
  
        ## visualizing model output(prediction)
        if batch_idx == 0:
            output = output.cpu().detach().numpy()
            show(output[0])
            # show(output[1])



    return loss


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_true = disp_true * 1000
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output3 = torch.squeeze(output3)
    pred_disp = output3.data.cpu()
    # print(pred_disp.shape,'\n')
    # print(disp_true.shape,'\n')

    # show(pred_disp[0])
    # show(disp_true)
    #computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]]-pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
        disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]]*0.05)
    torch.cuda.empty_cache()

    return 1-(float(torch.sum(correct))/float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    lr = random.uniform(0.0001, 0.01)
    # if epoch <= 200:
    #     lr = 0.01
    # else:
    #     lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def show(img):
    plt.imshow(img, cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def main():

    max_acc = 0
    max_epo = 0
    start_full_time = time.time()
    gradient_accumulations = 10
    # scaler = GradScaler()

    for epoch in range(1, args.epochs+1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            if batch_idx == 0:
                show(disp_crop_L[0])
            # print('disp_crop_L', disp_crop_L.shape)
            loss = train(batch_idx, imgL_crop, imgR_crop, disp_crop_L)

            # scaler.scale(loss / gradient_accumulations).backward()

            # if (batch_idx + 1) % gradient_accumulations == 0:
            #     scaler.step(optimizer)
            #     scaler.update()
            #     model.zero_grad()


            (loss/gradient_accumulations).backward()
            if (batch_idx + 1) % gradient_accumulations == 0:
                optimizer.step()
                model.zero_grad()
                torch.cuda.empty_cache()
            print('Iter %d training loss = %.3f , time = %.2f' %
                  (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %
              (epoch, total_train_loss/len(TrainImgLoader)))

        ## Test ##

        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            # print('disp_L', disp_L.shape)
            test_loss = test(imgL, imgR, disp_L)
            print('Iter %d 3-px error in val = %.3f' %
                  (batch_idx, test_loss*100))
            total_test_loss += test_loss

        print('epoch %d total 3-px error in val = %.3f' %
              (epoch, total_test_loss/len(TestImgLoader)*100))
        if total_test_loss/len(TestImgLoader)*100 > max_acc:
            max_acc = total_test_loss/len(TestImgLoader)*100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # SAVE
        savefilename = args.savemodel+'finetune_'+str(epoch)+'.tar'
        print(savefilename)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss/len(TrainImgLoader),
            'test_loss': total_test_loss/len(TestImgLoader)*100,
        }, savefilename)

    print('full finetune time = %.2f HR' %
          ((time.time() - start_full_time)/3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':
    main()
