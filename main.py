from __future__ import print_function
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import os
import random
from numpy.core.numeric import NaN, indices
from numpy.core.shape_base import block
from scripts.dataset.provider import DatasetProvider
from numpy.lib.function_base import disp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import copy
from torch.cuda.amp import GradScaler, autocast
#from dataloader import listflowfile as lt
# from dataloader import KITTILoader as DA
# from dataloader import DSECloader as ls
from models import *
from scripts import save_voxel_rep as voxel

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int ,default=192,
                    help='maxium disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/SceneFlowData/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--savemodel', default='./',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(args.datapath)

dsec_dir = Path(args.datapath)
dataset_provider = DatasetProvider(dsec_dir)

batch_size = 1
num_workers = 0


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
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))

def train(batch_idx, imgL,imgR, disp_true):
    model.train()
    # print('args.cuda: ', args.cuda)
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda().unsqueeze(0), imgR.cuda().unsqueeze(0), disp_true.cuda().unsqueeze(0)
        # print(imgL.shape, imgR.shape, disp_true.shape)

    #---------
    mask = disp_true > 0
    # mask = disp_true < args.maxdisp
    mask.detach_()
    #----
    optimizer.zero_grad()
    with torch.autograd.detect_anomaly():
        with autocast(): 
            if args.model == 'stackhourglass':
                output1, output2, output3 = model(imgL,imgR)
                output1 = torch.squeeze(output1,1)
                output2 = torch.squeeze(output2,1)
                output3 = torch.squeeze(output3,1)
                loss = 0.5*F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7*F.smooth_l1_loss(output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True) 
            elif args.model == 'basic':
                output = model(imgL,imgR)
                output = torch.squeeze(output,1)
                loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

                if torch.isnan(loss):
                    print(loss)
                    print(output)
                    # show(output[0])
                    loss = None #prevent backpropogation
                    return loss

                ## visualizing model output(prediction)
                if batch_idx%100 == 0:
                    output = output.cpu().detach().numpy()
                    disp_true = disp_true.cpu()
                    show(disp_true[0])
                    show(output[0])

        if torch.isnan(loss.data):
            print("nan found\n")
            return loss.data
        else:
            loss.backward()
            optimizer.step()

            return loss.data
        # return loss

def test(imgL,imgR,disp_true):

    model.eval()
    disp_true = disp_true.unsqueeze(0)

    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    with torch.autograd.detect_anomaly():

        if args.cuda:
            imgL, imgR = imgL.cuda().unsqueeze(0), imgR.cuda().unsqueeze(0)
        with torch.no_grad():
            output3 = model(imgL, imgR)

            # output3 = output3.cpu().detach().numpy().squeeze(0)
            # disp_true = disp_true.cpu()
            # show(disp_true[0])
            # show(output3[0])



    pred_disp = output3.data.cpu()
    show_output = output3.squeeze().cpu().detach().numpy()
    show_disp_true = disp_true.cpu().squeeze()
    # print(disp_true.shape)
    # print(show_disp_true.shape)

    show(show_disp_true)
    show(show_output)
    pred_disp = pred_disp.squeeze(0)
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
    lr = 0.0001
    # lr = 0.0009 * (0.99**(epoch))
    # if epoch > 50:
    #     lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def show(img):
    # npimg = img.numpy()
    plt.imshow(img,cmap='gray')
    plt.show(block=False)
    plt.pause(3)
    plt.close()

def main():


    start_full_time = time.time()

    for epoch in range(0, args.epochs):

        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)
        train_dataset = dataset_provider.get_train_dataset()
        test_dataset = dataset_provider.get_val_dataset()

        TrainImgLoader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    drop_last=False)
        print("lenth of TrainImgLoader: ",len(TrainImgLoader), '\n')
        TestImgLoader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    drop_last=False)

        ## training ##
        # with torch.no_grad():
        # random.shuffle(TrainImgLoader)
        count = 0
        for batch_idx, data in enumerate(TrainImgLoader):
            # print(data['file_index'])

            start_time = time.time()
            disp = voxel.get_disp(data)
            disp = torch.from_numpy(disp)
            left_voxel = voxel.get_left_voxel(data)
            right_voxel = voxel.get_right_voxel(data)
            if torch.any(torch.isnan(disp)):
                print("disp has nan\n")
                next
            if torch.any(torch.isnan(left_voxel)):
                print("left_voxel has nan\n")
                next
            if torch.any(torch.isnan(right_voxel)):
                print("right_voxel has nan\n")
                next
            
            loss = train(batch_idx,left_voxel,right_voxel, disp)
            # print(type(loss))
            assert not torch.isnan(loss)
            if torch.isnan(loss) or (loss == None):
                print("moving to next image\n")
                disp = disp.unsqueeze[0]
                left_voxel = left_voxel.unsqueeze[0]
                right_voxel = right_voxel.unsqueeze[0]
                show(disp[0])
                show(left_voxel[0])
                show(right_voxel[0])
                next

            else:
                print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
                print("updating total_train_loss\n")
                total_train_loss += loss
                # break
                if count > 0:
                        break
                count += 1
            
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/count))

        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/count,
        }, savefilename)

        print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
        total_test_loss = 0
        count = 0
        for batch_idx, data in enumerate(TestImgLoader):
                disp = voxel.get_disp(data)
                disp = torch.from_numpy(disp)
                left_voxel = voxel.get_left_voxel(data)
                right_voxel = voxel.get_right_voxel(data)
                test_loss = test(left_voxel,right_voxel, disp)
                print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
                total_test_loss += test_loss
                if count > 50:
                    break
                count += 1

        print('total test loss = %.3f' %(total_test_loss/count))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
            'test_loss': total_test_loss/count,
        }, savefilename)

def main3():

    start_full_time = time.time()
    scaler = GradScaler()
    # batch_size = 4
    gradient_accumulations = 16
    model.zero_grad()

    for epoch in range(0, args.epochs):

        print('This is %d-th epoch' %(epoch))
        total_train_loss = 0
        adjust_learning_rate(optimizer,epoch)


        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            # print(disp_crop_L.shape)
            if batch_idx%50 == 0:
                show(disp_crop_L[0])
            loss = train(batch_idx, imgL_crop,imgR_crop, disp_crop_L)

            scaler.scale(loss / gradient_accumulations).backward()

            if (batch_idx + 1) % gradient_accumulations == 0:
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad()
            print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

        #SAVE
        savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
        }, savefilename)

    print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

    #------------- TEST ------------------------------------------------------------
    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL,imgR, disp_L)
            print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
            total_test_loss += test_loss

    print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
    #----------------------------------------------------------------------------------
    #SAVE test information
    savefilename = args.savemodel+'testinformation.tar'
    torch.save({
            'test_loss': total_test_loss/len(TestImgLoader),
        }, savefilename)


def main2():

	start_full_time = time.time()
	for epoch in range(0, args.epochs):
	   print('This is %d-th epoch' %(epoch))
	   total_train_loss = 0
	   adjust_learning_rate(optimizer,epoch)
       

	   ## training ##
	   for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
	     start_time = time.time()

	     loss = train(imgL_crop,imgR_crop, disp_crop_L)
	     print('Iter %d training loss = %.3f , time = %.2f' %(batch_idx, loss, time.time() - start_time))
	     total_train_loss += loss
	   print('epoch %d total training loss = %.3f' %(epoch, total_train_loss/len(TrainImgLoader)))

	   #SAVE
	   savefilename = args.savemodel+'/checkpoint_'+str(epoch)+'.tar'
	   torch.save({
		    'epoch': epoch,
		    'state_dict': model.state_dict(),
                    'train_loss': total_train_loss/len(TrainImgLoader),
		}, savefilename)

	print('full training time = %.2f HR' %((time.time() - start_full_time)/3600))

	#------------- TEST ------------------------------------------------------------
	total_test_loss = 0
	for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
	       test_loss = test(imgL,imgR, disp_L)
	       print('Iter %d test loss = %.3f' %(batch_idx, test_loss))
	       total_test_loss += test_loss

	print('total test loss = %.3f' %(total_test_loss/len(TestImgLoader)))
	#----------------------------------------------------------------------------------
	#SAVE test information
	savefilename = args.savemodel+'testinformation.tar'
	torch.save({
		    'test_loss': total_test_loss/len(TestImgLoader),
		}, savefilename)


if __name__ == '__main__':
   main()
    
