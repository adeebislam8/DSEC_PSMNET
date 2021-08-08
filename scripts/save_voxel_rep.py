from pathlib import Path
import os

from numpy.core.fromnumeric import shape, size
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.dataset.provider import DatasetProvider
from scripts.dataset.visualization import disp_img_to_rgb_img, show_disp_overlay, show_image



def get_disp(data):
    with torch.no_grad():
        disp = data['disparity_gt'].numpy().squeeze()
    return disp


def get_left_voxel(data):
    with torch.no_grad():
        left_voxel_grid = data['representation']['left'].squeeze()
    return left_voxel_grid

def get_right_voxel(data):
    with torch.no_grad():
        right_voxel_grid = data['representation']['right'].squeeze()
    return right_voxel_grid







if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsec_dir', help='Path to DSEC dataset directory')
    parser.add_argument('--voxel_output_path', help='Path to write voxel image')
    args = parser.parse_args()

    dsec_dir = Path(args.dsec_dir)
    voxel_output_dir = Path(args.voxel_output_path)
    voxel_output_dir.mkdir(parents=True, exist_ok=True)
    #print (type(voxel_output_dir))
    left_voxel_dir = Path(voxel_output_dir, "left")
    #print (left_voxel_dir)
    left_voxel_dir.mkdir(parents=True, exist_ok=True)
    #os.path.join(output_file, "left") 
    right_voxel_dir = Path(voxel_output_dir, "right")
    right_voxel_dir.mkdir(parents=True, exist_ok=True)
    disp_dir = Path(voxel_output_dir, "disp")
    disp_dir.mkdir(parents=True, exist_ok=True)

    #os.path.join(output_file, "right")
    assert dsec_dir.is_dir()
    assert voxel_output_dir.is_dir() 
    assert left_voxel_dir.is_dir() 
    assert right_voxel_dir.is_dir()

    dataset_provider = DatasetProvider(dsec_dir)
    train_dataset = dataset_provider.get_train_dataset()

    batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False)

    with torch.no_grad():
        count = 0
        for data in tqdm(train_loader):
            disp = data['disparity_gt'].numpy().squeeze()
            disp  = disp/256.0
            # print(disp.type)
            # valid[y,x] = I[y,x]>0;
            # show_image(disp)
            # print (disp.tolist(), '\n')
            # print (disp.shape)
            # #disp_img = disp_img_to_rgb_img(disp)
            # disp_filepath = Path(disp_dir,str(count)).with_suffix('.png')
            # cv2.imwrite(str(disp_filepath), disp)



            left_voxel_grid = data['representation']['left'].squeeze()
            # ev_img = torch.sum(left_voxel_grid, axis=0).numpy()
            # ev_img = (ev_img/ev_img.max()*256).astype('uint8')
            # left_voxel_filepath = Path(left_voxel_dir,str(count)).with_suffix('.png')
            # #show_image(ev_img) ##This is what voxel grid looks like
            # cv2.imwrite(str(left_voxel_filepath), ev_img)

            right_voxel_grid = data['representation']['right'].squeeze()
            # ev_img = torch.sum(right_voxel_grid, axis=0).numpy()
            # ev_img = (ev_img/ev_img.max()*256).astype('uint8')
            # right_voxel_filepath = Path(right_voxel_dir,str(count)).with_suffix('.png')
            # cv2.imwrite(str(right_voxel_filepath), ev_img)
            # count += 1
            #show_image(ev_img) ##This is what voxel grid looks like
            
            #show_disp_overlay(ev_img, disp_img, height=480, width=640)
            