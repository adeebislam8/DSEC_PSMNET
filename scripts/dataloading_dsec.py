from pathlib import Path
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pyntcloud import PyntCloud
from dataset.provider import DatasetProvider
from dataset.visualization import disp_img_to_rgb_img, show_disp_overlay, show_image

def show_image():
    import argparse
    visualize = args.visualize
    dsec_dir = Path(args.dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetProvider(dsec_dir)
    train_dataset = dataset_provider.get_train_dataset()

    batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False)
    with torch.no_grad():
        for data in tqdm(train_loader):
            if batch_size == 1 and visualize:
                disp = data['disparity_gt'].numpy().squeeze()
                disp_img = disp_img_to_rgb_img(disp)
                if args.overlay:
                    left_voxel_grid = data['representation']['left'].squeeze()
                    ev_img = torch.sum(left_voxel_grid, axis=0).numpy()
                    ev_img = (ev_img/ev_img.max()*256).astype('uint8')
                    
                    show_disp_overlay(ev_img, disp_img, height=480, width=640)
                else:
                    show_image(disp_img)
                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dsec_dir', help='Path to DSEC dataset directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    parser.add_argument('--overlay', action='store_true', help='If visualizing, overlay disparity and voxel grid image')
    args = parser.parse_args()

    visualize = args.visualize
    dsec_dir = Path(args.dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetProvider(dsec_dir)
    train_dataset = dataset_provider.get_train_dataset()

    batch_size = 1
    num_workers = 0
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=False)
    with torch.no_grad():
        for data in tqdm(train_loader):
            if batch_size == 1 and visualize:
                disp = data['disparity_gt'].numpy().squeeze()
                disp_img = disp_img_to_rgb_img(disp)
                if args.overlay:
                    left_voxel_grid = data['representation']['left'].squeeze()


                    # left_voxel_grid = left_voxel_grid.cpu().detach().numpy()
                    # x = np.arange(left_voxel_grid.shape[0])[:, None, None]
                    # y = np.arange(left_voxel_grid.shape[1])[None, :, None]
                    # z = np.arange(left_voxel_grid.shape[2])[None, None, :]
                    # x, y, z = np.broadcast_arrays(x, y, z)
                    # c = np.tile(left_voxel_grid.ravel()[:, None], [1, 3])

                    # # Do the plotting in a single call.
                    # fig = plt.figure()
                    # ax = fig.gca(projection='3d')
                    # ax.scatter(x.ravel(),
                    #         y.ravel(),
                    #         z.ravel(),
                    #         c=c)
                    # left_voxel_grid = left_voxel_grid.tolist()
                    print(left_voxel_grid.shape)
                    # cloud = PyntCloud(left_voxel_grid)


                    # voxelgrid_id = cloud.add_structure("voxelgrid", n_x=128, n_y=128, n_z=128)
                    # voxelgrid = cloud.structures[voxelgrid_id]
                    # # cloud
                    # voxelgrid.plot(d=3, mode="density", cmap="hsv")
                    # o3d.visualization.draw_geometries([left_voxel_grid])
                    ev_img = torch.sum(left_voxel_grid, axis=0).numpy()
                    ev_img = (ev_img/ev_img.max()*256).astype('uint8')
                    show_disp_overlay(ev_img, disp_img, height=480, width=640)
                else:
                    show_image(disp_img)
