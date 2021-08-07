import torch.utils.data as torch
from scripts import save_voxel_rep as sv

def dataloader(filepath):   
    disp = sv.get_disp(filepath)
    left_voxel = sv.get_left_voxel(filepath)
    right_voxel = sv.get_right_voxel(filepath)


    return