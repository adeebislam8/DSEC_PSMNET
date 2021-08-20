from datetime import time
from os import times
from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from scripts.dataset.representations import VoxelGrid
from scripts.utils.eventslicer import EventSlicer
import pandas as pd


class Sequence(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', delta_t_ms: int=50, num_bins: int=15):
        assert num_bins >= 1
        assert delta_t_ms <= 100, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.num_bins = num_bins

        # Set event representation
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=True)

        self.locations = ['left', 'right']

        # Save delta timestamp in ms
        self.delta_t_us = delta_t_ms * 1000

        # load disparity timestamps
        if self.mode == 'train':
            disp_dir = seq_path / 'disparity'
            assert disp_dir.is_dir()
            self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')

            # load disparity paths
            ev_disp_dir = disp_dir / 'event'
            assert ev_disp_dir.is_dir()
            disp_gt_pathstrings = list()
            for entry in ev_disp_dir.iterdir():
                assert str(entry.name).endswith('.png')
                disp_gt_pathstrings.append(str(entry))
            disp_gt_pathstrings.sort()
            self.disp_gt_pathstrings = disp_gt_pathstrings

            assert len(self.disp_gt_pathstrings) == self.timestamps.size

            # Remove first disparity path and corresponding timestamp.
            # This is necessary because we do not have events before the first disparity map.
            assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
            self.disp_gt_pathstrings.pop(0)
            self.timestamps = self.timestamps[1:]

        elif self.mode == 'test':
            disp_dir = seq_path / 'disparity'
            assert disp_dir.is_dir()
            disp_timestamps = str(disp_dir) + "/timestamps.csv"
            self.timestamps = np.loadtxt(disp_timestamps, dtype='int64', skiprows=1, delimiter=',')
            print(self.timestamps[0][0])


            # timestamps = []
            # with open(disp_timestamps) as f:
            #     next(f)
            #     for row in f:
            #         timestamps.append(row.split(',')[0])
            #         # print(row.split(',')[0])
                    #extract file name
            # self.timestamps = df["# timestamp_us"]
            # self.timestamps = timestamps
            # print(self.timestamps)
            # self.timestamps = np.loadtxt(disp_dir / 'timestamps.txt', dtype='int64')


        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]


        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        if self.mode == 'train':
            return len(self.disp_gt_pathstrings)
        else:
            return len(self.timestamps)

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        

        if self.mode == 'train':
            ts_end = self.timestamps[index]
            # ts_start should be fine (within the window as we removed the first disparity map)
            ts_start = ts_end - self.delta_t_us
            disp_gt_path = Path(self.disp_gt_pathstrings[index])
            file_index = int(disp_gt_path.stem)
            output = {
                'disparity_gt': self.get_disparity_map(disp_gt_path),
                'file_index': file_index,
            }

        if self.mode == 'test':
            ts_end = self.timestamps[index][0]
            ts_start = ts_end - self.delta_t_us
            file_index = int(self.timestamps[index][1])
            print(file_index)
            output = {
                'file_index': file_index,
            }


        for location in self.locations:
            
            event_data = self.event_slicers[location].get_events(ts_start, ts_end)

            p = event_data['p']
            t = event_data['t']
            x = event_data['x']
            y = event_data['y']

            xy_rect = self.rectify_events(x, y, location)
            x_rect = xy_rect[:, 0]
            y_rect = xy_rect[:, 1]

            event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
            # if self.mode == 'train':
            #     if 'representation' not in output:
            #         output['representation'] = dict()
            # elif self.mode == 'test':
            #     output = {
            #         'representation' : []
            #     }
                # output['representation'] = dict()
            if 'representation' not in output:
                output['representation'] = dict()

            output['representation'][location] = event_representation

        return output
