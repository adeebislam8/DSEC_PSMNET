from os import SEEK_CUR
from pathlib import Path

import torch
import random
from scripts.dataset.sequence import Sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=15):
        train_path = dataset_path / 'train'
        test_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)
        assert test_path.is_dir(), str(test_path)


        sequences = list()
        image = train_path.iterdir()
        test_sequences = list()
        test_image = test_path.iterdir()
        for child in image:
            sequences.append(Sequence(child, 'train', delta_t_ms, num_bins))
            
        for child in test_image:
            test_sequences.append(Sequence(child, 'test', delta_t_ms, num_bins))

        random.shuffle(sequences) 
        self.train_dataset = torch.utils.data.ConcatDataset(sequences)
        # print(len(self.train_dataset[0:100]))

        random.shuffle(sequences)
        self.val_dataset = torch.utils.data.ConcatDataset(sequences)

        self.test_dataset = torch.utils.data.ConcatDataset(test_sequences)


    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        # Implement this according to your needs.
        # raise NotImplementedError
        return self.test_dataset
