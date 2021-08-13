from pathlib import Path

import torch

from scripts.dataset.sequence import Sequence

class DatasetProvider:
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=15):
        train_path = dataset_path / 'train'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)

        train_sequences = list()
        val_sequences = list()
        image = train_path.iterdir()
        # train = image[:180]
        # val = image[180:]
        for child in image:
            train_sequences.append(Sequence(child, 'train', delta_t_ms, num_bins))
            
        
        # for child in image:
        #     val_sequences.append(Sequence(child, 'val', delta_t_ms, num_bins))
        # print("len train: \n", len(train_sequences))
        self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
        # self.val_dataset = torch.utils.data.ConcatDataset(train_sequences[80:])


    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise self.val_dataset

    def get_test_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
