import os
import shutil
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from vtirt.const import DATA_DIR

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

class Wiener2PLDataset(Dataset):
    DATA_ATTR_LIST = [
        'trial_ability',
        'trial_ques',
        'trial_diff',
        'trial_disc',
        'trial_p',
        'diff',
        'disc',
        'resp',
        'mask',
        'kmap'
    ]

    def __init__(
            self,
            num_train,
            num_valid,
            num_ques,
            traj_len,
            split='valid',
            std_init=0.125,
            std_theta=0.125,
            std_diff=1,
            std_disc=1,
            overwrite=False
    ):
        super().__init__()
        self.num_train = num_train
        self.num_valid = num_valid

        self.n_U = num_train + num_valid
        self.n_Q = num_ques
        self.traj_len = traj_len
        self.std_init = std_init
        self.std_theta = std_theta
        self.std_diff = std_diff
        self.std_disc = std_disc

        self.dirname = os.path.join(DATA_DIR, 'bin',
                                    self.get_dataset_str())

        if os.path.isdir(self.dirname) and overwrite:
            shutil.rmtree(self.dirname)

        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
            self.generate_and_store_dataset()
        else:
            self.load_dataset()

        self.split = split

    def get_dataset_str(self):
        return (
            f'Wiener2PL_Q{self.n_Q}_traj{self.traj_len}'
            f'_stdInit{self.std_init}_stdAb{self.std_theta}'
            f'_stdDiff{self.std_diff}_stdDisc{self.std_disc}'
            f'_train{self.num_train}_val{self.num_valid}'
        )

    def load_dataset(self):
        for attr in self.DATA_ATTR_LIST:
            path = os.path.join(self.dirname, f'{attr}.npy')
            setattr(self, attr, np.load(path))

    def generate_and_store_dataset(self):
        ab_init = np.random.randn(self.n_U, 1, 1)*self.std_init
        ab_delta = np.random.randn(self.n_U, self.traj_len-1, 1)*self.std_theta

        trial_ability = np.concatenate([ab_init, ab_delta], axis=1).cumsum(axis=0)

        rng = np.random.default_rng()
        questions = np.expand_dims(np.arange(self.n_Q), axis=0)
        questions = np.tile(questions, (self.n_U, 1))
        trial_ques = rng.permuted(questions, axis=1)[:,:self.traj_len]

        diff = np.random.randn(self.n_Q, 1)*self.std_diff
        disc = np.random.randn(self.n_Q, 1)*self.std_disc

        trial_diff = diff[trial_ques,:]
        trial_disc = disc[trial_ques,:]

        trial_p = sigmoid(trial_disc*(trial_ability - trial_diff)).squeeze(-1)

        resp = stats.bernoulli(trial_p).rvs()
        mask = np.ones_like(resp).astype(bool)
        kmap = np.ones((self.n_Q, 1)).astype(bool)

        for attr in self.DATA_ATTR_LIST:
            val = locals()[attr]
            setattr(self, attr, val)
            path = os.path.join(self.dirname, f'{attr}.npy')
            np.save(path, val)

    def __getitem__(self, idx):
        combined_idx = idx + (0 if self.split == 'train'
                                else self.num_train)
        out_dict = {
            'u_id': combined_idx,
            'q_id': torch.LongTensor(self.trial_ques[combined_idx]),
            'resp': torch.FloatTensor(self.resp[combined_idx]),
            'mask': torch.LongTensor(self.mask[combined_idx]),
            'kmap': torch.BoolTensor(self.kmap)
        }
        return out_dict

    @staticmethod
    def collate_fn(batch):
        out_dict = {
            'u_id': [b['u_id'] for b in batch],
            'q_id': torch.stack([b['q_id'] for b in batch], dim=0),
            'resp': torch.stack([b['resp'] for b in batch], dim=0),
            'mask': torch.stack([b['mask'] for b in batch], dim=0),
            'kmap': batch[0]['kmap']
        }
        return out_dict

    def __len__(self):
        return self.num_train if self.split == 'train' else self.num_valid

if __name__=='__main__':
    from torch.utils.data import DataLoader
    dataset = Wiener2PLDataset(10,8,4)
    loader = DataLoader(dataset, batch_size=7)
    from pprint import pprint
    pprint(next(iter(loader)))
