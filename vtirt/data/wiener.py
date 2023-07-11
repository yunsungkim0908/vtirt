import os
import shutil
import numpy as np
import scipy.stats as stats
from scipy import special

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from vtirt.const import DATA_DIR

ITEM_CHAR_FUNC={
    'logistic': lambda x: 1 / (1 + np.exp(-x)),
    'normal': lambda x: (1 + special.erf(x/np.sqrt(2)))/2,
}

class Wiener2PLDataset(Dataset):
    DATA_ATTR_LIST = [
        'trial_logits',
        'trial_ability',
        'q_id',
        'trial_diff',
        'trial_disc',
        'trial_probs',
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
            std_init=0.25,
            std_theta=0.25,
            std_diff=1,
            std_disc=1,
            overwrite=False,
            item_char='logistic',
            data_no=0,
    ):
        super().__init__()
        self.num_train = num_train
        self.num_valid = num_valid

        self.n_U = num_train + num_valid
        self.num_ques = num_ques
        self.traj_len = traj_len
        self.std_init = float(std_init)
        self.std_theta = float(std_theta)
        self.std_diff = float(std_diff)
        self.std_disc = float(std_disc)
        self.mu_diff = 0.
        self.mu_disc = 1.
        self.mu_theta = 1.
        self.num_kcs = 1
        self.item_char = item_char
        self.data_no = data_no

        self.dirname = os.path.join(DATA_DIR, 'bin',
                                    self.get_dataset_str())

        if os.path.isdir(self.dirname) and overwrite:
            shutil.rmtree(self.dirname)

        if not os.path.isdir(self.dirname):
            os.makedirs(self.dirname)
            print(f'generating data: {self.dirname}')
            self.generate_and_store_full_dataset()
        else:
            print(f'loading data: {self.dirname}')
            self.load_full_dataset()

        self.split = split
        self.get_split()

    def get_dataset_str(self):
        return (
            f'Wiener_{self.item_char}_Q{self.num_ques}'
            f'_traj{self.traj_len}'
            f'_stdInit{self.std_init}_stdAb{self.std_theta}'
            f'_stdDiff{self.std_diff}_stdDisc{self.std_disc}'
            f'_train{self.num_train}_val{self.num_valid}'
            f'_no{self.data_no}'
        )

    def load_full_dataset(self):
        for attr in self.DATA_ATTR_LIST:
            path = os.path.join(self.dirname, f'{attr}.npy')
            setattr(self, attr, np.load(path))

    def get_split(self):
        if self.split == 'train':
            self.trial_logits = self.trial_logits[:self.num_train]
            self.trial_ability = self.trial_ability[:self.num_train]
            self.q_id = self.q_id[:self.num_train]
            self.trial_diff = self.trial_diff[:self.num_train]
            self.trial_disc = self.trial_disc[:self.num_train]
            self.trial_probs = self.trial_probs[:self.num_train]
            self.resp = self.resp[:self.num_train]
            self.mask = self.mask[:self.num_train]
        else:
            self.trial_logits = self.trial_logits[self.num_train:]
            self.trial_ability = self.trial_ability[self.num_train:]
            self.q_id = self.q_id[self.num_train:]
            self.trial_diff = self.trial_diff[self.num_train:]
            self.trial_disc = self.trial_disc[self.num_train:]
            self.trial_probs = self.trial_probs[self.num_train:]
            self.resp = self.resp[self.num_train:]
            self.mask = self.mask[self.num_train:]

    def generate_and_store_full_dataset(self):
        ab_init = np.random.randn(self.n_U, 1, 1)*self.std_init
        ab_delta = np.random.randn(self.n_U, self.traj_len-1, 1)*self.std_theta

        trial_ability = np.concatenate([ab_init, ab_delta], axis=1).cumsum(axis=1)

        rng = np.random.default_rng()
        questions = np.expand_dims(np.arange(self.num_ques), axis=0)
        questions = np.tile(questions, (self.n_U, 1))
        q_id = rng.permuted(questions, axis=1)[:,:self.traj_len]

        diff = np.random.randn(self.num_ques, 1)*self.std_diff
        disc = 1 + np.random.randn(self.num_ques, 1)*self.std_disc

        trial_diff = diff[q_id,:]
        trial_disc = disc[q_id,:]

        item_char_f = ITEM_CHAR_FUNC[self.item_char]
        trial_logits = trial_disc*(trial_ability - trial_diff)
        trial_probs = item_char_f(trial_logits).squeeze(-1)

        resp = stats.bernoulli(trial_probs).rvs()
        mask = np.ones_like(resp).astype(bool)
        kmap = np.ones((self.num_ques, 1)).astype(bool)

        for attr in self.DATA_ATTR_LIST:
            val = locals()[attr]
            setattr(self, attr, val)
            path = os.path.join(self.dirname, f'{attr}.npy')
            np.save(path, val)

    def __getitem__(self, idx):
        ability = torch.FloatTensor(self.trial_ability[idx])
        combined_idx = (0 if self.split == 'train' else self.num_train) + idx
        out_dict = {
            'u_id': combined_idx,
            'q_id': torch.LongTensor(self.q_id[idx]),
            'resp': torch.FloatTensor(self.resp[idx]),
            'mask': torch.LongTensor(self.mask[idx]),
            'kmap': torch.BoolTensor(self.kmap),
            'ability': ability
        }
        return out_dict

    @staticmethod
    def collate_fn(batch):
        out_dict = {
            'u_id': [b['u_id'] for b in batch],
            'q_id': torch.stack([b['q_id'] for b in batch], dim=0),
            'resp': torch.stack([b['resp'] for b in batch], dim=0),
            'mask': torch.stack([b['mask'] for b in batch], dim=0),
            'kmap': batch[0]['kmap'],
            'ability': torch.stack([b['ability'] for b in batch], dim=0)
        }
        return out_dict

    def __len__(self):
        return self.resp.shape[0]

if __name__=='__main__':
    from torch.utils.data import DataLoader
    dataset = Wiener2PLDataset(10,8,4)
    loader = DataLoader(dataset, batch_size=7)
    from pprint import pprint
    pprint(next(iter(loader)))
