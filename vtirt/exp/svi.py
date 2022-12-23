from dotmap import DotMap
import json
import numpy as np
import scipy.metric as metric
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from pyro.infer import Trace_ELBO, SVI
from pyro.optim import Adam

from vtirt.data.utils import to_device
from vtirt.data.wiener import Wiener2PLDataset
from vtirt.model.vtirt import VTIRT

class ExpSVI:

    def __init__(self, config, device):
        self.config = config
        self.device = device

        exp_config = self.config.exp

        data_class = globals()[exp_config.data]
        self.train_dataset = data_class(split='train', **self.config.data)
        self.valid_dataset = data_class(split='valid', **self.config.data)

        self.agent = globals()[exp_config.model](**self.config.agent)


    def train(self):
        loader = DataLoader(
            self.train_dataset,
            **self.config.train.loader,
            shuffle=True,
            pin_memory=True
        )
        optim = Adam(**self.config.optim)

        for epoch in range(self.train.num_epochs):
            print(f'============ epoch {epoch} ============')
            for batch in tqdm(loader):
                to_device(batch, self.device)
                elbo = Trace_ELBO()
                svi = SVI(self.agent.model, self.agent.guide, optim, elbo)
                svi.step(batch['mask'], batch['q_id'], batch['kmap'], batch['resp'])
            self.valid()

    @torch.no_grad()
    def valid(self):
        self.model.eval()

        loader = DataLoader(
            self.valid_dataset,
            **self.config.valid.loader,
            shuffle=False,
            pin_memory=True
        )

        logits_lst, target_lst = [], []
        for batch in tqdm(loader):
            to_device(batch, self.device)
            logits, target = self.agent.pred_response(batch)
            logits_lst.append(logits)
            target_lst.append(target)

        logits = np.concatenate(logits)
        target = np.concatenate(target)

    def run(self):
        pass
