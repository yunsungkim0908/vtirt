import argparse
import os
import json
import shutil
import time
from dotmap import DotMap
from tqdm import tqdm, trange
from pprint import pprint
import numpy as np
import scipy.stats as stats

import torch
from torch.utils.data import DataLoader

from vtirt.data.utils import to_device
from vtirt.data.wiener import Wiener2PLDataset
from vtirt.const import OUT_DIR

def Phi(x):
    return (1 + torch.erf(x/np.sqrt(2)))/2

class ExpTSKIRT:

    def __init__(self, config, device, out_dir):
        self.config = config
        self.device = device
        self.out_dir = out_dir
        os.makedirs(out_dir)

        exp_config = self.config.exp

        data_class = globals()[exp_config.data]
        train_dataset = data_class(split='train',
                                    **self.config.data)
        valid_dataset = data_class(split='valid',
                                    **self.config.data)
        self.diff = train_dataset.diff
        self.disc = train_dataset.disc
        self.std_theta = train_dataset.std_theta

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def infer_ability(self, q_id, mask, resp, ability):
        device = self.device
        q_id = q_id.cpu().numpy()
        mask = mask.bool()

        nu = self.std_theta

        num_users, max_seq_len = resp.size()
        R = 11 # theta range size

        t_range = torch.arange(max_seq_len).to(device)
        # (1,T,1)
        t_dist = max_seq_len - 1 - t_range[None,...,None]

        # (U,T,1)
        a_q = torch.tensor(self.disc[q_id]).to(device)
        b_q = torch.tensor(self.diff[q_id]).to(device)

        # (U,T,R)
        a_tilde = a_q/torch.sqrt(1 + a_q**2*nu**2/t_dist)

        # (U,T,1)
        r = resp[...,None]

        def _find_map_theta(theta_min=None, theta_max=None):
            if theta_min is None or theta_max is None:
                theta_min = torch.zeros(num_users).to(device) - 5
                theta_max = torch.zeros(num_users).to(device) + 5

            # theta_min, theta_max: (U)

            # (U,R)
            theta = np.linspace(theta_min.cpu().numpy(),
                                theta_max.cpu().numpy(), R, axis=-1)
            theta = torch.tensor(theta).to(device)

            # (U,T,R)
            p_tilde = Phi(a_tilde*(theta[:,None,:] - b_q))
            p_tilde = p_tilde.clip(max=1-1e-8, min=1e-8)

            # (U,T,R)
            log_lik = r*torch.log(p_tilde) + (1-r)*torch.log(1-p_tilde)

            # (U,R)
            log_p_theta = -theta**2/(nu*max_seq_len)**2
            log_post = log_p_theta + log_lik.sum(dim=1)

            argmax = log_post.argmax(dim=-1).unsqueeze(-1)
            min_idx = (argmax-1).clip(min=0)
            max_idx = (argmax+1).clip(max=R-1)

            theta_min = torch.gather(theta,1,min_idx).squeeze(-1)
            theta_max = torch.gather(theta,1,max_idx).squeeze(-1)

            return theta_min, theta_max

        theta_min = theta_max = None
        # for _ in range(3):
        for _ in range(1):
            theta_min, theta_max = _find_map_theta(theta_min, theta_max)

        trial_theta = (theta_min + theta_max)/2

        pred_ab = trial_theta[mask].cpu().numpy()
        if ability is not None:
            true_ab = ability[mask].cpu().numpy()
        else:
            true_ab = None

        return pred_ab, true_ab

    def run(self):
        loader = DataLoader(
            self.valid_dataset,
            **self.config.valid.loader,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn
        )

        total_inf_time = 0.0
        pred_ab_lst, true_ab_lst = [], []
        for batch in tqdm(loader):
            batch = to_device(batch, self.device)
            mask = batch['mask']
            q_id = batch['q_id']
            resp = batch['resp']
            ability = batch['ability'] if 'ability' in batch else None

            _, max_seq_len = mask.size()
            for t in trange(max_seq_len):
                start = time.time()
                pred_ab, true_ab = self.infer_ability(
                    q_id[:,:t+1],
                    mask[:,t],
                    resp[:,:t+1],
                    ability[:,t,0] if ability is not None else None
                )
                end = time.time()
                total_inf_time += (end - start)
                pred_ab_lst.append(pred_ab)
                if true_ab is not None:
                    true_ab_lst.append(true_ab)

        pred_ab = np.concatenate(pred_ab_lst)
        true_ab = np.concatenate(true_ab_lst)

        perf_dict = {}
        perf_dict['inf_time'] = total_inf_time
        if len(true_ab_lst) > 0:
            ab_spr = stats.spearmanr(pred_ab, true_ab).correlation
            ab_prr = stats.pearsonr(pred_ab, true_ab).statistic
            perf_dict['PredAbility/spearmanr'] = ab_spr
            perf_dict['PredAbility/pearsonr'] = ab_prr
            # inference is identical to prediction for tskirt
            perf_dict['InferAbility/spearmanr'] = ab_spr
            perf_dict['InferAbility/pearsonr'] = ab_prr

        pprint(perf_dict)
        perf_filename = os.path.join(self.out_dir, 'perf.json')
        with open(perf_filename, 'w') as f:
            json.dump(perf_dict, f, indent=4)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--infer-only', action='store_true')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'

    config['path'] = args.config_path
    config = DotMap(config)

    base_dirname = config.path.split('configs/')[-1].split('.json')[0]
    out_dirname = os.path.join(OUT_DIR, base_dirname)

    if args.overwrite and os.path.isdir(out_dirname):
        print(f'overwriting {out_dirname}...')
        shutil.rmtree(out_dirname)

    if not os.path.isdir(out_dirname):
        exp = ExpTSKIRT(config, device, out_dirname)
        exp.run()
