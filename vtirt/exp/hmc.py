import argparse
from dotmap import DotMap
import os
import json
from pprint import pprint
import time
import scipy.stats as stats
import numpy as np
import torch

from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.util import initialize_model
from torch.utils.tensorboard import SummaryWriter

from vtirt.data.wiener import Wiener2PLDataset
from vtirt.exp.utils import set_default_config, set_config_from_obj
from vtirt.model.irt import dirt_2pl
from vtirt.const import OUT_DIR, CONFIG_DIR

class ExpHMC:

    def __init__(self, config, device, out_dir):
        self.config = config
        self.device = device
        self.out_dir = out_dir

        exp_config = self.config.exp

        data_class = globals()[exp_config.data]
        train_dataset = data_class(split='train',
                                    **self.config.data)
        valid_dataset = data_class(split='valid',
                                    **self.config.data)
        self.get_data_params(train_dataset, valid_dataset)

        pprint(config.toDict())
        agent_config = self.config.agent
        set_default_config('num_chains', agent_config, 1)
        set_default_config('num_warmup', agent_config, 1)
        set_default_config('num_samples', agent_config, 4)
        set_config_from_obj('std_diff', train_dataset, agent_config)
        set_config_from_obj('std_disc', train_dataset, agent_config)
        set_config_from_obj('std_init', train_dataset, agent_config)
        set_config_from_obj('std_theta', train_dataset, agent_config)
        self.agent_config = agent_config

        self.model = globals()[exp_config.model]
        self.writer = SummaryWriter(log_dir=self.out_dir)

    def get_data_params(self, train_dataset, valid_dataset):
        device = self.device

        self.kmap = torch.BoolTensor(train_dataset.kmap).to(device)

        self.train_mask = torch.BoolTensor(train_dataset.mask).to(device)
        self.train_q_id = torch.LongTensor(train_dataset.q_id).to(device)
        self.train_resp = torch.FloatTensor(train_dataset.resp).to(device)

        self.valid_mask = torch.BoolTensor(valid_dataset.mask).to(device)
        self.valid_q_id = torch.LongTensor(valid_dataset.q_id).to(device)
        self.valid_resp = torch.FloatTensor(valid_dataset.resp).to(device)

        if hasattr(valid_dataset, 'trial_ability'):
            self.valid_trial_ab = (
                torch
                .FloatTensor(valid_dataset.trial_ability)
                .to(device)
            )
        else:
            self.valid_trial_ab = None

    def infer(self):
        post_samples = None

        init_params, potential_fn, transforms, _ = initialize_model(
            model=self.model,
            model_args=[
                self.kmap,
                self.agent_config.std_diff,
                self.agent_config.std_disc,
                self.agent_config.std_init,
                self.agent_config.std_theta,
                self.train_mask,
                self.train_q_id,
                self.train_resp,
                self.valid_mask,
                self.valid_q_id,
                self.valid_resp,
                self.device
            ],
            num_chains=self.agent_config.num_chains
        )

        def hook_fn(kernel, samples, stage, i):
            if stage == 'Warmup' or i % 5 != 0:
                return

            nonlocal post_samples
            if post_samples is None:
                post_samples = dict((k,v.unsqueeze(dim=0))
                                for k,v in samples.items())
            else:
                new_post_samples = {}
                for k,v in samples.items():
                    new_post_samples[k] = torch.cat(
                        [post_samples[k], v.unsqueeze(dim=0)]
                    )
                post_samples = new_post_samples
            self.valid(post_samples, i=i)

        nuts_kernel = NUTS(potential_fn=potential_fn)
        mcmc = MCMC(
            nuts_kernel,
            num_samples = self.agent_config.num_samples,
            warmup_steps = self.agent_config.num_warmup,
            num_chains = self.agent_config.num_chains,
            initial_params = init_params,
            hook_fn=hook_fn,
            transforms = transforms,
        )

        start = time.time()
        mcmc.run(
            self.kmap,
            self.agent_config.std_diff,
            self.agent_config.std_disc,
            self.agent_config.std_init,
            self.agent_config.std_theta,
            self.train_mask,
            self.train_q_id,
            self.train_resp,
            self.valid_mask,
            self.valid_q_id,
            self.valid_resp,
            self.device
        )
        end = time.time()

        samples = mcmc.get_samples()
        perf_dict = self.valid(samples)
        perf_dict['InferenceTime'] = end - start

        return perf_dict

    # TODO
    def valid(self, samples, i=None):
        valid_trial_ab = self.valid_trial_ab
        num_users, max_seq_len, _ = valid_trial_ab.shape

        inf_ab_lst, inf_true_ab_lst = [], []

        mask = self.valid_mask
        for t in range(max_seq_len):
            inf_ab = samples[f'inf_ability_{t+1}'].cpu().numpy().mean(0)
            inf_ab_lst.append(inf_ab)

            if valid_trial_ab is None:
                continue

            inf_true_ab = valid_trial_ab[:,t][mask[:,t]].cpu().numpy()
            inf_true_ab_lst.append(inf_true_ab)

        inf_ab = np.concatenate(inf_ab_lst)
        inf_true_ab = \
            np.concatenate(inf_true_ab_lst) if len(inf_true_ab_lst) > 0 else []

        if len(inf_true_ab) > 0:
            inf_ab_spr = stats.spearmanr(inf_ab, inf_true_ab).correlation
            inf_ab_prr = stats.pearsonr(inf_ab, inf_true_ab).statistic
            perf_dict = {}
            perf_dict['InferAbility/spearmanr'] = inf_ab_spr
            perf_dict['InferAbility/pearsonr'] = inf_ab_prr

        print()
        pprint(perf_dict)

        if i is None:
            perf_path = os.path.join(OUT_DIR, 'final_perf.json')
        else:
            perf_path = os.path.join(OUT_DIR, f'perf_{i}.json')

        with open(perf_path, 'w') as f:
            json.dump(perf_dict, f, indent=4)
        return perf_dict

    def run(self):
        self.infer()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--run-id', type=int, default=None)
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)

    device = 'cpu' if args.device == 'cpu' else f'cuda:{args.device}'

    config['path'] = args.config_path
    config = DotMap(config)

    base_dirname = config.path.split('configs/')[-1].split('.json')[0]
    base_dirname += '' if args.run_id is None else f'_{args.run_id}'
    out_dirname = os.path.join(OUT_DIR, base_dirname)

    exp = ExpHMC(config, device, out_dirname)
    exp.run()
