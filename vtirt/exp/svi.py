import argparse
from dotmap import DotMap
import os
import json
import numpy as np
import time
import sklearn.metrics as metrics
import scipy.stats as stats
from tqdm import tqdm
from pprint import pprint
import shutil

import torch
import torch.optim as torch_optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import pyro.infer as pyro_infer
import pyro.optim as pyro_optim

from vtirt.data.utils import to_device
from vtirt.data.wiener import Wiener2PLDataset
from vtirt.model.vtirt_multi_kc import VTIRTMultiKC
from vtirt.model.vtirt_single_kc import (
    VTIRTSingleKC, VTIRTSingleKCIndep, VTIRTSingleKCDirect, VIBOSingleKC
)
from vtirt.const import OUT_DIR, CONFIG_DIR

class ExpSVI:

    def __init__(self, config, device, out_dir, infer_only=False, resume_training=False):
        self.config = config
        self.device = device

        self.infer_only = infer_only
        exp_config = self.config.exp

        data_class = globals()[exp_config.data]
        self.train_dataset = data_class(split='train',
                                        **self.config.data)
        self.valid_dataset = data_class(split='valid',
                                        **self.config.data)

        def _set_agent_config_from_data(attr):
            if hasattr(self.train_dataset, attr):
                value = getattr(self.train_dataset, attr)
                setattr(self.config.agent, attr, value)

        # set these values from data in case of synthetic data
        _set_agent_config_from_data('num_ques')
        _set_agent_config_from_data('num_kcs')
        _set_agent_config_from_data('std_init')
        _set_agent_config_from_data('std_theta')
        _set_agent_config_from_data('std_diff')
        _set_agent_config_from_data('std_disc')
        _set_agent_config_from_data('item_char')

        pprint(config.toDict())

        self.out_dir = out_dir
        self.last_ckpt_path = os.path.join(self.out_dir, 'last.ckpt')
        self.agent = globals()[exp_config.model](**self.config.agent).to(device)
        if resume_training:
            ckpt = torch.load(self.last_ckpt_path, map_location=self.device)
            self.agent.load_state_dict(ckpt['state_dict'])
            self.start_epoch = ckpt['epoch'] + 1
            print(f'resuming training from epoch {self.start_epoch}...')
        else:
            self.start_epoch = 0
        self.writer = SummaryWriter(log_dir=self.out_dir)

    def train(self):
        loader = DataLoader(
            self.train_dataset,
            **self.config.train.loader,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn
        )
        optim = pyro_optim.Adam(self.config.optim)

        best_metric = float('-inf')
        best_perf_path = os.path.join(self.out_dir, 'best_perf.json')

        total_train_time = 0.0
        start_epoch = self.start_epoch
        end_epoch = start_epoch + self.config.train.num_epochs
        for epoch in range(start_epoch, end_epoch):
            print(f'============ epoch {epoch} ============')
            self.agent.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            for batch in tqdm(loader):
                batch = to_device(batch, self.device)
                elbo = pyro_infer.Trace_ELBO()
                svi = pyro_infer.SVI(
                    self.agent.model, self.agent.guide, optim, elbo
                )
                epoch_loss += svi.step(batch['mask'],
                                       batch['q_id'],
                                       batch['kmap'],
                                       batch['resp'])
            epoch_end = time.time()
            total_train_time += (epoch_end - epoch_start)

            print(f'loss: {epoch_loss}')
            self.writer.add_scalar('Loss', epoch_loss, epoch)

            perf_dict = self.valid(epoch)
            perf_dict['epoch'] = epoch
            perf_dict['total_train_time'] = total_train_time

            if 'NextResp/auroc' in perf_dict:
                main_metric = 'NextResp/auroc'
            else:
                main_metric = 'InferAbility/pearsonr'

            if perf_dict[main_metric] > best_metric:
                with open(best_perf_path, 'w') as f:
                    json.dump(perf_dict, f, indent=4)

            perf_path = os.path.join(self.out_dir, f'perf_{epoch}.json')
            with open(perf_path, 'w') as f:
                json.dump(perf_dict, f, indent=4)

            ckpt = {
                'epoch': epoch,
                'state_dict': self.agent.state_dict()
            }
            torch.save(ckpt, self.last_ckpt_path)

    @torch.no_grad()
    def valid(self, epoch, stochastic=True):
        self.agent.eval()

        perf_dict = {}
        def _add_metric(name, value):
            perf_dict[name] = value
            self.writer.add_scalar(name, value, epoch)

        loader = DataLoader(
            self.valid_dataset,
            **self.config.valid.loader,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.valid_dataset.collate_fn
        )

        logits_lst, target_lst, pred_ab_lst, true_ab_lst = [], [], [], []
        inf_ab_lst, inf_ab_true_lst = [], []
        inf_time, pred_time = 0, 0
        for batch in tqdm(loader):
            batch = to_device(batch, self.device)

            if not self.infer_only:
                pred_start = time.time()
                logits, target, pred_ab, true_ab \
                    = self.agent.pred_response(batch, stochastic=stochastic)
                pred_end = time.time()
                pred_time += (pred_end - pred_start)

                logits_lst.append(logits)
                target_lst.append(target)

                if 'ability' in batch:
                    pred_ab_lst.append(pred_ab)
                    true_ab_lst.append(true_ab)

            if 'ability' in batch:
                inf_start = time.time()
                inf_ab, inf_true_ab = self.agent.infer_ability(batch)
                inf_end = time.time()
                inf_time += (inf_end - inf_start)

                inf_ab_lst.append(inf_ab)
                inf_ab_true_lst.append(inf_true_ab)

        if not self.infer_only:
            logits = np.concatenate(logits_lst)
            target = np.concatenate(target_lst)

            auroc = metrics.roc_auc_score(target, logits)
            acc = metrics.accuracy_score(target, (logits > 0))

            _add_metric('NextResp/auroc', auroc)
            _add_metric('NextResp/acc', acc)

            if hasattr(self.valid_dataset, 'trial_ability'):
                true_ab = np.concatenate(true_ab_lst)
                pred_ab = np.concatenate(pred_ab_lst)

                ab_spr = stats.spearmanr(pred_ab, true_ab).correlation
                ab_prr = stats.pearsonr(pred_ab, true_ab).statistic
                _add_metric('PredAbility/spearmanr', ab_spr)
                _add_metric('PredAbility/pearsonr', ab_prr)
            perf_dict['pred_time'] = pred_time

        if hasattr(self.valid_dataset, 'trial_ability'):
            inf_ab = np.concatenate(inf_ab_lst)
            inf_true_ab = np.concatenate(inf_ab_true_lst)

            inf_ab_spr = stats.spearmanr(inf_ab, inf_true_ab).correlation
            inf_ab_prr = stats.pearsonr(inf_ab, inf_true_ab).statistic
            _add_metric('InferAbility/spearmanr', inf_ab_spr)
            _add_metric('InferAbility/pearsonr', inf_ab_prr)
            perf_dict['inf_time'] = inf_time

        if hasattr(self.valid_dataset, 'diff'):
            true_diff = self.valid_dataset.diff
            true_disc = self.valid_dataset.disc
            pred_diff, pred_disc = self.agent.get_item_features()

            diff_spr = stats.spearmanr(pred_diff, true_diff).correlation
            diff_prr = stats.pearsonr(pred_diff, true_diff).statistic[0]
            _add_metric('PredDiff/spearmanr', diff_spr)
            _add_metric('PredDiff/pearsonr', diff_prr)

            disc_spr = stats.spearmanr(pred_disc, true_disc).correlation
            disc_prr = stats.pearsonr(pred_disc, true_disc).statistic[0]
            _add_metric('PredDisc/spearmanr', disc_spr)
            _add_metric('PredDisc/pearsonr', disc_prr)

        pprint(perf_dict)
        return perf_dict

    def run(self):
        self.train()

class ExpMLE(ExpSVI):

    def train(self):
        loader = DataLoader(
            self.train_dataset,
            **self.config.train.loader,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn
        )
        optimizer = torch_optim.Adam(self.agent.parameters(),
                                     self.config.optim)

        for epoch in range(self.config.train.num_epochs):
            print(f'============ epoch {epoch} ============')
            self.agent.train()
            epoch_loss = 0.0
            for batch in tqdm(loader):
                to_device(batch, self.device)

                optimizer.zero_grads()
                loss = self.agent.get_loss(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f'loss: {epoch_loss}')
            self.writer.add_scalar('Loss', epoch_loss, epoch)
            self.valid(epoch)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--valid-once', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume-training', action='store_true')
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

    if args.overwrite and os.path.isdir(out_dirname):
        print(f'overwriting {out_dirname}...')
        shutil.rmtree(out_dirname)

    if not os.path.isdir(out_dirname) or args.resume_training:
        exp = ExpSVI(config, device, out_dirname, args.infer_only, args.resume_training)
        if args.valid_once:
            exp.valid(epoch=0, stochastic=False)
        else:
            exp.run()
