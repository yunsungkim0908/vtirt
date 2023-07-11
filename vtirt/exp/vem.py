import argparse
import os
import json
import time
import shutil
from dotmap import DotMap
from pprint import pprint
import scipy.stats as stats
import numpy as np
import torch
from copy import deepcopy

from torch.utils.tensorboard import SummaryWriter

from vtirt.data.wiener import Wiener2PLDataset
from vtirt.exp.utils import set_default_config, set_config_from_obj
from vtirt.const import OUT_DIR, CONFIG_DIR

class ExpVEM:

    def __init__(self, config, device, out_dir):
        self.config = config
        self.device = device
        self.out_dir = out_dir

        self.phi = stats.norm().pdf
        self.PHI = stats.norm().cdf

        exp_config = self.config.exp

        data_class = globals()[exp_config.data]
        train_dataset = data_class(split='train',
                                    **self.config.data)
        valid_dataset = data_class(split='valid',
                                    **self.config.data)
        self.get_data_params(train_dataset, valid_dataset)

        pprint(config.toDict())
        agent_config = self.config.agent
        set_default_config('num_iter', agent_config, 100)
        set_config_from_obj('std_diff', train_dataset, agent_config)
        set_config_from_obj('std_disc', train_dataset, agent_config)
        set_config_from_obj('std_theta', train_dataset, agent_config)
        set_config_from_obj('mu_diff', train_dataset, agent_config)
        set_config_from_obj('mu_disc', train_dataset, agent_config)
        set_config_from_obj('mu_theta', train_dataset, agent_config)
        set_config_from_obj('std_init', train_dataset, agent_config)
        set_config_from_obj('item_char', train_dataset, agent_config)
        self.agent_config = agent_config

        self.writer = SummaryWriter(log_dir=self.out_dir)

    def get_data_params(self, train_dataset, valid_dataset):
        device = self.device
        self.num_ques = train_dataset.num_ques

        self.kmap = torch.BoolTensor(train_dataset.kmap).to(device)
        self.diff = train_dataset.diff.squeeze(-1)
        self.disc = train_dataset.disc.squeeze(-1)

        self.mask = valid_dataset.mask
        self.q_id = valid_dataset.q_id
        self.resp = valid_dataset.resp

        self.num_train = 0
        # train_mask = train_dataset.mask
        # valid_mask = valid_dataset.mask
        # self.mask = np.concatenate([train_mask, valid_mask], axis=0)

        # train_q_id = train_dataset.q_id
        # valid_q_id = valid_dataset.q_id
        # self.q_id = np.concatenate([train_q_id, valid_q_id], axis=0)

        # train_resp = train_dataset.resp
        # valid_resp = valid_dataset.resp
        # self.resp = np.concatenate([train_resp, valid_resp], axis=0)

        # self.num_train = train_dataset.num_train
        self.num_valid = train_dataset.num_valid

        if hasattr(valid_dataset, 'trial_ability'):
            self.valid_trial_ab = valid_dataset.trial_ability
        else:
            self.valid_trial_ab = None

    def update_propensity(
            self,
            E_th_l_t,
            E_a_q,
            E_d_q,
    ):
        q_id = self.q_id
        resp = self.resp.astype(float)

        E_a_l_t = E_a_q[q_id]
        E_d_l_t = E_d_q[q_id]

        # == 1. update latent propensity ==

        m_l_t = E_a_l_t*(E_th_l_t - E_d_l_t)
        phi_m = self.phi(m_l_t)
        PHI_m = self.PHI(m_l_t)

        # update expectations
        E_r_l_t = m_l_t + resp*(phi_m/PHI_m) - (1-resp)*(phi_m/(1-PHI_m))

        return E_r_l_t

    def update_item_param(
            self,
            E_th_l_t,
            E_th2_l_t,
            E_a_q,
            E_a2_q,
            E_d_q,
            E_d2_q,
            E_r_l_t
    ):
        n_Q = self.num_ques
        n_L, T = self.mask.shape
        mask = self.mask.astype(float)
        q_id = self.q_id

        mu_disc = self.agent_config.mu_disc
        mu_diff = self.agent_config.mu_diff
        std_disc = self.agent_config.std_disc
        std_diff = self.agent_config.std_diff

        E_a_l_t = E_a_q[q_id]
        E_a2_l_t = E_a2_q[q_id]
        E_d_l_t = E_d_q[q_id]
        E_d2_l_t = E_d2_q[q_id]

        # disc
        # trial-wise variance and mu*var temrs
        prec_a_l_t = E_th2_l_t - 2*E_th_l_t*E_d_l_t + E_d2_l_t
        mu_prec_a_l_t = E_r_l_t*(E_th_l_t - E_d_l_t)

        assert not (prec_a_l_t < 0).any()
        sctr_prec_a_q = np.zeros((n_L, n_Q, T)).astype(float)
        sctr_mu_prec_a_q = np.zeros((n_L, n_Q, T)).astype(float)
        np.put_along_axis(sctr_prec_a_q, q_id[:,None,:], prec_a_l_t[:,None,:], axis=1)
        np.put_along_axis(sctr_mu_prec_a_q, q_id[:,None,:], mu_prec_a_l_t[:,None,:], axis=1)

        sctr_prec_a_q = sctr_prec_a_q*self.mask[:,None,:]
        sctr_mu_prec_a_q = sctr_mu_prec_a_q*self.mask[:,None,:]

        prec_a_q = sctr_prec_a_q.sum(axis=(0,2)) + 1/std_disc**2
        mu_prec_a_q = sctr_mu_prec_a_q.sum(axis=(0,2)) + mu_disc/std_disc**2
        # prec_a_q = np.full(n_Q, 1/std_disc**2)
        # mu_prec_a_q = np.full(n_Q, mu_disc/std_disc**2)
        # for q in range(n_Q):
        #     q_mask = (q_id == q)*mask
        #     prec_a_q[q] += (q_mask*prec_a_l_t).sum()
        #     mu_prec_a_q[q] += (q_mask*mu_prec_a_l_t).sum()
        mu_a_q = mu_prec_a_q / prec_a_q
        var_a_q = 1 / prec_a_q

        # diff
        prec_d_l_t = E_a2_l_t
        mu_prec_d_l_t = E_a2_l_t*E_th_l_t - E_a_l_t*E_r_l_t
        assert not (prec_d_l_t < 0).any()
        sctr_prec_d_q = np.zeros((n_L, n_Q, T)).astype(float)
        sctr_mu_prec_d_q = np.zeros((n_L, n_Q, T)).astype(float)
        np.put_along_axis(sctr_prec_d_q, q_id[:,None,:], prec_d_l_t[:,None,:], axis=1)
        np.put_along_axis(sctr_mu_prec_d_q, q_id[:,None,:], mu_prec_d_l_t[:,None,:], axis=1)

        sctr_prec_d_q = sctr_prec_d_q*self.mask[:,None,:]
        sctr_mu_prec_d_q = sctr_mu_prec_d_q*self.mask[:,None,:]

        prec_d_q = sctr_prec_d_q.sum(axis=(0,2)) + 1/std_diff**2
        mu_prec_d_q = sctr_mu_prec_d_q.sum(axis=(0,2)) + mu_diff/std_diff**2
        # prec_d_q = np.full(n_Q, 1/std_diff**2)
        # mu_prec_d_q = np.full(n_Q, mu_diff/std_diff**2)
        # for q in range(n_Q):
        #     q_mask = (q_id == q)*mask
        #     prec_d_q[q] += (q_mask*prec_d_l_t).sum()
        #     mu_prec_d_q[q] += (q_mask*mu_prec_d_l_t).sum()
        mu_d_q = mu_prec_d_q / prec_d_q
        var_d_q = 1 / prec_d_q

        # update expectations
        E_a_q = mu_a_q
        E_a2_q = var_a_q + mu_a_q**2
        E_d_q = mu_d_q
        E_d2_q = var_d_q + mu_d_q**2

        return E_a_q, E_a2_q, E_d_q, E_d2_q

    def update_ability_lgm(
            self,
            E_a_q,
            E_a2_q,
            E_d_q,
            E_r_l_t
    ):
        n_L, T = self.mask.shape
        E_a_l_t = E_a_q[self.q_id]
        E_a2_l_t = E_a2_q[self.q_id]
        E_d_l_t = E_d_q[self.q_id]

        s_theta = self.agent_config.std_theta
        # "potentials"
        pt_mu_l_t = (E_a_l_t*E_r_l_t + E_a2_l_t*E_d_l_t)/E_a2_l_t
        pt_std_l_t = 1/np.sqrt(E_a2_l_t)

        a_lst = [np.ones(n_L).astype(float)]
        b_lst = [np.zeros(n_L).astype(float)]
        for t in reversed(range(T)):
            mask_t = self.mask[:,t].astype(bool)

            mu_t = pt_mu_l_t[:,t][mask_t]
            std_t = pt_std_l_t[:,t][mask_t]

            a_t = a_lst[0][mask_t]
            b_t = b_lst[0][mask_t]

            a_t = 1/(1 + (s_theta/std_t)**2 + (1-a_t))
            b_t = a_t*(b_t + (s_theta/std_t)**2*mu_t)

            a_next = deepcopy(a_lst[0])
            b_next = deepcopy(b_lst[0])
            a_next[mask_t] = a_t
            b_next[mask_t] = b_t

            a_lst = [a_next] + a_lst
            b_lst = [b_next] + b_lst

        mu_theta_t = np.zeros(n_L).astype(float)
        std_theta_t = np.zeros(n_L).astype(float)
        mu_theta = []
        std_theta = []
        for t in range(T):
            mask_t = self.mask[:,t].astype(bool)
            a_t = a_lst[t]
            b_t = b_lst[t]
            prev_mu_theta_t = mu_theta_t[mask_t]

            mu_t = a_t*prev_mu_theta_t + b_t
            std_t = np.sqrt(std_theta_t**2
                            + s_theta**2*a_t)

            mu_theta_t = deepcopy(mu_theta_t)
            mu_theta_t[mask_t] = mu_t
            std_theta_t = deepcopy(std_theta_t)
            std_theta_t[mask_t] = std_t

            mu_theta.append(mu_theta_t)
            std_theta.append(std_theta_t)

        mu_theta = np.stack(mu_theta, axis=1)
        std_theta = np.stack(std_theta, axis=1)

        E_th_l_t = mu_theta
        E_th2_l_t = std_theta**2 + mu_theta**2
        return E_th_l_t, E_th2_l_t

    def update_ability(
            self,
            E_a_q,
            E_a2_q,
            E_d_q,
            E_r_l_t
    ):
        n_L, T = self.mask.shape
        E_a_l_t = E_a_q[self.q_id]
        E_a2_l_t = E_a2_q[self.q_id]
        E_d_l_t = E_d_q[self.q_id]

        # "potentials"
        pt_prec_l_t = np.sqrt(E_a2_l_t)
        pt_mu_l_t = (E_a_l_t*E_r_l_t + E_a2_l_t*E_d_l_t)/E_a2_l_t

        # (Imai et al. 2016) Equation 63
        prev_c = np.full(n_L, self.agent_config.mu_theta)
        prev_C = np.full(n_L, self.agent_config.std_theta**2)

        c_lst, C_lst, O_lst = [], [], []
        for t in range(T):
            mask_t = self.mask[:,t]

            y_t = pt_mu_l_t[:,t]
            b_t = pt_prec_l_t[:,t]

            curr_O = self.agent_config.std_theta**2 + prev_C
            O_t = curr_O[mask_t]
            S_t = b_t[mask_t]**2*O_t + 1
            K_t = y_t[mask_t]*O_t/S_t

            curr_c = deepcopy(prev_c)
            curr_C = deepcopy(prev_C)

            curr_c[mask_t] = prev_c[mask_t] + K_t*(y_t - b_t*prev_C[mask_t])
            curr_C[mask_t] = (1 - K_t*b_t)*O_t

            c_lst.append(curr_c)
            C_lst.append(curr_C)
            O_lst.append(curr_O)

            prev_c, prev_C = curr_c, curr_C

        c = np.stack(c_lst, axis=-1)
        C = np.stack(C_lst, axis=-1)
        O = np.stack(O_lst, axis=-1)

        # (Imai et al. 2016) Equation 64
        next_d = deepcopy(c[:,-1])
        next_D = deepcopy(C[:,-1])

        d_lst, D_lst = [next_d], [next_D]
        for t in reversed(range(T-1)):
            mask_t = self.mask[:,t]

            J_t = (C[:,t]/O[:,t+1])[mask_t]

            curr_d = deepcopy(next_d)
            curr_D = deepcopy(next_D)

            curr_d[mask_t] = c[:,t][mask_t] \
                + J_t*(next_d[mask_t] - c[:,t][mask_t])
            curr_D[mask_t] = C[:,t][mask_t] \
                + J_t**2*(next_d[mask_t] - O[:,t+1][mask_t])

            d_lst = [curr_d] + d_lst
            D_lst = [curr_D] + D_lst

        d = np.stack(d_lst, axis=-1)
        D = np.stack(D_lst, axis=-1)

        E_th_l_t = d
        E_th2_l_t = D + d**2

        return E_th_l_t, E_th2_l_t

    def run(self):

        E_th_l_t = np.zeros_like(self.mask)
        E_th2_l_t = np.ones_like(self.mask)
        E_a_q = np.full(self.num_ques, self.agent_config.mu_disc)
        E_a2_q = E_a_q**2
        E_d_q = np.full(self.num_ques, self.agent_config.mu_diff)
        E_d2_q = E_a_q**2

        total_inf_time = 0.0
        for i in range(self.agent_config.num_iter):
            start = time.time()
            E_r_l_t = self.update_propensity(E_th_l_t, E_a_q, E_d_q)
            prop_update = time.time()
            E_a_q, E_a2_q, E_d_q, E_d2_q = \
                self.update_item_param(
                    E_th_l_t,
                    E_th2_l_t,
                    E_a_q,
                    E_a2_q,
                    E_d_q,
                    E_d2_q,
                    E_r_l_t
                )
            item_update = time.time()
            E_th_l_t, E_th2_l_t = self.update_ability_lgm(
                E_a_q,
                E_a2_q,
                E_d_q,
                E_r_l_t
            )
            end = time.time()
            total_inf_time += (end - start)

            perf_dict = self.valid(i, E_th_l_t, E_a_q, E_d_q)
            perf_dict['total_inf_time'] = total_inf_time
            perf_dict['prop_update_time'] = prop_update - start
            perf_dict['item_update_time'] = item_update - prop_update
            perf_dict['ability_update_time'] = end - item_update

            print(f'==== iteration {i} ====')
            pprint(perf_dict)

            perf_path = os.path.join(self.out_dir, f'perf_{i}.json')
            with open(perf_path, 'w') as f:
                json.dump(perf_dict, f, indent=4)

    def valid(self, i, inf_ability, inf_disc, inf_diff):
        valid_ability_inf = inf_ability[self.num_train:,:]
        print(inf_ability.shape)

        perf_dict = {}
        def _add_metric(name, value):
            perf_dict[name] = value
            if i is not None:
                self.writer.add_scalar(name, value, i)

        if self.valid_trial_ab is not None:
            valid_mask = self.mask[self.num_train:,:]
            inf_ab = valid_ability_inf[valid_mask]
            inf_true_ab = self.valid_trial_ab[valid_mask].squeeze(-1)

            inf_ab_spr = stats.spearmanr(inf_ab, inf_true_ab).correlation
            inf_ab_prr = stats.pearsonr(inf_ab, inf_true_ab).statistic
            _add_metric('InferAbility/spearmanr', inf_ab_spr)
            _add_metric('InferAbility/pearsonr', inf_ab_prr)

        diff_spr = stats.spearmanr(inf_diff, self.diff).correlation
        diff_prr = stats.pearsonr(inf_diff, self.diff).statistic
        _add_metric('PredDiff/spearmanr', diff_spr)
        _add_metric('PredDiff/pearsonr', diff_prr)

        disc_spr = stats.spearmanr(inf_disc, self.disc).correlation
        disc_prr = stats.pearsonr(inf_disc, self.disc).statistic
        _add_metric('PredDisc/spearmanr', disc_spr)
        _add_metric('PredDisc/pearsonr', disc_prr)

        return perf_dict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--infer-only', action='store_true')
    parser.add_argument('--run-id', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
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

    if not os.path.isdir(out_dirname):
        exp = ExpVEM(config, device, out_dirname)
        exp.run()
    else:
        print(r'run already exists. aborting.')
