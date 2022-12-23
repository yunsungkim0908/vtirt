import pyro
import pyro.distributions as dist
from typing import Dict, List

import torch
import torch.nn as nn

class VTIRTMultiKC(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_ques,
            num_kcs=1,
            std_init=1,
            std_theta=1,
            std_diff=1,
            std_disc=1
    ):
        super().__init__()
        self.std_init = std_init
        self.std_theta = std_theta
        self.std_diff = std_diff
        self.std_disc = std_disc

        self.hidden_dim = hidden_dim
        self.num_ques = num_ques
        self.num_kcs = num_kcs

        self.device = None
        self.declare_modules()

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def declare_modules(self):
        self.diff_mu = nn.Embedding(self.num_ques, 1)
        self.diff_logvar = nn.Embedding(self.num_ques, 1)
        self.disc_mu = nn.Embedding(self.num_ques, 1)
        self.disc_logvar = nn.Embedding(self.num_ques, 1)

        self.ab_poten_enc = nn.Sequential(
            nn.Linear(3,self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
            nn.GELU()
        )

    def model(self, mask, q_id, kmap, resp=None):
        """
            mask: shape (U,T)
            q_id: shape (U,T)
            resp: shape (U,T)
            kmap: shape (Q,K)
        """
        device = self.device

        num_users, max_seq_len = q_id.size()
        num_ques, num_kcs = kmap.size()
        trial_kc_mask = kmap[q_id,:]

        zero_mean = torch.zeros((num_ques,)).to(device)
        unit_std = torch.ones((num_ques,)).to(device)
        diff = pyro.sample('diff_mu', dist.Normal(zero_mean,unit_std).to_event(1))
        disc = pyro.sample('disc_mu', dist.Normal(zero_mean,unit_std).to_event(1))

        curr_kc_ability = torch.zeros((num_users, num_kcs)).to(device)
        trial_ability_kc = []
        for t in range(max_seq_len):
            trial_kc_mask_t = trial_kc_mask[:,t]

            mu_t = (
                torch.masked_select(curr_kc_ability, trial_kc_mask_t)
            )
            std_t = self.std_theta * (
                torch.ones_like(mu_t).to(device)
            )
            ability_next = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_t, std_t).to_event(1)
            )

            curr_kc_ability = curr_kc_ability.masked_scatter(
                trial_kc_mask_t, ability_next
            )
            trial_ability_kc.append(curr_kc_ability)

        trial_ability_kc = torch.stack(trial_ability_kc, dim=1)

        trial_ability = (
            (trial_ability_kc*trial_kc_mask.float()).sum(dim=-1)
            /(trial_kc_mask.sum(dim=-1).clamp(min=1e-8))
        )

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)

        _ = pyro.sample(
            'resp',
            dist.Bernoulli(logits=trial_logits).mask(mask).to_event(2),
            obs=resp
        )

    def guide(self, mask, q_id, kmap, resp):
        pyro.module('diff_mu', self.diff_mu)
        pyro.module('diff_logvar', self.diff_logvar)
        pyro.module('disc_mu', self.disc_mu)
        pyro.module('disc_logvar', self.disc_logvar)
        pyro.module('ab_poten_enc', self.ab_poten_enc)

        device = self.device

        num_users, max_seq_len = q_id.size()
        num_ques, num_kcs = kmap.size()
        trial_kc_mask = kmap[q_id,:]

        qid_enum = torch.arange(num_ques).to(device)
        diff = pyro.sample(
            'diff_mu',
            dist.Normal(
                self.diff_mu(qid_enum).squeeze(-1),
                torch.exp(0.5*self.diff_logvar(qid_enum)).squeeze(-1)
            ).to_event(1)
        )
        disc = pyro.sample(
            'disc_mu',
            dist.Normal(
                self.disc_mu(qid_enum).squeeze(-1),
                torch.exp(0.5*self.disc_logvar(qid_enum)).squeeze(-1)
            ).to_event(1)
        )

        # U x T x K
        trial_diff_kc = diff[q_id,None].repeat(1,1,num_kcs).permute(1,0,2)
        trial_disc_kc = disc[q_id,None].repeat(1,1,num_kcs).permute(1,0,2)
        trial_resp_kc = resp[...,None].repeat(1,1,num_kcs).permute(1,0,2).float()

        # inp_flat is indexed time-first
        trial_diff_flat = trial_diff_kc[trial_kc_mask.permute(1,0,2)]
        trial_disc_flat = trial_disc_kc[trial_kc_mask.permute(1,0,2)]
        trial_resp_flat = trial_resp_kc[trial_kc_mask.permute(1,0,2)]

        trial_poten_inp_flat = torch.stack([trial_diff_flat,
                                            trial_disc_flat,
                                            trial_resp_flat], axis=-1)

        trial_ab_poten_flat = self.ab_poten_enc(trial_poten_inp_flat)

        # split flat potential
        split_size_tensor = trial_kc_mask.sum(dim=[0,2])
        split_size: List[int] = []
        for i in range(split_size_tensor.shape[0]):
            split_size.append(split_size_tensor[i])

        ab_poten_lst = torch.split(trial_ab_poten_flat, split_size)
        alpha_lst, beta_lst = self.get_posterior_ability_params(trial_kc_mask,
                                                                ab_poten_lst)
        self.sample_posterior_ability(trial_kc_mask,
                                      ab_poten_lst,
                                      alpha_lst,
                                      beta_lst)

        # trial_diff = diff[q_id]
        # trial_disc = disc[q_id]

        # trial_logits = trial_disc*(trial_ability - trial_diff)

        # _ = pyro.sample(
        #     'resp',
        #     dist.Bernoulli(logits=trial_logits).mask(mask),
        # )

    def get_posterior_ability_params(self, trial_kc_mask, ab_poten_lst):
        device = self.device
        lmda_theta = 1/self.std_theta**2
        num_users, max_seq_len, num_kcs = trial_kc_mask.size()

        # prepend alpha_t
        alpha_lst = [torch.tensor(0).to(device)]
        beta_lst = [torch.tensor(0).to(device)]
        for t in reversed(range(max_seq_len)):
            # lst[0] has parameters from one time step ahead
            mu_t = ab_poten_lst[t][:,0]
            logvar_t = ab_poten_lst[t][:,1].clamp(min=1e-8)
            lmda_t = torch.exp(-logvar_t)
            alpha_t = (
                (lmda_t + alpha_lst[0]*lmda_theta)
                /(lmda_theta + lmda_t + alpha_lst[0]*lmda_theta)
            )
            beta_t = (
                (lmda_t*mu_t + alpha_t*lmda_t*beta_lst[0])
                /(lmda_t + alpha_t*lmda_t)
            )

            alpha_lst = [alpha_t] + alpha_lst
            beta_lst = [beta_t] + beta_lst
        return alpha_lst, beta_lst

    def sample_posterior_ability(
            self,
            trial_kc_mask,
            ab_poten_lst,
            alpha_lst,
            beta_lst
    ):
        device = self.device
        std_theta = self.std_theta
        lmda_theta = 1/self.std_theta**2
        num_users, max_seq_len, num_kcs = trial_kc_mask.size()

        curr_kc_ability = torch.zeros((num_users, num_kcs)).to(device)
        trial_ability_kc = []
        for t in range(max_seq_len):
            mu_t = ab_poten_lst[t][:,0]
            logvar_t = ab_poten_lst[t][:,1].clamp(min=1e-8)
            lmda_t = torch.exp(-logvar_t)
            alpha_next = alpha_lst[t+1]
            beta_next = alpha_lst[t+1]

            trial_kc_mask_t = trial_kc_mask[:,t]

            ability_prev = (
                torch.masked_select(curr_kc_ability, trial_kc_mask_t)
            )

            std_tilde_t = std_theta*torch.sqrt(1-alpha_lst[t])
            mu_tilde_t = (
                (lmda_theta*ability_prev
                    + mu_t*lmda_t
                    + alpha_next*lmda_theta*beta_next)
                /(lmda_theta + lmda_t + alpha_next*lmda_theta)
            )

            ability_next = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_tilde_t, std_tilde_t).to_event(1)
            )

            curr_kc_ability = curr_kc_ability.masked_scatter(
                trial_kc_mask_t, ability_next
            )

            trial_ability_kc.append(curr_kc_ability)

        trial_ability_kc = torch.stack(trial_ability_kc, dim=1)

        trial_ability = (
            (trial_ability_kc*trial_kc_mask.float()).sum(dim=-1)
            /(trial_kc_mask.sum(dim=-1).clamp(min=1e-8))
        )
        return trial_ability

    def get_item_features(self):
        all_qids = torch.arange(self.num_ques).to(self.device)
        all_diffs = self.diff_mu[all_qids].cpu().detach().numpy()
        all_discs = self.diff_mu[all_qids].cpu().detach().numpy()
        return all_diffs, all_discs

    def get_ability_estimates(self):
        pass


if __name__=='__main__':
    from vtirt.data.wiener import Wiener2PLDataset
    from vtirt.data.utils import to_device
    from torch.utils.data import DataLoader
    vtirt = VTIRTMultiKC(8, 10).to('cuda:0')
    dataset = Wiener2PLDataset(8,2,10,5, overwrite=True)
    loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)
    batch = to_device(next(iter(loader)), 'cuda:0')
    vtirt.model(batch['mask'], batch['q_id'], batch['kmap'], None)
    vtirt.guide(batch['mask'], batch['q_id'], batch['kmap'], batch['resp'])

    from pyro.infer import Trace_ELBO, JitTrace_ELBO, TraceEnum_ELBO, JitTraceEnum_ELBO, SVI
    from pyro.optim import Adam

    pyro.clear_param_store()

    vtirt.guide(batch['mask'], batch['q_id'], batch['kmap'], batch['resp'])

    elbo = JitTrace_ELBO()
    svi = SVI(vtirt.model, vtirt.guide, Adam({'lr': 0.01}), elbo)
    for i in range(10):
        svi.step(batch['mask'], batch['q_id'], batch['kmap'], batch['resp'])

