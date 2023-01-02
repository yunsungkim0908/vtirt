import pyro
import pyro.distributions as dist
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.init as init

from vtirt.model.irt import dirt_2pl

class VTIRTOld(nn.Module):
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

        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                init.constant_(m.bias.data, 0)

        self.apply(_init_weights)

    def model(self, mask, q_id, kmap, resp=None):
        dirt_2pl(kmap=kmap, std_theta=self.std_theta, device=self.device,
                train_mask=mask, train_q_id=q_id, train_resp=resp)

    def guide(self, mask, q_id, kmap, resp, stochastic=True):
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
        if stochastic:
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
        else:
            diff = self.diff_mu(qid_enum).squeeze(-1)
            disc = self.disc_mu(qid_enum).squeeze(-1)

        # T x U x K
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
        b_lst, c_lst = self.get_posterior_ability_params(trial_kc_mask,
                                                                ab_poten_lst)
        trial_ability_kc = self.sample_posterior_ability(trial_kc_mask,
                                      b_lst,
                                      c_lst)

        trial_ability = (
            (trial_ability_kc*trial_kc_mask.float()).sum(dim=-1)
            /(trial_kc_mask.sum(dim=-1).clamp(min=1e-8))
        )

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        last_ability_kc = trial_ability_kc[:,-1]
        return trial_logits, last_ability_kc

    def get_posterior_ability_params(self, trial_kc_mask, ab_poten_lst):
        device = self.device
        std_theta = self.std_theta
        num_users, max_seq_len, num_kcs = trial_kc_mask.size()

        # prepend alpha_t
        b_lst = [torch.tensor(1).to(device)]
        c_lst = [torch.tensor(0).to(device)]
        for t in reversed(range(max_seq_len)):
            # lst[0] has parameters from one time step ahead
            mu_t = ab_poten_lst[t][:,0]
            logvar_t = ab_poten_lst[t][:,1]
            std_t = torch.exp(0.5*logvar_t).clamp(min=1e-8)

            b_t = 1/(1 + (std_theta/std_t)**2 + (1-b_lst[0]))
            c_t = b_t*(c_lst[0] + (std_theta/std_t)**2*mu_t)

            b_lst = [b_t] + b_lst
            c_lst = [c_t] + c_lst

        return b_lst, c_lst

    def sample_posterior_ability(self, trial_kc_mask, b_lst, c_lst):
        device = self.device
        std_theta = self.std_theta
        std_theta = self.std_theta
        num_users, max_seq_len, num_kcs = trial_kc_mask.size()

        curr_kc_ability = torch.zeros((num_users, num_kcs)).to(device)
        trial_ability_kc = []
        for t in range(max_seq_len):
            b_t = b_lst[t]
            c_t = c_lst[t]

            trial_kc_mask_t = trial_kc_mask[:,t]

            ability_prev = (
                torch.masked_select(curr_kc_ability, trial_kc_mask_t)
            )

            std_tilde_t = std_theta*torch.sqrt(b_t)
            mu_tilde_t = b_t*ability_prev + c_t

            ability_next = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_tilde_t, std_tilde_t).to_event(1)
            )

            curr_kc_ability = curr_kc_ability.masked_scatter(
                trial_kc_mask_t, ability_next
            )

            trial_ability_kc.append(curr_kc_ability)

        trial_ability_kc = torch.stack(trial_ability_kc, dim=1)
        return trial_ability_kc

    def get_item_features(self):
        all_qids = torch.arange(self.num_ques).to(self.device)
        all_diffs = self.diff_mu(all_qids).squeeze(-1).cpu().detach().numpy()
        all_discs = self.disc_mu(all_qids).squeeze(-1).cpu().detach().numpy()
        return all_diffs, all_discs

    def pred_response(self, batch, stochastic=True):
        mask = batch['mask']
        q_id = batch['q_id']
        kmap = batch['kmap']
        resp = batch['resp']
        true_ability = batch['ability'] if 'ability' in batch else None

        num_learners, max_seq_len = mask.size()

        preds, target, pred_ab, true_ab = [], [], [], []
        for t in range(max_seq_len):
            logits, last_ability_kc = self.guide(
                mask[:,:t+1], q_id[:,:t+1], kmap, resp[:,:t+1], stochastic=False
            )
            mask_t = mask[:,t].bool()
            if true_ability is not None:
                mask_kc = kmap[q_id[:,t],:]*mask_t.unsqueeze(-1)
                pred_ab.append(
                    torch.masked_select(last_ability_kc, mask_kc)
                )
                true_ab.append(
                    torch.masked_select(true_ability[:,t], mask_kc)
                )
            if t < max_seq_len-1:
                target.append(torch.masked_select(resp[:,t], mask_t))
                preds.append(torch.masked_select(logits[:,t], mask_t))

        preds = torch.cat(preds).cpu().detach().numpy()
        target = torch.cat(target).cpu().detach().numpy()

        if true_ability is not None:
            pred_ab = torch.cat(pred_ab).cpu().detach().numpy()
            true_ab = torch.cat(true_ab).cpu().detach().numpy()
        else:
            pred_ab = true_ab = None

        return preds, target, pred_ab, true_ab


if __name__=='__main__':
    device = 'cuda:0'
    from vtirt.data.wiener import Wiener2PLDataset
    from vtirt.data.utils import to_device
    from torch.utils.data import DataLoader
    vtirt = VTIRTOld(8, 10).to(device)
    dataset = Wiener2PLDataset(8,2,10,5, overwrite=True)
    loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)
    batch = to_device(next(iter(loader)), device)
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

