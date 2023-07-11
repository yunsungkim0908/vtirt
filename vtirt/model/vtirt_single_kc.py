import math
import pyro
import pyro.distributions as dist
from typing import Dict, List

from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils

from vtirt.model.irt import dirt_2param

ITEM_CHAR_FUNC={
    'logistic': lambda x: torch.sigmoid(x),
    'normal': lambda x: (1 + torch.erf(x/math.sqrt(2)))/2
}

class VTIRTSingleKC(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_ques,
            num_kcs=1,
            std_init=1,
            std_theta=1,
            std_diff=1,
            std_disc=1,
            item_char='logistic'
    ):
        super().__init__()
        self.std_init = std_init
        self.std_theta = std_theta
        self.std_diff = std_diff
        self.std_disc = std_disc
        self.item_char = item_char

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
        )

        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                init.constant_(m.bias.data, 0)

        self.apply(_init_weights)

    def model(self, mask, q_id, kmap, resp=None):
        """
            mask: shape (U,T)
            q_id: shape (U,T)
            resp: shape (U,T)
            kmap: shape (Q,K)
        """
        device = self.device
        std_theta = self.std_theta

        num_users, max_seq_len = q_id.size()
        num_ques = self.num_ques

        zeros = torch.zeros((num_ques,)).to(device)
        ones = torch.ones((num_ques,)).to(device)
        diff = pyro.sample('diff_mu',
                           dist.Normal(zeros,ones*self.std_diff).to_event(1))
        disc = pyro.sample('disc_mu',
                           dist.Normal(ones,ones*self.std_disc).to_event(1))

        ability_t = torch.zeros((num_users,)).to(device)
        ability = []
        for t in pyro.markov(range(max_seq_len)):

            mask_t = mask[:,t].bool()
            ab_t = torch.masked_select(ability_t, mask_t)
            std_t = std_theta * torch.ones_like(ab_t).to(device)

            # update ability
            ab_t = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(ab_t, std_t).to_event(1)
            )
            ability_t = ability_t.masked_scatter(mask_t, ab_t)
            ability.append(ability_t)

        trial_ability = torch.stack(ability, dim=1)

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        trial_probs = ITEM_CHAR_FUNC[self.item_char](trial_logits)

        _ = pyro.sample(
            'resp',
            dist.Bernoulli(probs=trial_probs)
                .mask(mask.bool())
                .to_event(2),
            obs=resp
        )

    def guide(self, mask, q_id, kmap, resp, stochastic=True):
        diff, disc = self.sample_post_item_features(stochastic=stochastic)

        # U x T x 3
        pyro.module('ab_poten_enc', self.ab_poten_enc)
        trial_poten_inp = torch.stack([diff[q_id], disc[q_id], resp], dim=-1)
        trial_ab_poten = self.ab_poten_enc(trial_poten_inp)

        # split flat potential
        a_lst, b_lst = self.get_posterior_ability_params(
            trial_ab_poten, mask
        )
        trial_ability = self.sample_posterior_ability(
            mask, trial_ab_poten, a_lst, b_lst, stochastic=stochastic
        )

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        return trial_logits, trial_ability

    def sample_post_item_features(self, stochastic=True):
        pyro.module('diff_mu', self.diff_mu)
        pyro.module('diff_logvar', self.diff_logvar)
        pyro.module('disc_mu', self.disc_mu)
        pyro.module('disc_logvar', self.disc_logvar)

        device = self.device

        num_ques = self.num_ques

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

        return diff, disc

    def get_posterior_ability_params(self, trial_ab_poten, mask):
        device = self.device
        lmda_theta = 1/self.std_theta**2
        num_users, max_seq_len = mask.size()

        # prepend alpha_t
        a_lst = [torch.zeros(num_users).to(device)]
        b_lst = [torch.zeros(num_users).to(device)]
        for t in reversed(range(max_seq_len)):
            mask_t = mask[:,t].bool()
            mu_t = torch.masked_select(trial_ab_poten[:,t,0], mask_t)
            logvar_t = torch.masked_select(trial_ab_poten[:,t,1], mask_t)
            lmda_t = torch.exp(-logvar_t).clamp(max=1e32)

            a_t = torch.masked_select(a_lst[0], mask_t)
            b_t = torch.masked_select(b_lst[0], mask_t)

            # update a_t and b_t
            b_t = (
                (lmda_t*mu_t + a_t*lmda_theta*b_t)
                /
                (lmda_t + a_t*lmda_theta)
            )
            a_t = (
                (lmda_t + a_t*lmda_theta)
                /
                (lmda_theta + lmda_t + a_t*lmda_theta)
            )

            a_next = a_lst[0].masked_scatter(mask_t, a_t)
            b_next = b_lst[0].masked_scatter(mask_t, b_t)

            a_lst = [a_next] + a_lst
            b_lst = [b_next] + b_lst

        return a_lst, b_lst

    def sample_posterior_ability(self, mask, trial_ab_poten, a_lst, b_lst, stochastic=True):
        device = self.device
        std_theta = self.std_theta
        lmda_theta = 1/self.std_theta**1
        num_users, max_seq_len = mask.size()

        ability_t = torch.zeros(num_users).to(device)
        ability = []
        for t in pyro.markov(range(max_seq_len)):
            mask_t = mask[:,t].bool()
            mu_t = torch.masked_select(trial_ab_poten[:,t,0], mask_t)
            logvar_t = torch.masked_select(trial_ab_poten[:,t,1], mask_t)
            lmda_t = torch.exp(-logvar_t).clamp(max=1e32)

            mask_t = mask[:,t].bool()
            a_next = a_lst[t+1]
            b_next = b_lst[t+1]
            ab_t = torch.masked_select(ability_t, mask_t)

            std_tilde_t = std_theta*torch.sqrt(1-a_next)
            mu_tilde_t = (
                (lmda_theta*ab_t + lmda_t*mu_t + a_next*lmda_theta*b_next)
                /
                (lmda_theta + lmda_t + a_next*lmda_theta)
            )

            if stochastic or t == (max_seq_len - 1):
                ab_t = pyro.sample(
                    f'ability_{t+1}',
                    dist.Normal(mu_tilde_t, std_tilde_t).to_event(1)
                )
            else:
                ab_t = mu_tilde_t

            ability_t = ability_t.masked_scatter(mask_t, ab_t)
            if stochastic:
                ability.append(ability_t)

        if stochastic:
            trial_ability= torch.stack(ability, dim=1)
        else:
            trial_ability = ability_t.unsqueeze(1)
        return trial_ability

    def get_item_features(self):
        all_qids = torch.arange(self.num_ques).to(self.device)
        all_diffs = self.diff_mu(all_qids).squeeze(-1).cpu().detach().numpy()
        all_discs = self.disc_mu(all_qids).squeeze(-1).cpu().detach().numpy()
        return all_diffs, all_discs

    def infer_ability(self, batch, stochastic=False):
        if 'ability' not in batch:
            return None, None

        mask = batch['mask'].bool()
        q_id = batch['q_id']
        resp = batch['resp']
        true_ability = batch['ability']

        num_learners, max_seq_len = mask.size()

        logits, trial_ability = self.guide(mask, q_id, None, resp)

        infer_ab = trial_ability[mask].cpu().numpy()
        true_ab = true_ability[mask].squeeze(-1).cpu().numpy()

        return infer_ab, true_ab

    def pred_response(self, batch, stochastic=True):
        mask = batch['mask']
        q_id = batch['q_id']
        resp = batch['resp']
        true_ability = (
            batch['ability'].squeeze(-1) if 'ability' in batch else None
        )

        num_learners, max_seq_len = mask.size()

        preds, target, pred_ab, true_ab = [], [], [], []
        for t in trange(max_seq_len, leave=False):
            logits, trial_ability = self.guide(
                mask[:,:t+1], q_id[:,:t+1], None, resp[:,:t+1], stochastic=False
            )
            last_ability = trial_ability[:,-1]
            mask_t = mask[:,t].bool()
            if true_ability is not None:
                pred_ab.append(
                    torch.masked_select(last_ability, mask_t)
                )
                true_ab.append(
                    torch.masked_select(true_ability[:,t], mask_t)
                )
            if t != max_seq_len-1:
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

class VTIRTSingleKCIndep(VTIRTSingleKC):

    def guide(self, mask, q_id, kmap, resp, stochastic=True):
        diff, disc = self.sample_post_item_features(stochastic=stochastic)
        pyro.module('ab_poten_enc', self.ab_poten_enc)

        device = self.device

        num_users, max_seq_len = q_id.size()

        # U x T x 3
        trial_poten_inp = torch.stack([diff[q_id], disc[q_id], resp],
                                      dim=-1)
        trial_ab_poten = self.ab_poten_enc(trial_poten_inp)

        ability_t = torch.zeros(num_users).to(device)
        ability = []
        for t in range(max_seq_len):
            mask_t = mask[:,t]
            mu_t = trial_ab_poten[:,t,0][mask_t]
            logvar_t = trial_ab_poten[:,t,0][mask_t]
            std_t = torch.exp(0.5*logvar_t)[mask_t]

            # update ability
            ab_t = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_t, std_t).to_event(1)
            )
            ability_t = ability_t.masked_scatter(mask_t, ab_t)
            ability.append(ability_t)

        trial_ability = torch.stack(ability, dim=1)

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        return trial_logits, trial_ability

class VTIRTSingleKCDirect(VTIRTSingleKC):

    def __init__(self, encoder_type, **kwargs):
        self.encoder_type = encoder_type
        super().__init__(**kwargs)

    def declare_modules(self):
        self.diff_mu = nn.Embedding(self.num_ques, 1)
        self.diff_logvar = nn.Embedding(self.num_ques, 1)
        self.disc_mu = nn.Embedding(self.num_ques, 1)
        self.disc_logvar = nn.Embedding(self.num_ques, 1)

        if self.encoder_type == 'loc':
            self.post_param_enc = nn.Sequential(
                nn.Linear(3,self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 3),
            )
        elif self.encoder_type == 's2s':
            self.post_param_lstm = nn.LSTM(input_size=3,
                                          hidden_size=self.hidden_dim,
                                          num_layers=1,
                                          bidirectional=True,
                                          batch_first=True)
            self.post_param_enc = nn.Sequential(
                nn.Linear(2*self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 3)
            )
        else:
            raise NotImplementedError

        def _init_weights(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                init.xavier_normal_(m.weight.data)
            if isinstance(m, nn.Linear):
                init.constant_(m.bias.data, 0)

        self.apply(_init_weights)

    def guide(self, mask, q_id, kmap, resp, stochastic=True):
        diff, disc = self.sample_post_item_features(stochastic=stochastic)

        num_users, max_seq_len = q_id.size()

        pyro.module('post_param_enc', self.post_param_enc)
        # U x T x 3
        post_param_inp = torch.stack([diff[q_id], disc[q_id], resp],
                                      dim=-1)

        if self.encoder_type == 'loc':
            post_param = self.post_param_enc(post_param_inp)
        elif self.encoder_type == 's2s':
            lengths = mask.sum(dim=1).cpu()
            lstm_in = rnn_utils.pack_padded_sequence(
                post_param_inp,
                lengths,
                batch_first=True
            )
            lstm_out, _ = self.post_param_lstm(lstm_in)
            lstm_out, length = rnn_utils.pad_packed_sequence(
                lstm_out,
                batch_first=True
            )
            post_param = self.post_param_enc(lstm_out)
        else:
            raise NotImplementedError

        trial_ability = self.sample_posterior_ability(mask, post_param)

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        return trial_logits, trial_ability

    def sample_posterior_ability(self, mask, post_param):
        device = self.device
        num_users, max_seq_len = mask.size()

        scale = post_param[...,0]
        bias = post_param[...,1]
        std = torch.exp(0.5*post_param[...,2])

        ability_t = torch.zeros(num_users).to(device)
        ability = []
        for t in pyro.markov(range(max_seq_len)):
            mask_t = mask[:,t].bool()

            ab_t = ability_t.masked_select(mask_t)
            scale_t = scale[:,t].masked_select(mask_t)
            bias_t = bias[:,t].masked_select(mask_t)

            mu_t = scale_t*ab_t + bias_t
            std_t = std[:,t].masked_select(mask_t)

            ab_t = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_t, std_t).to_event(1)
            )

            ability_t = ability_t.masked_scatter(mask_t, ab_t)
            ability.append(ability_t)

        trial_ability= torch.stack(ability, dim=1)
        return trial_ability

class VTIRTSingleKCLinearDynamic(VTIRTSingleKC):

    def get_posterior_ability_params(self, trial_ab_poten, mask):
        device = self.device
        std_theta = self.std_theta
        num_users, max_seq_len = mask.size()

        # prepend alpha_t
        a_lst = [torch.ones(num_users).to(device)]
        b_lst = [torch.zeros(num_users).to(device)]
        for t in reversed(range(max_seq_len)):
            mask_t = mask[:,t].bool()
            mu_t = torch.masked_select(trial_ab_poten[:,t,0], mask_t)
            logvar_t = torch.masked_select(trial_ab_poten[:,t,1], mask_t)
            std_t = torch.exp(0.5*logvar_t).clamp(min=1e-8)

            a_t = torch.masked_select(a_lst[0], mask_t)
            b_t = torch.masked_select(b_lst[0], mask_t)

            # update a_t and b_t
            a_t = 1/(1 + (std_theta/std_t)**2 + (1-a_t))
            b_t = a_t*(b_t + (std_theta/std_t)**2*mu_t)

            a_next = a_lst[0].masked_scatter(mask_t, a_t)
            b_next = b_lst[0].masked_scatter(mask_t, b_t)

            a_lst = [a_next] + a_lst
            b_lst = [b_next] + b_lst

        return a_lst, b_lst

    def sample_posterior_ability(self, mask, trial_ab_poten, a_lst, b_lst):
        device = self.device
        std_theta = self.std_theta
        num_users, max_seq_len = mask.size()

        ability_t = torch.zeros(num_users).to(device)
        ability = []
        for t in pyro.markov(range(max_seq_len)):
            mask_t = mask[:,t].bool()
            a_t = a_lst[t]
            b_t = b_lst[t]
            ab_t = torch.masked_select(ability_t, mask_t)

            std_tilde_t = std_theta*torch.sqrt(a_t)
            mu_tilde_t = a_t*ab_t + b_t

            ab_t = pyro.sample(
                f'ability_{t+1}',
                dist.Normal(mu_tilde_t, std_tilde_t).to_event(1)
            )

            ability_t = ability_t.masked_scatter(mask_t, ab_t)
            ability.append(ability_t)

        trial_ability= torch.stack(ability, dim=1)
        return trial_ability

class VIBOSingleKC(VTIRTSingleKC):

    def model(self, mask, q_id, kmap, resp=None):
        """
            mask: shape (U,T)
            q_id: shape (U,T)
            resp: shape (U,T)
            kmap: shape (Q,K)
        """
        device = self.device

        num_users, max_seq_len = q_id.size()
        num_ques = self.num_ques

        zeros = torch.zeros((num_ques,)).to(device)
        ones = torch.ones((num_ques,)).to(device)
        diff = pyro.sample('diff_mu',
                           dist.Normal(zeros,ones*self.std_diff).to_event(1))
        disc = pyro.sample('disc_mu',
                           dist.Normal(ones,ones*self.std_disc).to_event(1))

        trial_ability = pyro.sample(
            'ability', dist.Normal(
                torch.zeros(num_users).to(device),
                torch.ones(num_users).to(device)*self.std_theta
            ).to_event(1)
        ).unsqueeze(-1)

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)
        trial_probs = ITEM_CHAR_FUNC[self.item_char](trial_logits)

        _ = pyro.sample(
            'resp',
            dist.Bernoulli(probs=trial_probs)
                .mask(mask.bool())
                .to_event(2),
            obs=resp
        )

    def guide(self, mask, q_id, kmap, resp=None, stochastic=True):
        diff, disc = self.sample_post_item_features(stochastic=stochastic)

        _, max_seq_len = mask.size()

        pyro.module('ab_poten_enc', self.ab_poten_enc)
        trial_poten_inp = torch.stack([diff[q_id], disc[q_id], resp], dim=-1)
        trial_ab_poten = self.ab_poten_enc(trial_poten_inp)

        ab_mu, ab_logvar = product_of_experts(
            mu=trial_ab_poten[...,0],
            logvar=trial_ab_poten[...,1],
            mask=mask,
            expert_dim=1
        )
        ab_std = torch.exp(0.5*ab_logvar)
        trial_ability = pyro.sample(
            'ability', dist.Normal(ab_mu, ab_std).to_event(1)
        ).unsqueeze(-1).repeat(1,max_seq_len)

        trial_diff = diff[q_id]
        trial_disc = disc[q_id]

        trial_logits = trial_disc*(trial_ability - trial_diff)

        return trial_logits, trial_ability


def product_of_experts(
        mu,
        logvar,
        mask,
        total_experts=None,
        p_mu=0,
        p_logvar=0,
        expert_dim=0,
        eps=1e-8
):
    if total_experts is None:
        total_experts = mask.size(expert_dim)

    dim = expert_dim
    num_p = total_experts - mask.sum(dim, keepdims=False)

    # precision of i-th Gaussian expert at point x
    prec = 1 / torch.exp(logvar) + eps
    p_prec = 1 / math.exp(p_logvar)

    pd_mu = (
        torch.sum(mu*prec*mask, dim=dim) + num_p*p_mu*p_prec
        /
        torch.sum(prec*mask, dim=dim) + num_p*p_prec
    )
    pd_var = 1 / (torch.sum(prec*mask, dim=dim) + num_p*p_prec)
    pd_logvar = torch.log(pd_var)

    return pd_mu, pd_logvar

if __name__=='__main__':
    device = 'cuda:0'
    from vtirt.data.wiener import Wiener2PLDataset
    from vtirt.data.utils import to_device
    from torch.utils.data import DataLoader
    vtirt = VTIRTSingleKC(8, 10).to(device)
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

