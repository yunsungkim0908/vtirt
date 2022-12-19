import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn

class VTIRT(nn.Module):
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

        pyro.module('diff_mu', self.diff_mu)
        pyro.module('diff_logvar', self.diff_logvar)
        pyro.module('disc_mu', self.disc_mu)
        pyro.module('disc_logvar', self.disc_logvar)
        pyro.module('ab_poten_enc', self.ab_poten_enc)

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
        diff = pyro.sample('diff_mu', dist.Normal(zero_mean,unit_std))
        disc = pyro.sample('disc_mu', dist.Normal(zero_mean,unit_std))

        curr_kc_ability = torch.zeros((num_users, num_kcs)).to(device)
        trial_ability_kc = []
        for i in range(max_seq_len):
            trial_kc_mask_t = trial_kc_mask[:,i]
            curr_ability_kc = torch.masked_select(curr_kc_ability, trial_kc_mask_t)
            next_ability_kc_std = self.std_theta * (
                torch.ones_like(curr_ability_kc).to(device)
            )
            next_ability_kc = pyro.sample(
                f'ability_{i+1}',
                dist.Normal(curr_ability_kc, next_ability_kc_std)
            )
            curr_kc_ability = curr_kc_ability.masked_scatter(
                trial_kc_mask_t, next_ability_kc
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
            dist.Bernoulli(logits=trial_logits).mask(mask),
            obs=resp
        )

    def guide(self, mask, q_id, kmap, resp):
        device = self.device

        num_users, max_seq_len = q_id.size()
        num_ques, num_kcs = kmap.size()
        trial_kc_mask = kmap[q_id,:]

        qid_enum = torch.arange(num_ques).to(device)
        diff = pyro.sample(
            'diff_mu',
            dist.Normal(
                self.diff_mu(qid_enum).squeeze(-1),
                self.diff_logvar(qid_enum).squeeze(-1)**2
            )
        )
        disc = pyro.sample(
            'diff_mu',
            dist.Normal(
                self.disc_mu(qid_enum).squeeze(-1),
                self.disc_logvar(qid_enum).squeeze(-1)**2
            )
        )

        pass

    def get_item_features(self):
        pass

    def get_ability_estimates(self):
        pass


if __name__=='__main__':
    from vtirt.data.wiener import Wiener2PLDataset
    from vtirt.data.utils import to_device
    from torch.utils.data import DataLoader
    vtirt = VTIRT(8, 10).to('cuda:0')
    dataset = Wiener2PLDataset(8,2,10,5, overwrite=True)
    loader = DataLoader(dataset, batch_size=3, collate_fn=dataset.collate_fn)
    batch = to_device(next(iter(loader)), 'cuda:0')
    vtirt.model(batch['mask'], batch['q_id'], batch['kmap'], None)
    # vtirt.guide(batch['mask'], batch['q_id'], batch['kmap'], None)
