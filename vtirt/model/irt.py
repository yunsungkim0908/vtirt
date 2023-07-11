import numpy as np

import math
import torch
import pyro
import pyro.distributions as dist

ITEM_CHAR_FUNC={
    'logistic': lambda x: torch.sigmoid(x),
    'normal': lambda x: (1 + torch.erf(x/math.sqrt(2)))/2
}


def dirt_2param(
        kmap,
        std_diff,
        std_disc,
        std_init,
        std_theta,
        train_mask,
        train_q_id,
        train_resp,
        inf_mask=None,
        inf_q_id=None,
        inf_resp=None,
        item_char='logistic',
        device='cpu',
):
    """
        mask: shape (U,T)
        q_id: shape (U,T)
        resp: shape (U,T)
        kmap: shape (Q,K)
    """
    num_ques, num_kcs = kmap.size()

    zero_mean = torch.zeros((num_ques,)).to(device)
    unit_std = torch.ones((num_ques,)).to(device)
    diff = pyro.sample('diff_mu',
                       dist.Normal(zero_mean,unit_std*std_diff).to_event(1))
    disc = pyro.sample('disc_mu',
                       dist.Normal(zero_mean,unit_std*std_disc).to_event(1))

    for mode in (['train'] if inf_mask is None else ['train', 'infer']):
        q_id = train_q_id if mode == 'train' else inf_q_id
        mask = train_mask if mode == 'train' else inf_mask
        resp = train_resp if mode == 'train' else inf_resp
        mode_prefix = ('inf_' if mode == 'infer' else '')

        num_users, max_seq_len = q_id.size()
        trial_kc_mask = kmap[q_id,:]

        curr_kc_ability = torch.zeros((num_users, num_kcs)).to(device)
        trial_ability_kc = []
        for t in range(max_seq_len):
            trial_kc_mask_t = trial_kc_mask[:,t]

            mu_t = (
                torch.masked_select(curr_kc_ability, trial_kc_mask_t)
            )
            std_t = (std_theta if t > 0 else std_init) * (
                torch.ones_like(mu_t).to(device)
            )
            ability_next = pyro.sample(
                mode_prefix + f'ability_{t+1}',
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
        trial_probs = ITEM_CHAR_FUNC[item_char](trial_logits)

        _ = pyro.sample(
            mode_prefix + 'resp',
            dist.Bernoulli(probs=trial_probs)
                .mask(mask.bool())
                .to_event(2),
            obs=resp
        )

