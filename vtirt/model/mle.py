import torch
import torch.nn as nn


class MLE(nn.module):
    def __init__(self, num_ques, num_users, max_seq_len):
        super().__init__()

        self.diff = nn.Parameter(torch.randn(num_ques))
        self.disc = nn.Parameter(torch.randn(num_ques))
        self.trial_ability = nn.Parameter(
            torch.randn(num_users, max_seq_len)
        )

        self.device = None

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def get_loss(self, batch):
        u_id = torch.LongTensor(batch['u_id']).to(self.device)
        mask = batch['mask']
        q_id = batch['q_id']
        resp = batch['resp']
