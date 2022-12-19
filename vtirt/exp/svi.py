from dotmap import DotMap
import json

from vtirt.data.wiener import Wiener2PLDataset
from vtirt.model.vtirt import VTIRT

class ExpSVI:

    def __init__(self, config):
        self.config = config

        exp_config = self.config.exp
        data_config = self.config.data
        agent_config = self.config.model

        data_class = globals()[exp_config.data]
        self.train_dataset = data_class(split='train', **data_config)
        self.val_dataset = data_class(split='valid', **data_config)

        self.agent = globals()[exp_config.model](**agent_config)


    def train(self):
        pass

    def valid(self):
        pass

    def run(self):
        pass
