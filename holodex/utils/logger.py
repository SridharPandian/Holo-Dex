import os
import wandb
import csv
from torch.utils.tensorboard import SummaryWriter

class Writer(object):
    def __init__(
        self,
        log_path
    ):
        self.log_path = log_path
        self.values = {}

    def update(self, data, step):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


class TBWriter(Writer):
    def __init__(self, log_path):
        super().__init__(log_path)
        self.writer = SummaryWriter(self.log_path)

    def update(self, data, step):
        for key in data.keys():
            self.writer.add_scalar(key, data[key], step)
        
    def dump(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

class WandBWriter(Writer):
    def __init__(self, configs):
        super().__init__(configs.log_path)
        wandb.init(project = configs.project_name, entity = configs.wandb_user, config = dict(configs))
        wandb.run.name = configs.run_name

    def update(self, data, step):
        wandb.log(data, step = step)

    def dump(self):
        pass

    def close(self):
        wandb.finish()

class Logger(object):
    def __init__(self, configs):
        self.writers = dict()
        if configs.wandb:
            self.writers['wandb'] = WandBWriter(configs)

        if configs.tb:
            self.writers['tb'] = TBWriter(configs.log_path)

    def update(self, data, step):
        for key in self.writers.keys():
            self.writers[key].update(data, step)

    def dump(self):
        for key in self.writers.keys():
            self.writers[key].dump()

    def close(self):
        for key in self.writers.keys():
            self.writers[key].close()