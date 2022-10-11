import numpy as np
import torch
import torch.optim as optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import AdamW


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        if self.optim_dict["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict["base_lr"],
                momentum=0.9,
                nesterov=self.optim_dict["nesterov"],
                weight_decay=self.optim_dict["weight_decay"],
            )
        elif self.optim_dict["optimizer"] == "Adam":
            alpha = self.optim_dict["learning_ratio"]
            self.optimizer = optim.Adam(
                # [
                #     {'params': model.conv2d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.conv1d.parameters(), 'lr': self.optim_dict['base_lr']*alpha},
                #     {'params': model.rnn.parameters()},
                #     {'params': model.classifier.parameters()},
                # ],
                # model.conv1d.fc.parameters(),
                model.parameters(),
                lr=self.optim_dict["base_lr"],
                weight_decay=self.optim_dict["weight_decay"],
            )
        elif self.optim_dict["optimizer"] == "AdamW":
            self.optimizer = AdamW(
                model.parameters(),
                lr=self.optim_dict["base_lr"],
                eps=1e-8,
                betas=(0.9, 0.98),
                weight_decay=self.optim_dict["weight_decay"],
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(
            self.optimizer, self.optim_dict["step"]
        )

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ["SGD", "Adam"]:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.2
            )
            return lr_scheduler
        else:
            raise ValueError()

    def set_lr_scheduler(self, decay_power, max_steps):
        if decay_power == "cosine":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.optim_dict["warmup_steps"],
                num_training_steps=max_steps,
            )
        else:
            self.scheduler = get_polynomial_decay_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.optim_dict["warmup_steps"],
                num_training_steps=max_steps,
                lr_end=self.optim_dict["end_lr"],
                power=decay_power,
            )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
