import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import faulthandler
import importlib
import random
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.utils.data.distributed import DistributedSampler

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_eval, seq_feature_generation, seq_train
from utils.dist import init_dist, master_only


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval, self.arg
        )
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(
            self.arg.dataset_info["dict_path"], allow_pickle=True
        ).item()
        self.arg.model_args["num_classes"] = len(self.gloss_dict) + 1
        self.model, self.optimizer = self.loading()

    def start(self):
        if self.arg.phase == "train":
            self.recoder.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
            seq_model_list = []
            for epoch in range(
                self.arg.optimizer_args["start_epoch"], self.arg.num_epoch
            ):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                seq_train(
                    self.data_loader["train"],
                    self.model,
                    self.optimizer,
                    epoch,
                    self.recoder,
                )
                if eval_model:
                    dev_wer = seq_eval(
                        self.arg,
                        self.data_loader["dev"],
                        self.model,
                        "dev",
                        epoch,
                        self.arg.work_dir,
                        self.recoder,
                        self.arg.evaluate_tool,
                    )
                    self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_wer))
                    # train_wer = seq_eval(
                    #     self.arg,
                    #     self.data_loader["train_eval"],
                    #     self.model,
                    #     "train",
                    #     epoch,
                    #     self.arg.work_dir,
                    #     self.recoder,
                    #     self.arg.evaluate_tool,
                    # )
                    # self.recoder.print_log("Train WER: {:05.2f}%".format(train_wer))
                    # self.recoder.print_wandb(
                    #     {
                    #         "epoch": epoch,
                    #         "Train WER": train_wer,
                    #     }
                    # )

                if save_model and dist.get_rank() == 0:
                    model_path = "{}dev_{:05.5f}_epoch{}_model.pt".format(
                        self.arg.work_dir, dev_wer * 0.01, epoch
                    )
                    self.save_model(epoch, model_path)
                    seq_model_list.append(model_path)
                    seq_model_list = sorted(seq_model_list)
                    for path in seq_model_list[3:]:
                        os.remove(path)
                    seq_model_list = seq_model_list[:3]
                    print("seq_model_list", seq_model_list)
        elif self.arg.phase == "test":
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError("Please appoint --load-weights.")
            self.recoder.print_log("Model:   {}.".format(self.arg.model))
            self.recoder.print_log("Weights: {}.".format(self.arg.load_weights))
            train_wer = seq_eval(
                self.arg,
                self.data_loader["train_eval"],
                self.model,
                "train",
                self.epoch,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
            )
            dev_wer = seq_eval(
                self.arg,
                self.data_loader["dev"],
                self.model,
                "dev",
                self.epoch,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
            )
            test_wer = seq_eval(
                self.arg,
                self.data_loader["test"],
                self.model,
                "test",
                self.epoch,
                self.arg.work_dir,
                self.recoder,
                self.arg.evaluate_tool,
            )
            self.recoder.print_log("Evaluation Done.\n")
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model,
                    mode,
                    self.arg.work_dir,
                    self.recoder,
                )

    @master_only
    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    @master_only
    def save_model(self, epoch, save_path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.optimizer.scheduler.state_dict(),
                "rng_state": self.rng.save_rng_state(),
            },
            save_path,
        )

    def loading(self):
        print("Loading data")
        self.load_data()
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        # optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)
        if "decay_power" in self.arg.optimizer_args:
            max_steps = len(self.data_loader["train"]) * self.arg.num_epoch
            optimizer.set_lr_scheduler(self.arg.optimizer_args["decay_power"], max_steps)
        print("Loading model finished.")
        self.recoder.print_log("Params: {}".format(self.get_parameter_number(model)))
        return model, optimizer

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    def model_to_device(self, model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(),
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print("Successfully Remove Weights: {}.".format(w))
                else:
                    print("Can Not Remove Weights: {}.".format(w))
        weights = self.modified_weights(state_dict["model_state_dict"], False)
        self.epoch = state_dict["epoch"]
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict(
            [(k.replace(".module", ""), v) for k, v in state_dict.items()]
        )
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(
            self.arg.load_checkpoints, map_location=torch.device("cpu")
        )

        if len(torch.cuda.get_rng_state_all()) == len(state_dict["rng_state"]["cuda"]):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict["rng_state"])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args["start_epoch"] = state_dict["epoch"] + 1
        self.recoder.print_log(
            f"Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}"
        )

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(
            ["train", "train_eval", "dev", "test"], [True, False, False, False]
        )
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info["dataset_root"]
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(
                self.dataset[mode], mode, train_flag
            )
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        rank = dist.get_rank()
        sampler = DistributedSampler(dataset, shuffle=train_flag)

        dataloader_args = dict(
            dataset=self.dataset[mode],
            batch_size=self.arg.batch_size if train_flag else self.arg.test_batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.arg.num_worker,
            pin_memory=True,
            drop_last=train_flag,
            collate_fn=self.feeder.collate_fn,
        )
        dataloader_args["worker_init_fn"] = partial(
            worker_init_fn,
            num_workers=self.arg.num_worker,
            rank=rank,
            seed=self.arg.random_seed,
        )
        return torch.utils.data.DataLoader(**dataloader_args)


def set_random_seed(seed, cuda_deterministic=True):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def import_class(name):
    components = name.rsplit(".", 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == "__main__":
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, "r") as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", "r") as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    init_dist()
    set_random_seed(args.random_seed + args.local_rank, cuda_deterministic=True)
    processor = Processor(args)
    utils.pack_code("./", args.work_dir)
    processor.start()
