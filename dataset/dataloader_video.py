import glob
import os
import pdb
import random
import sys
import time
import warnings

import cv2
import pandas
import six
import torch

warnings.simplefilter(action="ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T

# import pyarrow as pa
from PIL import Image
from torch.utils.data.sampler import Sampler

from utils import video_augmentation

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(
        self,
        prefix,
        gloss_dict,
        drop_ratio=1,
        num_gloss=-1,
        mode="train",
        transform_mode=True,
        datatype="lmdb",
        proposal_num=1,
        max_label_len=30,
    ):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.feat_prefix = f"{prefix}/features/fullFrame-256x256px/{mode}"
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(
            f"./preprocess/phoenix2014/{mode}_info.npy", allow_pickle=True
        ).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        self.img_randaug = T.RandAugment()
        self.img_norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.max_label_len = max(
            [
                len(inp["label"].split(" "))
                for k, inp in self.inputs_list.items()
                if type(k) == int
            ]
        )  # 28
        self.mean_label_len = [
            len(inp["label"].split(" "))
            for k, inp in self.inputs_list.items()
            if type(k) == int
        ]  # 28
        self.mean_label_len = sum(self.mean_label_len) / len(self.mean_label_len)  # 28
        # print("!!!", self.max_label_len, self.mean_label_len)
        # train 5671
        # Apply testing transform.
        # !!! 28 11.499030153412097
        # dev 540
        # Apply testing transform.
        # !!! 23 10.383333333333333
        # test 629
        # Apply testing transform.
        # !!! 21 10.50556438791733
        self.proposal_num = proposal_num
        self.max_label_len = max_label_len
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi, label_proposals = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return (
                input_data,
                torch.LongTensor(label),
                self.inputs_list[idx]["original_info"],
                label_proposals,
            )
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return (
                input_data,
                torch.LongTensor(label),
                self.inputs_list[idx]["original_info"],
            )
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]["original_info"]

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(
            self.prefix, "features/fullFrame-256x256px/" + fi["folder"]
        )
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi["label"].split(" "):
            if phase == "":
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])
        img_list = [Image.open(img_path).convert("RGB") for img_path in img_list]
        if self.transform_mode == "train":
            img_list = [self.img_randaug(img) for img in img_list]
        img_list = [np.asarray(img) for img in img_list]
        label_proposals = [self.del_ins_sub(label_list, op_ratio=0)] + [
            self.del_ins_sub(label_list) for _ in range(self.proposal_num)
        ]
        return (
            # [
            #     cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            #     for img_path in img_list
            # ],
            img_list,
            label_list,
            fi,
            label_proposals,
        )

    def del_ins_sub(self, label_list, op_ratio=0.2):
        label_list = label_list.copy()  # Warning: shallow copy
        op_num = int(len(label_list) * op_ratio)
        for op in torch.rand(op_num):
            op = int(op * 3)
            if op == 0:  # del
                del_idx = random.choice([i for i in range(len(label_list))])
                label_list = label_list[:del_idx] + label_list[del_idx + 1 :]
            elif op == 1:  # ins
                ins_idx = random.choice([i for i in range(len(label_list) + 1)])
                ins_label = random.choice([i for i in range(len(self.dict))]) + 1
                label_list = label_list[:ins_idx] + [ins_label] + label_list[ins_idx:]
            elif op == 2:  # sub
                sub_idx = random.choice([i for i in range(len(label_list))])
                sub_label = random.choice([i for i in range(len(self.dict))]) + 1
                label_list[sub_idx] = sub_label
        label_list = (
            [len(self.dict) + 1]
            + label_list
            + [0 for i in range(self.max_label_len - len(label_list) - 1)]
        )
        label_list = label_list[: self.max_label_len]
        return label_list

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(
            f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True
        ).item()
        return data["features"], data["label"]

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        # video = video.float() / 127.5 - 1
        video = self.img_norm(video)
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose(
                [
                    # video_augmentation.CenterCrop(224),
                    # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                    video_augmentation.RandomCrop(224),
                    video_augmentation.RandomHorizontalFlip(0.5),
                    video_augmentation.ToTensor(),
                    video_augmentation.TemporalRescale(0.2),
                    # video_augmentation.Resize(0.5),
                ]
            )
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose(
                [
                    video_augmentation.CenterCrop(224),
                    # video_augmentation.Resize(0.5),
                    video_augmentation.ToTensor(),
                ]
            )

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info, label_proposals = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor(
                [np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video]
            )
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [
                torch.cat(
                    (
                        vid[0][None].expand(left_pad, -1, -1, -1),
                        vid,
                        vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [
                torch.cat(
                    (
                        vid,
                        vid[-1][None].expand(max_len - len(vid), -1),
                    ),
                    dim=0,
                )
                for vid in video
            ]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            label_proposals = torch.LongTensor(label_proposals)
            label_proposals_mask = torch.zeros(label_proposals.shape).masked_fill_(
                torch.eq(label_proposals, 0), -float("inf")
            )
            return (
                padded_video,
                video_length,
                padded_label,
                label_length,
                label_proposals,
                label_proposals_mask,
                info,
            )

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
