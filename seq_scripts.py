import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.slr_eval.wer_calculation import evaluate
from utils.dist import master_only


def reduce_loss_dict(loss_dict):
    """reduce loss dict.

    In distributed training, it averages the losses among different GPUs .

    Args:
        loss_dict (dict): Loss dict.
    """
    with torch.no_grad():
        keys = []
        losses = []
        for name, value in loss_dict.items():
            keys.append(name)
            losses.append(value)
        losses = torch.tensor(losses).cuda()
        dist.all_reduce(losses, op=dist.ReduceOp.SUM)
        if dist.get_rank() == 0:
            losses /= dist.get_world_size()
        loss_dict = {key: loss for key, loss in zip(keys, losses)}

        log_dict = {}
        loss_sum = 0.0
        for name, value in loss_dict.items():
            log_dict[name] = value.mean().item()
            loss_sum += log_dict[name]
        log_dict["Loss/sum"] = loss_sum
        return log_dict


def seq_train(loader, model, optimizer, epoch_idx, recoder):
    model.train()
    clr = [group["lr"] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(tqdm(loader)):
        vid = data[0]
        vid_lgt = data[1]
        label = data[2]
        label_lgt = data[3]
        loss, loss_kv = model(
            vid, vid_lgt, label=label, label_lgt=label_lgt, return_loss=True
        )
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            continue
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        # del vid, vid_lgt, label, label_lgt
        # torch.cuda.empty_cache()
        if batch_idx % recoder.log_interval == 0:
            loss_kv = reduce_loss_dict(loss_kv)
            recoder.print_log(
                "\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}".format(
                    epoch_idx, batch_idx, len(loader), loss.item(), clr[0]
                )
            )
            recoder.print_wandb(
                {
                    "epoch": epoch_idx,
                    "step": epoch_idx * len(loader) + batch_idx,
                    "Loss": loss.item(),
                    "lr": clr[0],
                    **loss_kv,
                }
            )
    optimizer.scheduler.step()


def data_to_device(data):
    if isinstance(data, torch.FloatTensor):
        return data.to(torch.device("cuda"), non_blocking=True)
    elif isinstance(data, torch.DoubleTensor):
        return data.float().to(torch.device("cuda"), non_blocking=True)
    elif isinstance(data, torch.ByteTensor):
        return data.long().to(torch.device("cuda"), non_blocking=True)
    elif isinstance(data, torch.LongTensor):
        return data.to(torch.device("cuda"), non_blocking=True)
    elif isinstance(data, list) or isinstance(data, tuple):
        return [data_to_device(d) for d in data]
    else:
        raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))


@torch.no_grad()
def seq_eval(
    cfg, loader, model, mode, epoch, work_dir, recoder, evaluate_tool="python"
):
    model.eval()
    total_sent = []
    total_info = []
    total_conv_sent = []
    stat = {i: [0, 0] for i in range(len(loader.dataset.dict))}
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        vid = data[0]
        vid_lgt = data[1]
        label = data[2]
        label_lgt = data[3]
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt, label=label, label_lgt=label_lgt)

        total_info += [file_name.split("|")[0] for file_name in data[-1]]
        total_sent += ret_dict["recognized_sents"]
        total_conv_sent += ret_dict["conv_sents"]

    gather_total_info = [None for _ in range(dist.get_world_size())]
    gather_total_sent = [None for _ in range(dist.get_world_size())]
    gather_total_conv_sent = [None for _ in range(dist.get_world_size())]
    dist.barrier()
    dist.all_gather_object(
        gather_total_info,
        total_info,
    )
    dist.all_gather_object(
        gather_total_sent,
        total_sent,
    )
    dist.all_gather_object(
        gather_total_conv_sent,
        total_conv_sent,
    )
    total_sent = []
    total_info = []
    total_conv_sent = []
    for i in gather_total_info:
        total_info += i
    for i in gather_total_sent:
        total_sent += i
    for i in gather_total_conv_sent:
        total_conv_sent += i
    lstm_ret = 100.0
    if dist.get_rank() == 0:
        try:
            python_eval = True if evaluate_tool == "python" else False
            write2file(
                work_dir + "output-hypothesis-{}.ctm".format(mode),
                total_info,
                total_sent,
            )
            write2file(
                work_dir + "output-hypothesis-{}-conv.ctm".format(mode),
                total_info,
                total_conv_sent,
            )
            conv_ret = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-{}-conv.ctm".format(mode),
                evaluate_dir=cfg.dataset_info["evaluation_dir"],
                evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
            )
            lstm_ret = evaluate(
                prefix=work_dir,
                mode=mode,
                output_file="output-hypothesis-{}.ctm".format(mode),
                evaluate_dir=cfg.dataset_info["evaluation_dir"],
                evaluate_prefix=cfg.dataset_info["evaluation_prefix"],
                output_dir="epoch_{}_result/".format(epoch),
                python_evaluate=python_eval,
                triplet=True,
            )
        except:
            print("Unexpected error:", sys.exc_info()[0])
            lstm_ret = 100.0
        finally:
            pass
    recoder.print_log(
        f"Epoch {epoch}, {mode} {lstm_ret: 2.2f}%", f"{work_dir}/{mode}.txt"
    )
    recoder.print_wandb(
        {
            "epoch": epoch,
            f"{mode} WER": lstm_ret,
        }
    )
    return lstm_ret


def seq_feature_generation(loader, model, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
        os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(
            os.listdir(src_path)
        ):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = data_to_device(data[0])
        vid_lgt = data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict["framewise_features"][sample_idx][
                    :, : vid_lgt[sample_idx]
                ]
                .T.cpu()
                .detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(
                    info[sample_idx],
                    word_idx * 1.0 / 100,
                    (word_idx + 1) * 1.0 / 100,
                    word[0],
                )
            )
