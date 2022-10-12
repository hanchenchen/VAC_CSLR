from itertools import groupby

import ctcdecode
import numpy as np
import torch
import torch.nn.functional as F


class Decode(object):
    def __init__(self, gloss_dict, num_classes, search_mode, blank_id=0):
        self.i2g_dict = dict((v[0], k) for k, v in gloss_dict.items())
        self.g2i_dict = {v: k for k, v in self.i2g_dict.items()}
        self.num_classes = num_classes
        self.search_mode = search_mode
        self.blank_id = blank_id
        vocab = [chr(x) for x in range(20000, 20000 + num_classes)]
        self.ctc_decoder = ctcdecode.CTCBeamDecoder(
            vocab, beam_width=10, blank_id=blank_id, num_processes=10
        )

    def decode(self, nn_output, vid_lgt, batch_first=True, probs=False):
        if not batch_first:
            nn_output = nn_output.permute(1, 0, 2)
        if self.search_mode == "max":
            return self.MaxDecode(nn_output, vid_lgt)
        else:
            return self.BeamSearch(nn_output, vid_lgt, probs)

    def BeamSearch(self, nn_output, vid_lgt, probs=False):
        """
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        """
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(
            nn_output, vid_lgt
        )
        ret_list = []
        for batch_idx in range(len(nn_output)):
            first_result = beam_result[batch_idx][0][: out_seq_len[batch_idx][0]]
            if len(first_result) != 0:
                first_result = torch.stack([x[0] for x in groupby(first_result)])
            ret_list.append(
                [
                    (self.i2g_dict[int(gloss_id)], idx)
                    for idx, gloss_id in enumerate(first_result)
                ]
            )
        return ret_list

    def BeamSearch_N(
        self, nn_output, vid_lgt, probs=False, N_beams=1, max_label_len=30
    ):
        """
        CTCBeamDecoder Shape:
                - Input:  nn_output (B, T, N), which should be passed through a softmax layer
                - Output: beam_resuls (B, N_beams, T), int, need to be decoded by i2g_dict
                          beam_scores (B, N_beams), p=1/np.exp(beam_score)
                          timesteps (B, N_beams)
                          out_lens (B, N_beams)
        """
        nn_output = nn_output.permute(1, 0, 2)
        if not probs:
            nn_output = nn_output.softmax(-1).cpu()
        vid_lgt = vid_lgt.cpu()
        beam_result, beam_scores, timesteps, out_seq_len = self.ctc_decoder.decode(
            nn_output, vid_lgt
        )
        ret_list = []
        label_proposals = []
        for batch_idx in range(len(nn_output)):
            label_proposals.append([])
            ret_list.append({})
            for beam_idx in range(N_beams):
                first_result = beam_result[batch_idx][beam_idx][
                    : out_seq_len[batch_idx][beam_idx]
                ]
                if len(first_result) != 0:
                    first_result = torch.stack([x[0] for x in groupby(first_result)])
                ret_list[batch_idx][beam_idx] = {
                    "ctc": " ".join(
                        [
                            self.i2g_dict[int(gloss_id)]
                            for idx, gloss_id in enumerate(first_result)
                        ]
                    )
                }
                res = (
                    [len(self.i2g_dict) + 1]
                    + [int(gloss_id) for idx, gloss_id in enumerate(first_result)]
                    + [0 for i in range(max_label_len - len(first_result) - 1)]
                )
                label_proposals[batch_idx].append(res[:max_label_len])
        label_proposals = torch.LongTensor(label_proposals)
        label_proposals_mask = torch.zeros(label_proposals.shape).masked_fill_(
            torch.eq(label_proposals, 0), -float("inf")
        )
        return label_proposals, label_proposals_mask, ret_list

    def MaxDecode(self, nn_output, vid_lgt):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            group_result = [
                x[0] for x in groupby(index_list[batch_idx][: vid_lgt[batch_idx]])
            ]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered
            ret_list.append(
                [
                    (self.i2g_dict[int(gloss_id)], idx)
                    for idx, gloss_id in enumerate(max_result)
                ]
            )
        return ret_list

    def MaxDecodeCA(self, nn_output, mask):
        index_list = torch.argmax(nn_output, axis=2)
        batchsize, lgt = index_list.shape
        ret_list = []
        ca_decoded_list = []
        for batch_idx in range(batchsize):
            filtered = index_list[batch_idx][: torch.sum((mask[batch_idx] == 0).int())]
            ret_list.append(
                [
                    (self.i2g_dict[int(gloss_id)], idx)
                    for idx, gloss_id in enumerate(filtered)
                ]
            )
            ca_decoded_list.append(
                " ".join(
                    [
                        self.i2g_dict[int(gloss_id)]
                        for idx, gloss_id in enumerate(filtered)
                    ]
                )
            )
        return ret_list, ca_decoded_list

    def i2g(self, index_list):
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(batchsize):
            filtered = index_list[batch_idx]
            ret_list.append(
                " ".join(
                    [
                        self.i2g_dict[int(gloss_id)]
                        for idx, gloss_id in enumerate(filtered)
                        if int(gloss_id) in self.i2g_dict
                    ]
                )
            )
        return ret_list
