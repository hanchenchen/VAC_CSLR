from itertools import groupby
import os
import ctcdecode
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
fig, ax = plt.subplots(12+1, 1, sharex='col', sharey='row')
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


    def visualize_maxdecode(self, upsampled_nn_output, len_x, x, label=None, label_lgt=None, filename=""):
        index_list = torch.argmax(upsampled_nn_output, axis=2)
        probs = upsampled_nn_output.softmax(-1).cpu()
        batchsize, lgt = index_list.shape
        ret_list = []
        for batch_idx in range(1):
            group_result = [x[0] for x in groupby(index_list[batch_idx][:len_x[batch_idx].int()])]
            filtered = [*filter(lambda x: x != self.blank_id, group_result)]
            if len(filtered) > 0:
                max_result = torch.stack(filtered)
                max_result = [x[0] for x in groupby(max_result)]
            else:
                max_result = filtered   
            ret_list.append([(self.i2g_dict[int(gloss_id)], idx) for idx, gloss_id in
                             enumerate(max_result)])
            x = np.linspace(0,len_x[batch_idx].int().cpu(),num=len_x[batch_idx].int().cpu()) # 创建x的取值范围
            max_result = label[label_lgt[:batch_idx].sum():label_lgt[:batch_idx+1].sum()].tolist() if label is not None else max_result
            
            
            fig.set_size_inches(18, 25)
            ax[0].set_xlabel('Frame') #设置x轴名称 x label
            ax[0].set_ylabel('Probability') #设置y轴名称 y label
            ax[0].set_title('Probability of glosses') #设置图名为Simple Plot
            plt.cm.get_cmap('rainbow_r')
            color_list = plt.cm.Set3(np.linspace(0, 1, len(max_result)))
            print(batch_idx, label, max_result, probs.max())
            for idx, gloss_id in enumerate(max_result):
                ax[0].plot(x, probs[batch_idx,:len_x[batch_idx].int(),gloss_id], 
                label=self.i2g_dict[int(gloss_id)],
                color=color_list[idx], 
                linewidth=3) # 作y1 = x 图，并标记此线名为linear
                ax[idx+1].set_ylim((0, 1.5))
                ax[idx+1].bar(x, probs[batch_idx,:len_x[batch_idx].int(),gloss_id], 
                label=self.i2g_dict[int(gloss_id)],
                color=color_list[idx]) # 作y1 = x 图，并标记此线名为linear
                # ax[idx+1].set_xlabel(f'{self.i2g_dict[int(gloss_id)]}, {int(gloss_id)}') #设置图名为Simple Plot
                ax[idx+1].legend() #设置图名为Simple Plot
            # fig.legend() #自动检测要在图例中显示的元素，并且显示
            fig.tight_layout()#调整整体空白
            plt.subplots_adjust(wspace =0, hspace =0)#调整子图间距
            os.makedirs(f'visualization/24-0929-debug-flip-13/', exist_ok=True)
            plt.savefig(f'visualization/24-0929-debug-flip-13/{filename}-{batch_idx}-{label is not None}.pdf') #图形可视化
            # plt.clf()

        return ret_list