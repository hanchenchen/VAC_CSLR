import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertConfig, BertModel

import utils
from modules import BiLSTMLayer, TemporalConv
from modules.criterions import SeqKD


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
        self,
        num_classes,
        c2d_type,
        conv_type,
        use_bn=False,
        hidden_size=1024,
        gloss_dict=None,
        loss_weights=None,
        weight_norm=True,
        share_classifier=True,
    ):
        super(SLRModel, self).__init__()
        self.device = torch.device("cuda")
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(
            input_size=512,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )
        self.decoder = utils.Decode(gloss_dict, num_classes, "beam")
        # self.temporal_model = BiLSTMLayer(
        #     rnn_type="LSTM",
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     bidirectional=True,
        # )
        configuration = BertConfig(
            num_hidden_layers=2, hidden_size=hidden_size, num_attention_heads=8
        )
        self.temporal_model = BertModel(configuration)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier

    #     self.register_backward_hook(self.backward_hook)

    # def backward_hook(self, module, grad_input, grad_output):
    #     for g in grad_input:
    #         g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat(
                [
                    tensor,
                    tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_(),
                ]
            )

        x = torch.cat(
            [
                inputs[len_x[0] * idx : len_x[0] * idx + lgt]
                for idx, lgt in enumerate(len_x)
            ]
        )
        x = self.conv2d(x)
        x = torch.cat(
            [
                pad(x[sum(len_x[:idx]) : sum(len_x[: idx + 1])], len_x[0])
                for idx, lgt in enumerate(len_x)
            ]
        )
        return x

    def infer(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs["visual_feat"]
        lgt = conv1d_outputs["feat_len"]
        T, B, C = x.shape
        attention_mask = torch.ones(B, T).to(x)
        for b in range(batch):
            attention_mask[b][lgt[b].int() :] = 0
        tm_outputs = self.temporal_model(
            inputs_embeds=x.transpose(0, 1), attention_mask=attention_mask
        ).last_hidden_state.permute(1, 0, 2)
        l2r_tm_outputs = self.temporal_model(
            inputs_embeds=x.transpose(0, 1),
            attention_mask=torch.tril(attention_mask, diagonal=0),
        ).last_hidden_state.permute(1, 0, 2)
        r2l_tm_outputs = self.temporal_model(
            inputs_embeds=x.transpose(0, 1),
            attention_mask=torch.triu(attention_mask, diagonal=0),
        ).last_hidden_state.permute(1, 0, 2)
        outputs = self.classifier(tm_outputs)
        l2r_outputs = self.classifier(l2r_tm_outputs)
        r2l_outputs = self.classifier(r2l_tm_outputs)
        pred = (
            None
            if self.training
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        )
        conv_pred = (
            None
            if self.training
            else self.decoder.decode(
                conv1d_outputs["conv_logits"], lgt, batch_first=False, probs=False
            )
        )

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "sequence_logits": outputs,
            "l2r_logits": l2r_outputs,
            "r2l_logits": r2l_outputs,
            "conv_sents": conv_pred,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        loss_kv = {}
        for k, weight in self.loss_weights.items():
            if k == "ConvCTC":
                l = (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["conv_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "SeqCTC":
                l = (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["sequence_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "Dist":
                l = weight * self.loss["distillation"](
                    ret_dict["conv_logits"],
                    ret_dict["sequence_logits"].detach(),
                    use_blank=False,
                )
            elif k == "l2r-CTC":
                l = (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["l2r_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            elif k == "r2l-CTC":
                l = (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["r2l_logits"].log_softmax(-1),
                        label.cpu().int(),
                        ret_dict["feat_len"].cpu().int(),
                        label_lgt.cpu().int(),
                    ).mean()
                )
            loss_kv[f"Loss/{k}"] = l.item()
            if not (np.isinf(l.item()) or np.isnan(l.item())):
                loss += l
        return loss, loss_kv

    def criterion_init(self):
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = SeqKD(T=8)
        return self.loss

    def forward(self, x, len_x, label=None, label_lgt=None, return_loss=False):
        x = x.to(self.device, non_blocking=True)
        # label = label.to(self.device, non_blocking=True)
        res = self.infer(x, len_x, label, label_lgt)
        if return_loss:
            return self.criterion_calculation(res, label, label_lgt)
        else:
            return res
