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
        self.conv2d.fc = nn.Linear(512, hidden_size)
        # self.conv1d = TemporalConv(
        #     input_size=512,
        #     hidden_size=hidden_size,
        #     conv_type=conv_type,
        #     use_bn=use_bn,
        #     num_classes=num_classes,
        # )
        self.decoder = utils.Decode(gloss_dict, num_classes, "beam")
        # self.temporal_model = BiLSTMLayer(
        #     rnn_type="LSTM",
        #     input_size=hidden_size,
        #     hidden_size=hidden_size,
        #     num_layers=2,
        #     bidirectional=True,
        # )
        encoder_configuration = BertConfig(
            num_hidden_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=8,
            # hidden_dropout_prob=0.3,
            # attention_probs_dropout_prob=0.3,
        )
        self.temporal_model = BertModel(encoder_configuration)
        decoder_configuration = BertConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=8,
            # hidden_dropout_prob=0.3,
            # attention_probs_dropout_prob=0.3,
        )
        self.reg_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.reg_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.reg_decoder = BertModel(decoder_configuration)
        self.reg_pred = nn.Linear(hidden_size, hidden_size)
        
        self.ctc_embed = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ctc_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.ctc_decoder = BertModel(decoder_configuration)
        self.ctc_pred = nn.Linear(hidden_size, self.num_classes)

        torch.nn.init.normal_(self.reg_mask_token, std=.02)
        torch.nn.init.normal_(self.ctc_mask_token, std=.02)
        # if weight_norm:
        #     self.classifier = NormLinear(hidden_size, self.num_classes)
        #     self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        # else:
        #     self.classifier = nn.Linear(hidden_size, self.num_classes)
        #     self.classifier = nn.Linear(hidden_size, self.num_classes)
        # if share_classifier:
        #     self.conv1d.fc = self.classifier

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

    def random_masking(self, x, mask_ratio, attention_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        masked_attention_mask = torch.gather(attention_mask, dim=1, index=ids_keep)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, masked_attention_mask, ids_keep

    def infer(self, x, len_x, label=None, label_lgt=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1)
        else:
            # frame-wise features
            framewise = x

        x = framewise
        lgt = len_x
        B, T, C = x.shape
        attention_mask = torch.ones(B, T).to(x)
        for b in range(batch):
            attention_mask[b][lgt[b].int() :] = 0

        x_masked, mask, ids_restore, masked_attention_mask, ids_keep = self.random_masking(x, mask_ratio=0.75, attention_mask=attention_mask)
        tm_outputs = self.temporal_model(
            inputs_embeds=x_masked, attention_mask=masked_attention_mask, position_ids=ids_keep
        ).last_hidden_state

        ctc_emb = self.ctc_embed(tm_outputs)
        # append mask tokens to sequence
        mask_tokens = self.ctc_mask_token.repeat(ctc_emb.shape[0], ids_restore.shape[1] + 1 - ctc_emb.shape[1], 1)
        ctc_emb = torch.cat([ctc_emb, mask_tokens], dim=1)  # no cls token
        ctc_emb = torch.gather(ctc_emb, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, ctc_emb.shape[2]))  # unshuffle

        ctc_hs = self.ctc_decoder(
            inputs_embeds=ctc_emb, attention_mask=attention_mask
        ).last_hidden_state

        # predictor projection
        ctc_logits = self.ctc_pred(ctc_hs).permute(1, 0, 2)

        reg_emb = self.reg_embed(tm_outputs)
        # append mask tokens to sequence
        mask_tokens = self.reg_mask_token.repeat(reg_emb.shape[0], ids_restore.shape[1] + 1 - reg_emb.shape[1], 1)
        reg_emb = torch.cat([reg_emb, mask_tokens], dim=1)  # no cls token
        reg_emb = torch.gather(reg_emb, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, reg_emb.shape[2]))  # unshuffle

        reg_hs = self.reg_decoder(
            inputs_embeds=reg_emb, attention_mask=attention_mask
        ).last_hidden_state

        # predictor projection
        reg_logits = self.reg_pred(reg_hs)

        conv_pred = (
            None
            if self.training
            else self.decoder.decode(
                ctc_logits, lgt, batch_first=False, probs=False
            )
        )

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "mask": mask,
            "attention_mask": attention_mask,
            "ctc_logits": ctc_logits,
            "reg_logits": reg_logits,
            "conv_sents": conv_pred,
        }

    def infer_wo_mask(self, ret_dict):
        x = ret_dict['framewise_features']
        lgt = ret_dict['feat_len']
        B, T, C = x.shape
        attention_mask = torch.ones(B, T).to(x)
        for b in range(B):
            attention_mask[b][lgt[b].int() :] = 0

        tm_outputs = self.temporal_model(
            inputs_embeds=x, attention_mask=attention_mask
        ).last_hidden_state

        ctc_emb = self.ctc_embed(tm_outputs)
        # append mask tokens to sequence

        ctc_hs = self.ctc_decoder(
            inputs_embeds=ctc_emb, attention_mask=attention_mask
        ).last_hidden_state

        # predictor projection
        ctc_logits = self.ctc_pred(ctc_hs).permute(1, 0, 2)

        pred = (
            None
            if self.training
            else self.decoder.decode(ctc_logits, lgt, batch_first=False, probs=False)
        )

        return {
            "sequence_logits": ctc_logits,
            "recognized_sents": pred,
        }

    def criterion_calculation(self, ret_dict, label, label_lgt, phase):
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
            elif k == "DecoderCTC":
                l = (
                    weight
                    * self.loss["CTCLoss"](
                        ret_dict["ctc_logits"].log_softmax(-1),
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
            elif k == "DecoderReg":
                pred = ret_dict["reg_logits"]
                target = ret_dict["framewise_features"].detach()
                mask = ret_dict["mask"] * ret_dict["attention_mask"]
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

                l = (pred - target) ** 2
                l = l.mean(dim=-1)  # [N, L], mean loss per patch

                l = weight * (l * mask).sum() / mask.sum()  # mean loss on removed patches

            loss_kv[f"{phase}/Loss/{k}"] = l.item()
            if not (np.isinf(l.item()) or np.isnan(l.item())):
                loss += l
            else:
                print("NAN")
        return loss, loss_kv

    def criterion_init(self):
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        # self.loss["distillation"] = SeqKD(T=8)
        self.loss["MSE"] = nn.MSELoss()
        return self.loss

    def forward(self, x, len_x, label=None, label_lgt=None, return_loss=False, phase='val'):
        x = x.to(self.device, non_blocking=True)
        # label = label.to(self.device, non_blocking=True)
        res = self.infer(x, len_x, label, label_lgt)
        res.update(self.infer_wo_mask(res))
        if return_loss:
            return self.criterion_calculation(res, label, label_lgt, phase=phase)
        else:
            return res, self.criterion_calculation(res, label, label_lgt, phase=phase)
