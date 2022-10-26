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
        hidden_size=512,
        gloss_dict=None,
        loss_weights=None,
        weight_norm=True,
        share_classifier=True,
        proposal_num=1,
        max_label_len=30,
    ):
        super(SLRModel, self).__init__()
        self.device = torch.device("cuda")
        self.decoder = None
        self.loss = dict()
        self.num_classes = num_classes
        self.criterion_init()
        self.loss_weights = loss_weights
        self.proposal_num = proposal_num
        self.max_label_len = max_label_len
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = nn.Identity()
        self.conv1d = TemporalConv(
            input_size=hidden_size,
            hidden_size=hidden_size,
            conv_type=conv_type,
            use_bn=use_bn,
            num_classes=num_classes,
        )
        self.decoder = utils.Decode(gloss_dict, num_classes, "beam")
        self.temporal_model = BiLSTMLayer(
            rnn_type="LSTM",
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
        )

        decoder_configuration = BertConfig(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=8,
            # hidden_dropout_prob=0.3,
            # attention_probs_dropout_prob=0.3,
        )
        if "DecoderReg" in self.loss_weights:
            self.reg_embed = nn.Linear(hidden_size, hidden_size, bias=True)
            self.reg_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.reg_decoder = BertModel(decoder_configuration)
            self.reg_pred = nn.Linear(hidden_size, hidden_size)
            self.h1 = nn.Linear(hidden_size, hidden_size)
            self.h2 = nn.Linear(hidden_size, hidden_size)
            torch.nn.init.normal_(self.reg_mask_token, std=0.02)

        if "SimsiamAlign" in self.loss_weights:
            self.h1 = nn.Linear(hidden_size, hidden_size)
            self.h2 = nn.Linear(hidden_size, hidden_size)

        if "DecoderCTC" in self.loss_weights:
            self.ctc_embed = nn.Linear(hidden_size, hidden_size, bias=True)
            self.ctc_mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.ctc_decoder = BertModel(decoder_configuration)
            self.ctc_pred = nn.Linear(hidden_size, self.num_classes)
            torch.nn.init.normal_(self.ctc_mask_token, std=0.02)

        if "LenPred" in self.loss_weights:
            self.len_model = BiLSTMLayer(
                rnn_type="LSTM",
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            )
            self.len_predictor = nn.Sequential(
                nn.Linear(hidden_size * 4, 1),
                nn.Sigmoid(),
            )

        if "SignGlossContrast" in self.loss_weights:
            self.sign_encoder = BiLSTMLayer(
                rnn_type="LSTM",
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            )
            self.gloss_encoder = BiLSTMLayer(
                rnn_type="LSTM",
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                bidirectional=True,
                dropout=0.3,
            )
            self.gloss_embedding_layer = nn.Embedding(
                num_embeddings=self.num_classes + 2, embedding_dim=hidden_size
            )

        if "CADecoder" in self.loss_weights:
            # self.pos_emb = nn.Parameter(torch.zeros(1, self.max_label_len, hidden_size))
            # self.pos_kv_emb = nn.Parameter(torch.zeros(1, 200, hidden_size))
            # torch.nn.init.normal_(self.pos_emb, std=0.02)
            # torch.nn.init.normal_(self.pos_kv_emb, std=0.02)
            # self.embedding_layer = nn.Embedding(
            #     num_embeddings=self.num_classes + 2, embedding_dim=hidden_size
            # )
            # decoder_layer = nn.TransformerDecoderLayer(
            #     d_model=hidden_size, nhead=8, batch_first=True
            # )
            # self.ca_decoder = nn.TransformerDecoder(
            #     decoder_layer=decoder_layer, num_layers=2
            # )
            # encoder_configuration = BertConfig(
            #     num_hidden_layers=4,
            #     hidden_size=hidden_size,
            #     num_attention_heads=8,
            #     # hidden_dropout_prob=0.3,
            #     # attention_probs_dropout_prob=0.3,
            # )
            # self.l_model_1 = BertModel(encoder_configuration, add_pooling_layer=False)
            # self.l_model_2 = BertModel(encoder_configuration, add_pooling_layer=False)

            # self.ca_decoder = nn.Transformer(
            #     d_model=hidden_size,
            #     nhead=8,
            #     batch_first=True,
            #     num_encoder_layers=4,
            #     num_decoder_layers=4,
            # )
            
            # self.ca_conf_model = nn.Transformer(
            #     d_model=hidden_size,
            #     nhead=8,
            #     batch_first=True,
            #     num_encoder_layers=4,
            #     num_decoder_layers=4,
            # )
            self.gloss_embedding_layer = nn.Embedding(
                num_embeddings=self.num_classes + 2, embedding_dim=hidden_size
            )
            self.gloss_pos_emb = nn.Parameter(torch.zeros(1, 200, hidden_size))
            torch.nn.init.normal_(self.gloss_pos_emb, std=0.02)
            self.sign_pos_emb = nn.Parameter(torch.zeros(1, 200, hidden_size))
            torch.nn.init.normal_(self.sign_pos_emb, std=0.02)
            self.gloss_ca_pos_emb = nn.Parameter(torch.zeros(1, 200, hidden_size))
            torch.nn.init.normal_(self.gloss_ca_pos_emb, std=0.02)
            self.sign_ca_pos_emb = nn.Parameter(torch.zeros(1, 200, hidden_size))
            torch.nn.init.normal_(self.sign_ca_pos_emb, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.gloss_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.sign_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True)
            self.gloss_sign_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

            self.conf_predictor = nn.Linear(hidden_size*2, 1)
            
            # self.conv1d_1 = TemporalConv(
            #     input_size=2048,
            #     hidden_size=hidden_size,
            #     conv_type=conv_type,
            #     use_bn=use_bn,
            #     num_classes=num_classes,
            # )
            
            # self.conv1d_2 = TemporalConv(
            #     input_size=2048,
            #     hidden_size=hidden_size,
            #     conv_type=conv_type,
            #     use_bn=use_bn,
            #     num_classes=num_classes,
            # )

        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
            # self.conv1d_1.fc = NormLinear(hidden_size, self.num_classes)
            # self.conv1d_2.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.classifier = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
            # self.conv1d_1.fc = self.classifier
            # self.conv1d_2.fc = self.classifier

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

        noise = torch.rand(N, L, device=x.device) + (
            1 - attention_mask
        )  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
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

    def forward_conv_layer(self, x, len_x):
        # embed frames by resnet
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        frame_feat = self.masked_bn(inputs, len_x)
        frame_feat = frame_feat.reshape(batch, temp, -1)

        conv1d_outputs = self.conv1d(frame_feat.transpose(1, 2), len_x)
        visual_feat = conv1d_outputs["visual_feat"].permute(1, 0, 2)
        lgt = conv1d_outputs["feat_len"].int()

        # conv1d_outputs_1 = self.conv1d_1(frame_feat.transpose(1, 2), len_x)
        # visual_feat_1 = conv1d_outputs_1["visual_feat"].permute(1, 0, 2)
        
        # conv1d_outputs_2 = self.conv1d_2(frame_feat.transpose(1, 2), len_x)
        # visual_feat_2 = conv1d_outputs_2["visual_feat"].permute(1, 0, 2)

        B, T, C = visual_feat.shape
        attention_mask = torch.ones(B, T).to(x)
        for b in range(batch):
            attention_mask[b][lgt[b] :] = 0

        conv_pred = (
            None
            if self.training
            else self.decoder.decode(
                conv1d_outputs["conv_logits"], lgt, batch_first=False, probs=False
            )
        )

        return {
            "frame_feat": frame_feat,
            "frame_num": len_x,
            "visual_feat": visual_feat,
            # "visual_feat_1": visual_feat_1,
            # "visual_feat_2": visual_feat_2,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs["conv_logits"],
            "conv_pred": conv_pred,
            "attention_mask": attention_mask,
        }

    def forward_masked_encoder(self, ret):

        x = ret["visual_feat"]
        lgt = ret["feat_len"]
        attention_mask = ret["attention_mask"]

        (
            x_masked,
            mask,
            ids_restore,
            masked_attention_mask,
            ids_keep,
        ) = self.random_masking(x, mask_ratio=0.9, attention_mask=attention_mask)
        masked_hs = self.temporal_model(
            inputs_embeds=x_masked,
            attention_mask=masked_attention_mask,
            position_ids=ids_keep,
        ).last_hidden_state

        return {
            "masked_hs": masked_hs,
            "ids_restore": ids_restore,
            "mask": mask,
        }

    def forward_ctc_decoder(self, ret):
        x = ret["masked_hs"]
        attention_mask = ret["attention_mask"]
        ids_restore = ret["ids_restore"]

        ctc_emb = self.ctc_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.ctc_mask_token.repeat(
            ctc_emb.shape[0], ids_restore.shape[1] - ctc_emb.shape[1], 1
        )
        ctc_emb = torch.cat([ctc_emb, mask_tokens], dim=1)  # no cls token
        ctc_emb = torch.gather(
            ctc_emb,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, ctc_emb.shape[2]),
        )  # unshuffle

        ctc_hs = self.ctc_decoder(
            inputs_embeds=ctc_emb, attention_mask=attention_mask
        ).last_hidden_state
        # predictor projection
        ctc_logits = self.ctc_pred(ctc_hs).permute(1, 0, 2)

        decoder_ctc_pred = (
            None
            if self.training
            else self.decoder.decode(ctc_logits, lgt, batch_first=False, probs=False)
        )

        return {
            "decoder_ctc_logits": ctc_logits,
            "decoder_ctc_pred": decoder_ctc_pred,
        }

    def forward_reg_decoder(self, ret):
        x = ret["masked_hs"]
        attention_mask = ret["attention_mask"]
        ids_restore = ret["ids_restore"]

        reg_emb = self.reg_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.reg_mask_token.repeat(
            reg_emb.shape[0], ids_restore.shape[1] - reg_emb.shape[1], 1
        )
        reg_emb = torch.cat([reg_emb, mask_tokens], dim=1)  # no cls token
        reg_emb = torch.gather(
            reg_emb,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, reg_emb.shape[2]),
        )  # unshuffle

        reg_hs = self.reg_decoder(
            inputs_embeds=reg_emb, attention_mask=attention_mask
        ).last_hidden_state
        # predictor projection
        reg_logits = self.reg_pred(reg_hs)

        return {
            "decoder_reg_logits": reg_logits,
        }

    def forward_encoder(self, ret):
        x = ret["visual_feat"]
        lgt = ret["feat_len"]
        attention_mask = ret["attention_mask"]

        # encoded_hs = self.temporal_model(
        #     inputs_embeds=x, attention_mask=attention_mask
        # ).last_hidden_state.permute(1, 0, 2)
        encoded_hs = self.temporal_model(x.permute(1, 0, 2), lgt)

        logits = self.classifier(encoded_hs["predictions"])
        pred = (
            None
            if self.training
            else self.decoder.decode(logits, lgt, batch_first=False, probs=False)
        )

        if "LenPred" in self.loss_weights:
            encoded_len_hs = self.len_model(x.permute(1, 0, 2), lgt)
            L, B, C = encoded_len_hs["hidden"].shape
            len_logits = self.len_predictor(
                encoded_len_hs["hidden"].permute(1, 0, 2).reshape(B, -1) # mean?
            )
        else:
            len_logits = None

        return {
            "sequence_feat": encoded_hs["predictions"],
            "sequence_logits": logits,
            "ctc_pred": pred,
            "len_logits": len_logits,
        }

    def forward_ca_decoder(self, ret, label_proposals=None, label_proposals_mask=None):
        lgt = ret["feat_len"]
        sign_mask = torch.zeros(ret["attention_mask"].shape).to(
            self.device, non_blocking=True
        )
        sign_mask = sign_mask.masked_fill(
            torch.eq(ret["attention_mask"], 0), -float("inf")
        )

        ca_label = label_proposals[:, :1, :]
        ca_label = ca_label.masked_fill(torch.eq(ca_label, 0), -100)
        ca_label = ca_label.repeat(1, label_proposals.shape[1], 1)
        ca_label = ca_label[:, :, 1:]
        
        gloss_emb = self.gloss_embedding_layer(label_proposals)
        B, K, N, C = gloss_emb.shape
        gloss_emb = gloss_emb.reshape(B * K, N, C) + self.gloss_pos_emb[:, :N, :]
        gloss_mask = label_proposals_mask # .reshape(B, K, N)
        inp_mask = gloss_mask.reshape(B, K, 1, 1, N).repeat(1, 1, 8, N, 1).reshape(B * K * 8, N, N)
        gloss_emb = self.gloss_encoder(
            src=gloss_emb,
            mask=inp_mask,
            )
        textual_gloss_emb = gloss_emb
        gloss_emb = gloss_emb+self.gloss_ca_pos_emb[:, :N, :]

        sign_emb = ret["visual_feat"]
        B, M, C = sign_emb.shape
        sign_emb = sign_emb + self.sign_pos_emb[:, :M, :]
        inp_mask = sign_mask.reshape(B, 1, 1, M).repeat(1, 8, M, 1).reshape(B * 8, M, M)
        sign_emb = self.sign_encoder(
            src=sign_emb,
            mask=inp_mask,
            )
        textual_sign_emb = sign_emb
        sign_emb = sign_emb+self.sign_ca_pos_emb[:, :M, :]

        sign_emb = sign_emb.reshape(B, 1, M, C).repeat(1, K, 1, 1).reshape(B * K, M, C)
        sign_mask = sign_mask.reshape(B, 1, M).repeat(1, K, 1)

        textual_gloss_emb = textual_gloss_emb.reshape(B, K, N, C)[:, :, 0, :]
        textual_sign_emb = textual_sign_emb.reshape(B, 1, M, C)[:, :, 0, :]

        contrast_logits = torch.mul(textual_gloss_emb, textual_sign_emb).sum(dim=-1)

        inp_emb = torch.cat([gloss_emb, sign_emb], dim=1)
        inp_mask = torch.cat([gloss_mask, sign_mask], dim=2)
        inp_mask = inp_mask.reshape(B, K, 1, 1, M+N).repeat(1, 1, 8, M+N, 1).reshape(B * K * 8, M+N, M+N)

        g_s_hs = self.gloss_sign_encoder(
            src=inp_emb,
            mask=inp_mask,
            )
        g_s_hs = g_s_hs.reshape(B, K, M+N, C)
        g_s_hs = torch.cat([g_s_hs[:, :, 0, :], g_s_hs[:, :, N, :]], dim=2)

        conf_logits = self.conf_predictor(g_s_hs).reshape(B, K)

        label_proposals_mask_w_max_conf = label_proposals_mask[
            torch.arange(B), torch.argmax(conf_logits, dim=1), 1:
        ]
        pred_label_proposals =[]
        conf_pred = (
            None
            if self.training
            else self.decoder.MaxDecodeCA(None, label_proposals_mask_w_max_conf, index_list=label_proposals[torch.arange(B), torch.argmax(conf_logits, dim=1), 1:])[0]
        )
        # pred = (
        #     None
        #     if self.training
        #     else self.decoder.MaxDecodeCA(logits_w_max_conf, label_proposals_mask_w_max_conf)[0]
        # )
        if not self.training:
            conf_score = conf_logits.softmax(-1)
            gt = self.decoder.i2g(ca_label[:, 0, :])
            ret_list = [{} for batch_idx in range(B)]
            for batch_idx in range(B):
                ret_list[batch_idx]["gt"] = gt[batch_idx]
                inp = self.decoder.i2g(label_proposals[batch_idx, :, 1:])
                # mask = label_proposals_mask.reshape(B, K, 8, N, N)[
                #     batch_idx, torch.arange(K), 0, 0, 1:
                # ]
                # p, ca_decoded_list = self.decoder.MaxDecodeCA(logits[batch_idx], None)
                for beam_idx in range(K):
                    ret_list[batch_idx][beam_idx] = {}
                    ret_list[batch_idx][beam_idx]["inp_"] = inp[beam_idx]
                    # ret_list[batch_idx][beam_idx]["pred"] = ca_decoded_list[beam_idx]
                    ret_list[batch_idx][beam_idx]["conf"] = conf_score[batch_idx][
                        beam_idx
                    ].item()
        else:
            ret_list = []
        return {
            # "ca_feat": encoded_hs,
            # "ca_logits": logits,
            "conf_logits": conf_logits,
            "contrast_logits": contrast_logits,
            # "ca_pred": pred,
            "conf_pred": conf_pred,
            # "ca_label": ca_label,
            "ca_results": ret_list,
            # "ca_unmatched_label": ca_unmatched_label
        }

    def forward_contrast(self, ret, label_proposals=None, label_proposals_mask=None):
        sign_len = ret["feat_len"]
        x = ret["visual_feat"]
        sign_feat = self.sign_encoder(x.permute(1, 0, 2), sign_len)
        sign_feat = sign_feat["hidden"].mean(dim=0) # L, B, C -> B, C
        
        gloss_len = (label_proposals_mask==0).float().sum(dim=-1)
        gloss_emb = self.gloss_embedding_layer(label_proposals)
        B, K, N, C = gloss_emb.shape
        gloss_len = gloss_len.reshape(B*K)
        gloss_emb = gloss_emb.reshape(B*K, N, C)
        
        gloss_feat = self.gloss_encoder(gloss_emb.permute(1, 0, 2), gloss_len.cpu().int(), enforce_sorted=False)
        gloss_feat = gloss_feat["hidden"].mean(dim=0)


        sign_feat = sign_feat.reshape(B, 1, C)
        gloss_feat = gloss_feat.reshape(B, K, C)
        contrast_logits = torch.mul(gloss_feat, sign_feat).sum(dim=-1)

        with torch.no_grad():
            if not self.training:
                score = contrast_logits.softmax(-1)
                pred_idx = torch.argmax(score, dim=1)
                label_proposals_mask_w_max_conf = label_proposals_mask[
                    torch.arange(B), pred_idx, 1:
                ]
                pred = self.decoder.MaxDecodeCA(None, label_proposals_mask_w_max_conf, index_list=label_proposals[torch.arange(B), pred_idx, 1:])[0]

                gt = self.decoder.i2g(label_proposals[:, 0, 1:])
                if "ca_results" in ret:
                    ret_list = ret["ca_results"]
                else:
                    ret_list = [{} for batch_idx in range(B)]
                for batch_idx in range(B):
                    ret_list[batch_idx]["gt"] = gt[batch_idx]
                    inp = self.decoder.i2g(label_proposals[batch_idx, :, 1:])
                    for beam_idx in range(K):
                        ret_list[batch_idx][beam_idx] = {}
                        ret_list[batch_idx][beam_idx]["inp_"] = inp[beam_idx]
                        ret_list[batch_idx][beam_idx]["contrast_score"] = score[batch_idx][
                            beam_idx
                        ].item()
                        ret_list[batch_idx][beam_idx]["contrast_logit"] = contrast_logits[batch_idx][
                            beam_idx
                        ].item()
            else:
                pred = None
                ret_list = []
        return {
            "contrast_logits": contrast_logits,
            "contrast_pred": pred,
            "ca_results": ret_list,
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
                        ret_dict["decoder_ctc_logits"].log_softmax(-1),
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
                pred = ret_dict["decoder_reg_logits"]
                target = ret_dict["visual_feat"]
                mask = ret_dict["mask"] * ret_dict["attention_mask"]
                # mean = target.mean(dim=-1, keepdim=True)
                # var = target.var(dim=-1, keepdim=True)
                # target = (target - mean) / (var + 1.e-6)**.5

                l = 1 - F.cosine_similarity(self.h1(pred), target.detach(), dim=2)
                l1 = (
                    weight * (l * mask).sum() / mask.sum()
                )  # mean loss on removed patches

                l = 1 - F.cosine_similarity(pred.detach(), self.h2(target), dim=2)
                l2 = (
                    weight * (l * mask).sum() / mask.sum()
                )  # mean loss on removed patches

                l = (l1 + l2) * 0.5
            elif k == "SimsiamAlign":
                pred = ret_dict["visual_feat"]
                target = ret_dict["sequence_feat"].permute(1, 0, 2)
                mask = ret_dict["attention_mask"]
                # mean = target.mean(dim=-1, keepdim=True)
                # var = target.var(dim=-1, keepdim=True)
                # target = (target - mean) / (var + 1.e-6)**.5
                l = -F.cosine_similarity(self.h1(pred), target.detach(), dim=2)
                l1 = (
                    weight * (l * mask).sum() / mask.sum()
                )  # mean loss on removed patches

                l = -F.cosine_similarity(pred.detach(), self.h2(target), dim=2)
                l2 = (
                    weight * (l * mask).sum() / mask.sum()
                )  # mean loss on removed patches

                l = (l1 + l2) * 0.5
            elif k == "LenPred":
                pred = ret_dict["len_logits"].reshape(-1)
                target = label_lgt.to(pred) / 50.0
                l = self.loss["MSE"](pred, target)
                loss_kv[f"{phase}/Acc/{k}"] = torch.abs(pred - target).mean()
            elif k == "CADecoder":
                # target = ret_dict["ca_label"].reshape(-1)
                # pred = ret_dict["ca_logits"].reshape(target.shape[0], -1)
                # ce_loss = self.loss["CE"](pred, target)
                # pred_idx = torch.argmax(pred, dim=1)
                # ce_acc_0 = (
                #     torch.eq(pred_idx[target == 0], target[target == 0]).float().mean()
                # )
                # ce_acc_wo_0 = (
                #     torch.eq(pred_idx[target != 0], target[target != 0]).float().mean()
                # )
                # loss_kv[f"{phase}/Loss/{k}-ce_loss"] = ce_loss.item()
                # loss_kv[f"{phase}/Acc/{k}-ce_acc-wo-0"] = ce_acc_wo_0.item()
                # loss_kv[f"{phase}/Acc/{k}-ce_acc-0"] = ce_acc_0.item()
                
                # target = ret_dict["ca_unmatched_label"].reshape(-1)
                # pred = ret_dict["ca_logits"].reshape(target.shape[0], -1)
                # ce_unmatch_loss = self.loss["CE"](pred, target)
                # pred_idx = torch.argmax(pred, dim=1)
                # ce_unmatch_acc = (
                #     torch.eq(pred_idx, target).float().mean()
                # )
                # loss_kv[f"{phase}/Loss/{k}-ce_unmatch_loss"] = ce_unmatch_loss.item()
                # loss_kv[f"{phase}/Acc/{k}-ce_unmatch_acc"] = ce_unmatch_acc.item()
                
                pred = ret_dict["conf_logits"]
                B = pred.shape[0]
                K = pred.shape[1]//2
                pred = torch.cat([pred[:, :1+K], torch.cat([pred[:, :1], pred[:, K+1:]], dim=1)], dim=0)
                target = (
                    torch.zeros(pred.shape[0]).long().to(self.device, non_blocking=True)
                )
                conf_loss = self.loss["CE"](pred, target)
                conf_hard_acc = torch.eq(torch.argmax(pred[:B], dim=1), target[:B]).float().mean()
                conf_easy_acc = torch.eq(torch.argmax(pred[B:], dim=1), target[B:]).float().mean()
                loss_kv[f"{phase}/Loss/{k}-conf_loss"] = conf_loss.item()
                loss_kv[f"{phase}/Acc/{k}-conf_hard_acc"] = conf_hard_acc.item()
                loss_kv[f"{phase}/Acc/{k}-conf_easy_acc"] = conf_easy_acc.item()

                pred = ret_dict["contrast_logits"]
                B = pred.shape[0]
                K = pred.shape[1]//2
                pred = torch.cat([pred[:, :1+K], torch.cat([pred[:, :1], pred[:, K+1:]], dim=1)], dim=0)
                contrast_loss = self.loss["CE"](pred, target)
                contrast_hard_acc = torch.eq(torch.argmax(pred[:B], dim=1), target[:B]).float().mean()
                contrast_easy_acc = torch.eq(torch.argmax(pred[B:], dim=1), target[B:]).float().mean()
                loss_kv[f"{phase}/Loss/{k}-contrast_loss"] = contrast_loss.item()
                loss_kv[f"{phase}/Acc/{k}-contrast_hard_acc"] = contrast_hard_acc.item()
                loss_kv[f"{phase}/Acc/{k}-contrast_easy_acc"] = contrast_easy_acc.item()

                l = conf_loss + contrast_loss
            elif k == "SignGlossContrast":

                pred = ret_dict["contrast_logits"]
                B = pred.shape[0]
                K = pred.shape[1]//2
                pred = torch.cat([pred[:, :1+K], torch.cat([pred[:, :1], pred[:, K+1:]], dim=1)], dim=0)
                target = (
                    torch.zeros(pred.shape[0]).long().to(self.device, non_blocking=True)
                )
                contrast_loss = self.loss["CE"](pred, target)
                contrast_hard_acc = torch.eq(torch.argmax(pred[:B], dim=1), target[:B]).float().mean()
                contrast_easy_acc = torch.eq(torch.argmax(pred[B:], dim=1), target[B:]).float().mean()
                loss_kv[f"{phase}/Loss/{k}-contrast_loss"] = contrast_loss.item()
                loss_kv[f"{phase}/Acc/{k}-contrast_hard_acc"] = contrast_hard_acc.item()
                loss_kv[f"{phase}/Acc/{k}-contrast_easy_acc"] = contrast_easy_acc.item()
                l = contrast_loss

            loss_kv[f"{phase}/Loss/{k}"] = l.item()
            if not (np.isinf(l.item()) or np.isnan(l.item())):
                loss += l
            else:
                print("NAN")
        return loss, loss_kv

    def criterion_init(self):
        self.loss["CTCLoss"] = torch.nn.CTCLoss(reduction="none", zero_infinity=False)
        self.loss["distillation"] = SeqKD(T=8)
        self.loss["MSE"] = nn.MSELoss()
        self.loss["CE"] = nn.CrossEntropyLoss()
        weight = torch.ones(self.num_classes).to(self.device, non_blocking=True)
        weight[0] = 1 / 10.0
        self.loss["weighted-CE"] = nn.CrossEntropyLoss(weight=weight)

        return self.loss

    def forward(
        self,
        x,
        len_x,
        label=None,
        label_lgt=None,
        label_proposals=None,
        label_proposals_mask=None,
        return_loss=False,
        phase="val",
    ):
        x = x.to(self.device, non_blocking=True)
        # label = label.to(self.device, non_blocking=True)
        res = self.forward_conv_layer(x, len_x)
        if "DecoderReg" in self.loss_weights or "DecoderCTC" in self.loss_weights:
            res.update(self.forward_masked_encoder(res))
        if "SeqCTC" in self.loss_weights or "Dist" in self.loss_weights:
            res.update(self.forward_encoder(res))
        if "DecoderReg" in self.loss_weights:
            res.update(self.forward_reg_decoder(res))
        if "DecoderCTC" in self.loss_weights:
            res.update(self.forward_ctc_decoder(res))
        if "CADecoder" in self.loss_weights:
            label_proposals = label_proposals.to(self.device, non_blocking=True)
            label_proposals_mask = label_proposals_mask.to(
                self.device, non_blocking=True
            )
            res.update(
                self.forward_ca_decoder(res, label_proposals, label_proposals_mask)
            )
        if "SignGlossContrast" in self.loss_weights:
            label_proposals = label_proposals.to(self.device, non_blocking=True)
            label_proposals_mask = label_proposals_mask.to(
                self.device, non_blocking=True
            )
            res.update(
                self.forward_contrast(res, label_proposals, label_proposals_mask)
            )
        if return_loss:
            return self.criterion_calculation(res, label, label_lgt, phase=phase)
        else:
            return res, self.criterion_calculation(res, label, label_lgt, phase=phase)
