import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import BertForMaskedLM, BertConfig
from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

import sys
sys.path.append("/work/Git/GST-Tacotron/")
from GST import GST

sys.path.append("/work/Git/Phone-Level-Mixture-Density-Network-for-TTS/")
from core.gmm_mdn import ProsodyExtractor, ProsodyPredictor
from hautils.util import get_mask_from_lengths_plpm

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.ifbert = model_config["transformer"]["encoder_architecture"]=="BERT"
        
        if self.ifbert:
            config = BertConfig(
                vocab_size=101,  
                hidden_size=256,  
                num_hidden_layers=4,
                num_attention_heads=2, 
                max_position_embeddings=512,  
            )
            pretrained_path = model_config["transformer"]["BERT_pretrained"]
            self.encoder = BertForMaskedLM.from_pretrained(pretrained_path)
        else:
            self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
            
        if self.ifbert:
            self.bert_linear = nn.Sequential(
                nn.Linear(512, model_config["transformer"]["encoder_hidden"]),
                # nn.Linear(768, model_config["transformer"]["encoder_hidden"]),
                nn.Tanh(),
            )
    
    def gaussian_probability(self, sigma, mu, target, mask=None, eps=1e-8):
        """
            sigma -- [B, src_len, num_gaussians, out_features]
            mu -- [B, src_len, num_gaussians, out_features]
            target -- [B, src_len, out_features]
            mask -- [B, src_len]

            prob -- [B, src_len, num_gaussians, out_features]
        """
        target = target.unsqueeze(2).expand_as(sigma)
        prob = (1.0 / math.sqrt(2 * math.pi)) * torch.exp(-0.5 * ((target - mu) / sigma) ** 2) / (sigma)
        if mask is not None:
            prob = prob.masked_fill(mask.unsqueeze(-1).unsqueeze(-1), 0)
        return prob
                
    def compute_mdn_loss(self, w, sigma, mu, target, mask=None, eps=1e-8):
        """
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        target -- [B, src_len, out_features]
        mask -- [B, src_len]
        """
        prob = w.unsqueeze(-1) * self.gaussian_probability(sigma, mu, target, mask)
        # print(torch.log(torch.sum(prob, dim=2)))
        # nll = -torch.log(torch.sum(prob+eps, dim=2) + eps)
        nll = -torch.log(torch.sum(prob, dim=2) + eps)
        if mask is not None:
            nll = nll.masked_fill(mask.unsqueeze(-1), 0)
        l_pp = torch.sum(nll, dim=1)
        return torch.mean(l_pp)
                

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        eds=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        inf_dts=None,
        inf_mlens=None,
        mdn_loss=0,
        training=True,
        inf=False,
    ):
        if self.ifbert:
            src_lens = src_lens - 2
            max_src_len = max_src_len - 2
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        if self.ifbert:
            data = {
                "input_ids": texts,
                "attention_mask": torch.tensor(texts>0, dtype=int),
                "token_type_ids": None,
                "position_ids": None,
                "head_mask": None,
                "inputs_embeds": None,
                "encoder_hidden_states": None,
                "encoder_attention_mask": None,
                "output_attentions": None,
                "output_hidden_states": None,
                "return_dict": True,
            }
            output = self.encoder.bert(**data)
            output = output[0][:, 1:-1, :]
            output = self.bert_linear(output)
        else:
            output = self.encoder(texts, src_masks)
        
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        # print(mel_masks)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )