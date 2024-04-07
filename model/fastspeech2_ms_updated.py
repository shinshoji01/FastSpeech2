import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths

import sys
sys.path.append("/work/Git/GST-Tacotron/")
from GST import GST

sys.path.append("/work/Git/Phone-Level-Mixture-Density-Network-for-TTS/")
from core.gmm_mdn import ProsodyExtractor, ProsodyPredictor
from hautils.util import get_mask_from_lengths_plpm

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        # Perform the transpose operation
        x_transposed = torch.transpose(x, self.dim0, self.dim1)
        return x_transposed
    
class Mean(nn.Module):
    def __init__(self, dim0):
        super(Mean, self).__init__()
        self.dim0 = dim0

    def forward(self, x):
        # Perform the transpose operation
        x_mean = x.mean(axis=self.dim0)
        return x_mean

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

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
        # Emotion Distribution
        self.include_ed = model_config["ed"]["include_ed"]
        self.ed_combination = model_config["ed"]["combination"]
        self.ed_bool_list = np.array(model_config["ed"]["phonemes_words_utterance"]).repeat(4)
                
        # MsEmoTTS
        self.include_msemotts = model_config["msemotts"]["include_msemotts"]
        if self.include_msemotts:
            
            mel_num = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
            enc_dim = model_config["transformer"]["encoder_hidden"]
            self.text_variation_encoder = nn.Sequential(
                Transpose(2,1),
                nn.Conv1d(enc_dim, enc_dim//2, 5, 1, 2),
                nn.ReLU(),
                Transpose(2,1),
                nn.LayerNorm(enc_dim//2),
                nn.Linear(enc_dim//2, enc_dim//2),
                nn.ReLU(),
                nn.Linear(enc_dim//2, enc_dim//2),
            )
            
            self.mel_variation_encoder = nn.Sequential(
                Transpose(2,1),
                nn.Conv1d(mel_num, enc_dim//4, 5, 1, 2),
                nn.Conv1d(enc_dim//4, enc_dim//2, 5, 1, 2),
                Transpose(2,1),
                nn.LayerNorm(enc_dim//2),
                nn.Dropout(),
                Mean(1),
            )
            
            self.ed_embedding = nn.Sequential(
                nn.Linear(1, enc_dim),
                nn.Tanh()
            )
            
            self.text_ed_embedding = nn.Sequential(
                Transpose(2,1),
                nn.Conv1d(enc_dim, enc_dim//2, 5, 1, 2),
                nn.ReLU(),
                Transpose(2,1),
                nn.LayerNorm(enc_dim//2),
                nn.Linear(enc_dim//2, enc_dim//2),
                nn.ReLU(),
                nn.Linear(enc_dim//2, 1),
                nn.Sigmoid(),
            )
            
            self.el_embedding = nn.Embedding(4, enc_dim//2)
            
            
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
        els=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        inference=False,
        ed_gt=False,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        
        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
        
        if self.include_msemotts:
            uv_predictions = self.text_variation_encoder(output)
            uv_embedding = self.mel_variation_encoder(mels)
            uv_embedding = uv_embedding.unsqueeze(1).repeat((1,output.shape[1],1))
            
            shape = eds.shape
            els_numpy = els.detach().cpu().numpy()
            bool_list = torch.tensor(np.eye(12)[els_numpy].astype(bool)).unsqueeze(1).repeat((1,shape[1],1))
            new_eds = eds[bool_list].view(shape[0],shape[1],1)
            ed_embedding = self.ed_embedding(new_eds)
            ed_predictions = self.text_ed_embedding(output)
            
            emotion_embedding = self.el_embedding(els)
            emotion_embedding = emotion_embedding.unsqueeze(1).repeat((1,output.shape[1],1))
            
            if inference:
                if ed_gt:
                    output = torch.concat([ed_embedding+output, uv_predictions, emotion_embedding], axis=2)
                else:
                    ed_predictions = self.ed_embedding(ed_predictions)
                    output = torch.concat([ed_predictions+output, uv_predictions, emotion_embedding], axis=2)
            else:
                output = torch.concat([ed_embedding+output, uv_embedding, emotion_embedding], axis=2)
            
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

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            uv_embedding,
            uv_predictions,
            ed_embedding,
            ed_predictions,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )