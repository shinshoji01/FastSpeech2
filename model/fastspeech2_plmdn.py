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
        if self.include_ed:
            if self.ed_combination=="addition":
                self.ed_embedding = nn.Sequential(nn.Linear(self.ed_bool_list.sum(), model_config["transformer"]["encoder_hidden"]), nn.Tanh())
                # self.ed_embedding = nn.Linear(self.ed_bool_list.sum(), model_config["transformer"]["encoder_hidden"])
            elif self.ed_combination=="concat_embedding":
                self.ed_embedding = nn.Sequential(nn.Linear(self.ed_bool_list.sum(), model_config["ed"]["concatenation_embedding_size"]), nn.Tanh())
            elif self.ed_combination=="concatenation":
                pass
            else:
                assert False, "'combination' in model_config should be either 'concatenation' or 'addition'"
                
        # Global Style Token
        self.include_gst = model_config["gst"]["include_gst"]
        if self.include_gst:
            self.gst = GST()
            
        # Phoneme-level Prosody Modeling
        self.include_plpm = model_config["plpm"]["include_plpm"]
        if self.include_plpm:
            encoder_hidden = model_config["transformer"]["encoder_hidden"]
            mel_hidden = model_config["plpm"]["mel_hidden"]
            self.prosody_extractor = ProsodyExtractor(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                d_model=mel_hidden,
                kernel_size=9,
            )
            self.prosody_predictor = ProsodyPredictor(
                d_model=encoder_hidden,
                kernel_size=[9,5],
                num_gaussians=model_config["plpm"]["num_gaussians"],
                dropout=0.2,
            )
            self.prosody_linear = torch.nn.Linear(2 * mel_hidden, encoder_hidden)
            self.w = None
            self.sigma = None
            self.mu = None
            self.prosody_embeddings = None
    
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
        inference_plpm=False,
        only_plpm=False,
    ):
        if self.include_plpm and only_plpm:
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
            ml = mel_lens if mel_lens is not None else inf_mlens
            dt = d_targets if mel_lens is not None else inf_dts
            src_masks_plpm = get_mask_from_lengths_plpm(src_lens)
            src_masks_plpm = (1-src_masks_plpm.type(torch.int)).type(torch.bool)
            w, sigma, mu = self.prosody_predictor(output.detach(), src_masks_plpm)
            if inference_plpm:
                prosody_embeddings = self.prosody_predictor.sample(w, sigma, mu)
            else:
                prosody_embeddings = self.prosody_extractor(mels, ml, dt, src_lens)
            output = output + self.prosody_linear(prosody_embeddings)
            if not training:
                max_mel_len = None
                d_targets = None
                mel_masks = None

            return (w, sigma, mu, prosody_embeddings)
        
        else:
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

            if self.include_ed:
                if self.ed_combination=="concatenation":
                    output = torch.cat([output, eds], axis=2)
                elif self.ed_combination=="concat_embedding":
                    output = torch.cat([output, self.ed_embedding(eds)], axis=2)
                elif self.ed_combination=="addition":
                    output = output + self.ed_embedding(eds)
                else:
                    assert False, "'combination' in model_config should be either 'concatenation' or 'addition'"

            if self.include_gst:
                style_embed = self.gst(mels)
                style_embed = style_embed.expand_as(output)
                output = output + style_embed

            if self.include_plpm:
                ml = mel_lens if mel_lens is not None else inf_mlens
                dt = d_targets if mel_lens is not None else inf_dts
                src_masks_plpm = get_mask_from_lengths_plpm(src_lens)
                src_masks_plpm = (1-src_masks_plpm.type(torch.int)).type(torch.bool)
                w, sigma, mu = self.prosody_predictor(output, src_masks_plpm)
                if inference_plpm:
                    prosody_embeddings = self.prosody_predictor.sample(w, sigma, mu)
                else:
                    prosody_embeddings = self.prosody_extractor(mels, ml, dt, src_lens)
                output = output + self.prosody_linear(prosody_embeddings)
                if not training:
                    max_mel_len = None
                    d_targets = None
                    mel_masks = None

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
                p_predictions,
                e_predictions,
                log_d_predictions,
                d_rounded,
                src_masks,
                mel_masks,
                src_lens,
                mel_lens,
                (w, sigma, mu, prosody_embeddings),
            )