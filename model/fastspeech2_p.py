import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer import Encoder, Decoder, PostNet
from .modules_p import VarianceAdaptor
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
        word_indices=None,
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
        
        (
            output,
            ed_predictions,
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
            eds,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
            word_indices,
        )
        # print(mel_masks)

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
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