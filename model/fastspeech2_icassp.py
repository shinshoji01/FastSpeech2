import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformers import BertForMaskedLM, BertConfig
from transformer import Encoder, Decoder, PostNet
from .modules_p import VarianceAdaptor
from utils.tools import get_mask_from_lengths

import sys
sys.path.append("/work/Git/GST-Tacotron/")
from GST import GST

sys.path.append("/work/Git/Phone-Level-Mixture-Density-Network-for-TTS/")
from core.gmm_mdn import ProsodyExtractor, ProsodyPredictor
from hautils.util import get_mask_from_lengths_plpm
from transformers.modeling_outputs import MaskedLMOutput

class FastSpeech2BERT(nn.Module):
    def __init__(self, model_config, BERT_config):
        super(FastSpeech2BERT, self).__init__()
        self.encoder = Encoder(model_config)
        bert = BertForMaskedLM(BERT_config)
        self.cls = bert.cls
        self.BERT_config = BERT_config
        self.loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
    
    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        token_type_ids= None,
        position_ids= None,
        head_mask= None,
        inputs_embeds= None,
        encoder_hidden_states= None,
        encoder_attention_mask= None,
        labels= None,
        output_attentions= None,
        output_hidden_states= None,
        return_dict= None,
        only_output=False,
    ):
        
        device = input_ids.device
        src_lens = torch.sum(attention_mask, axis=1).to(device)
        max_src_len = src_lens.max().to(device)
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        outputs = self.encoder(input_ids, src_masks)
        if only_output:
            return outputs
        else:
            prediction_scores = self.cls(outputs)

            masked_lm_loss = self.loss_fct(prediction_scores.view(-1, self.BERT_config.vocab_size), labels.view(-1))

            prediction_scores = self.cls(outputs)

            return MaskedLMOutput(loss=masked_lm_loss,
                                  logits=prediction_scores,
                                  hidden_states=outputs)

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
            self.ifencoder = "/Encoder" in pretrained_path
            if self.ifencoder:
                self.encoder = FastSpeech2BERT(model_config, config)
                model = torch.load(pretrained_path, map_location="cpu")
                self.encoder.load_state_dict(model)
            else:
                self.encoder = BertForMaskedLM.from_pretrained(pretrained_path)
        else:
            self.encoder = Encoder(model_config) # Original FastSpeech2 Encoder

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
            if self.ifencoder:
                hsize = 256
            else: 
                hsize = 512
                # hsize = 768
            self.bert_linear = nn.Sequential(
                nn.Linear(hsize, model_config["transformer"]["encoder_hidden"]),
                nn.Tanh(),
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
            
            # output = self.encoder.bert(**data).last_hidden_state[:, 1:-1, :]
            if self.ifencoder:
                data["labels"] = None
                data["only_output"] = True
                output = self.encoder(**data)
                output = output[:,1:-1,:]
            else:
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