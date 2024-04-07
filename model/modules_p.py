import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def AdjustED(ed, words_indices):
    # Utterance-level Emotion
    ed[:,:,8:] = torch.mean(ed[:,:,8:], axis=1)
    # Word-level Emotion
    for i in range(words_indices.max()+1):
        bool_list = words_indices==i
        ed[:,bool_list,4:8] = torch.mean(ed[:,bool_list,4:8], axis=1)
    return ed

class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.ed_predictor = VariancePredictor(model_config, 12)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]
        ed_min = model_config["variance_embedding"]["ed_min"]
        ed_max = model_config["variance_embedding"]["ed_max"]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )
        self.ed_bins = nn.Parameter(
            torch.linspace(ed_min, ed_max, n_bins - 1),
            requires_grad=False,
        )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["decoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["decoder_hidden"]
        )
        self.ed_embedding = nn.ModuleList([nn.Embedding(n_bins, model_config["transformer"]["decoder_hidden"]) for _ in range(12)])
        self.prosody_order = model_config["variance_embedding"]["prosody_order"]
        
    def get_ed_embedding(self, x, target, mask, word_indices=None):
        # print(target)
        prediction = self.ed_predictor(x, mask)
        if target is not None:
            for i in range(3): # levels
                for j in range(4): # emotions
                    idx = i*4+j
                    if idx==0:
                        embedding = self.ed_embedding[idx](torch.bucketize(target[:,:,idx], self.ed_bins))
                    else:
                        embedding += self.ed_embedding[idx](torch.bucketize(target[:,:,idx], self.ed_bins))
            # print(target)
        else:
            # prediction[:] = 1
            prediction = AdjustED(prediction, word_indices)
            for i in range(3): # levels
                for j in range(4): # emotions
                    idx = i*4+j
                    if idx==0:
                        embedding = self.ed_embedding[idx](torch.bucketize(prediction[:,:,idx], self.ed_bins))
                    else:
                        embedding += self.ed_embedding[idx](torch.bucketize(prediction[:,:,idx], self.ed_bins))
            # print(prediction)
        # print(embedding)
        return prediction, embedding/12

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        max_len=None,
        ed_target=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        word_indices=None,
    ):
        ed_prediction = None
        pitch_prediction = None
        energy_prediction = None
        log_duration_prediction = None
        duration_rounded = None
        
        for prosody in self.prosody_order:
            if prosody=="ed":
                ed_prediction, ed_embedding = self.get_ed_embedding(
                    x, ed_target, src_mask, word_indices,
                )
                x = x + ed_embedding
                
            elif prosody=="duration":
                log_duration_prediction = self.duration_predictor(x, src_mask)
        
            elif prosody=="pitch":
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    x, pitch_target, src_mask, p_control
                )
                x = x + pitch_embedding

            elif prosody=="energy":
                energy_prediction, energy_embedding = self.get_energy_embedding(
                    x, energy_target, src_mask, p_control
                )
                x = x + energy_embedding
                
            else:
                assert False, "The elements in prosody_list are not registered."

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        return (
            x,
            ed_prediction,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config, out_dim=1):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["decoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, out_dim)
        self.ed = True if out_dim==12 else False

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            if self.ed:
                mask = mask.unsqueeze(-1).repeat((1,1,12))
            out = out.masked_fill(mask, 0.0) 
            
        if self.ed:
            out = F.sigmoid(out)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
