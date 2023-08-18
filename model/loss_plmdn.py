import torch
import torch.nn as nn
import math

import sys
sys.path.append("/work/Git/Phone-Level-Mixture-Density-Network-for-TTS/")
from hautils.util import get_mask_from_lengths_plpm


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
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
        
    def mdn_loss(self, w, sigma, mu, target, mask=None):
        """
        w -- [B, src_len, num_gaussians]
        sigma -- [B, src_len, num_gaussians, out_features]
        mu -- [B, src_len, num_gaussians, out_features]
        target -- [B, src_len, out_features]
        mask -- [B, src_len]
        """
        prob = w.unsqueeze(-1) * self.gaussian_probability(sigma, mu, target, mask)
        nll = -torch.log(torch.sum(prob, dim=2))
        if mask is not None:
            nll = nll.masked_fill(mask.unsqueeze(-1), 0)
        l_pp = torch.sum(nll, dim=1)
        return torch.mean(l_pp)

    def forward(self, inputs, predictions):
        if inputs is not None:
            (
                src_lens,
                _,
                mel_targets,
                _,
                _,
                pitch_targets,
                energy_targets,
                duration_targets,
            ) = inputs[4:]
            (
                mel_predictions,
                postnet_mel_predictions,
                pitch_predictions,
                energy_predictions,
                log_duration_predictions,
                _,
                src_masks,
                mel_masks,
                _,
                _,
                prosody_info,
            ) = predictions
            
            src_masks = ~src_masks
            mel_masks = ~mel_masks
            log_duration_targets = torch.log(duration_targets.float() + 1)
            mel_targets = mel_targets[:, : mel_masks.shape[1], :]
            mel_masks = mel_masks[:, :mel_masks.shape[1]]

            log_duration_targets.requires_grad = False
            pitch_targets.requires_grad = False
            energy_targets.requires_grad = False
            mel_targets.requires_grad = False

            if self.pitch_feature_level == "phoneme_level":
                pitch_predictions = pitch_predictions.masked_select(src_masks)
                pitch_targets = pitch_targets.masked_select(src_masks)
            elif self.pitch_feature_level == "frame_level":
                pitch_predictions = pitch_predictions.masked_select(mel_masks)
                pitch_targets = pitch_targets.masked_select(mel_masks)

            if self.energy_feature_level == "phoneme_level":
                energy_predictions = energy_predictions.masked_select(src_masks)
                energy_targets = energy_targets.masked_select(src_masks)
            if self.energy_feature_level == "frame_level":
                energy_predictions = energy_predictions.masked_select(mel_masks)
                energy_targets = energy_targets.masked_select(mel_masks)

            log_duration_predictions = log_duration_predictions.masked_select(src_masks)
            log_duration_targets = log_duration_targets.masked_select(src_masks)

            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

            pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            energy_loss = self.mse_loss(energy_predictions, energy_targets)
            duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
        else:
            src_lens, prosody_info = predictions
        
        # PLPDN Loss
        w, sigma, mu, prosody_embeddings = prosody_info
        src_masks = get_mask_from_lengths_plpm(src_lens)
        src_masks = (1-src_masks.type(torch.int)).type(torch.bool)
        mdn_loss = 0.02 * self.mdn_loss(w, sigma, mu, prosody_embeddings.detach(), src_masks)
        
        
        if inputs is not None:

            total_loss = (
                mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + mdn_loss
            )

            return (
                total_loss,
                mel_loss,
                postnet_mel_loss,
                pitch_loss,
                energy_loss,
                duration_loss,
                mdn_loss,
            )

        else:
            return mdn_loss