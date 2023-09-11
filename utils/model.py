import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim

def get_model_PLPM(args, configs, device, train=False, power=-0.5):
    (preprocess_config, model_config, train_config) = configs

    base_model = FastSpeech2(preprocess_config, model_config).to(device)
    del base_model.prosody_predictor
    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path)
        base_model.load_state_dict(ckpt["model"], strict=False)
        print("load")
        model.load_state_dict(base_model.state_dict(), strict=False)

    scheduled_optim = ScheduledOptim(model, train_config, model_config, args.restore_step, "speaker_emb", power=power)
    model.train()
    return model, scheduled_optim
    # return base_model, scheduled_optim

def get_model(args, configs, device, train=False, power=-0.5, ckpt_check=True):
    (preprocess_config, model_config, train_config) = configs
    finetune = train_config["finetune"]["finetune"]
    multi_speaker = model_config["multi_speaker"]

    model = FastSpeech2(preprocess_config, model_config).to(device)
    same = True
    if finetune + args.restore_step:
        if args.restore_step:
            ckpt_path = os.path.join(
                train_config["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step),
            )
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
        elif finetune:
            ckpt_path = train_config["finetune"]["pretrained_path"]
            ckpt = torch.load(ckpt_path)
            if ckpt_check:
                model.load_state_dict(ckpt["model"])
            else:
                try:
                    model.load_state_dict(ckpt["model"])
                except RuntimeError:
                    model.load_state_dict(ckpt["model"], strict=False)

    if train:
        scheduled_optim = ScheduledOptim(model, train_config, model_config, args.restore_step, "speaker_emb", power=power)
        if finetune:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        if multi_speaker:
            parameters = [p for n, p in model.named_parameters() if "speaker_emb" in n]
            scheduled_optim._optimizer.add_param_group({"params": parameters})
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
            
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

# def get_model(args, configs, device, train=False):
#     (preprocess_config, model_config, train_config) = configs

#     model = FastSpeech2(preprocess_config, model_config).to(device)
#     if args.restore_step:
#         ckpt_path = os.path.join(
#             train_config["path"]["ckpt_path"],
#             "{}.pth.tar".format(args.restore_step),
#         )
#         ckpt = torch.load(ckpt_path)
#         model.load_state_dict(ckpt["model"])

#     if train:
#         scheduled_optim = ScheduledOptim(
#             model, train_config, model_config, args.restore_step
#         )
#         if args.restore_step:
#             scheduled_optim.load_state_dict(ckpt["optimizer"])
#         model.train()
#         return model, scheduled_optim

#     model.eval()
#     model.requires_grad_ = False
#     return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open(config["vocoder"]["hifigan_config"], "r") as f:
            config_hifigan = json.load(f)
        config_hifigan = hifigan.AttrDict(config_hifigan)
        vocoder = hifigan.Generator(config_hifigan)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        elif speaker == "0001": # 
            ckpt = torch.load(config["vocoder"]["hifigan_pretrained_model"])
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
