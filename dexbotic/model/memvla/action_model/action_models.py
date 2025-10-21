# --------------------------------------------------------
# Copyright (c) 2025 Dexmal. All Rights Reserved.
# --------------------------------------------------------
# Modified from microsoft's CogACT repos (https://github.com/microsoft/CogACT)
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------


import torch
import torch.nn as nn
from .dit import DiT
from .diffusion import create_diffusion, ModelVarType


def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)


def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)


# Model size
DiT_models = {'DiT-S': DiT_S, 'DiT-B': DiT_B, 'DiT-L': DiT_L}


class ActionModel(nn.Module):
    def __init__(self,
                 token_size,
                 model_type,
                 in_channels,
                 future_action_window_size,
                 diffusion_steps=100,
                 noise_schedule='squaredcos_cap_v2',
                 use_per_attn=False,
                 per_token_size=None,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.noise_schedule = noise_schedule
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [
                ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.future_action_window_size = future_action_window_size
        self.model_type = model_type
        self.net = DiT_models[model_type](
            token_size=token_size,
            in_channels=in_channels,
            class_dropout_prob=0.1,
            learn_sigma=learn_sigma,
            future_action_window_size=future_action_window_size,
            use_per_attn=use_per_attn,
            per_token_size=per_token_size,
        )

    # Given condition z and ground truth token x, compute loss
    def loss(self, x, z, per_token):
        # sample random noise and timestep
        noise = torch.randn_like(x)  # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps,
                                 (x.size(0),), device=x.device)

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)
        # print("xxxx", x_t.dtype, timestep.dtype, z.dtype)
        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, per_token)

        assert noise_pred.shape == noise.shape == x.shape
        # Compute L2 loss
        loss = ((noise_pred - noise) ** 2).mean()
        # Optional: loss += loss_vlb

        return loss

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim" + str(ddim_step),
            noise_schedule=self.noise_schedule,
            diffusion_steps=self.diffusion_steps,
            sigma_small=True,
            learn_sigma=False)
        return self.ddim_diffusion
