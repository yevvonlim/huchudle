"""
Util functions based on Diffuser framework.
"""


import os
import torch
import cv2
import numpy as np

import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision.utils import save_image
from torchvision.io import read_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from download import find_model
from models import DiT_models
import argparse
from masactrl.masactrl import MutualSelfAttentionControl
from masactrl.masactrl_utils import register_attention_editor_dit

'''FACE: Facial Attribute Control and Editing
'''

class FACEPipeline:
    def __init__(self, model, diffusion, vae, device, inversion="exceptional", real_step=50):
        self.model = model
        self.diffusion = diffusion
        self.vae = vae
        self.inversion = inversion
        self.real_step = real_step
        self.device = device


    def invert(self, image, landmark_img, prompt, return_intermediate=False):
        device = self.device
        with torch.no_grad():
            x = self.vae.encode(image).latent_dist.sample().mul_(0.18215)
        if self.inversion == "exceptional":
            self.model.exceptional_prompt = self.model.exceptional_prompt.to(device)
            model_kwargs = dict(y="", landmark=landmark_img, text_emb=self.model.exceptional_prompt)
        elif self.inversion == "unconditional":
            model_kwargs = dict(y="", landmark=torch.zeros_like(landmark_img))
        elif self.inversion == "npi" or self.inversion == "cfg":
            model_kwargs = dict(y=list(prompt), landmark=landmark_img)
        
        z, intermediates = self.diffusion.ddim_reverse_sample_loop(self.model.forward,
                                                    (image.shape[0], 3, 256, 256),
                                                    x,
                                                    clip_denoised=False,
                                                    model_kwargs=model_kwargs,
                                                    device=device,
                                                    real_step=self.real_step,
                                                    progress=True,
                                                    return_intermediate=return_intermediate,)
    
        return z, intermediates
    

    def sample(self, prompt:str, landmark_img, latent=None, save_img=True):
        if latent is None:
            z = torch.randn(1, 4, 256//8, 256//8, device=self.device)
        else:
            z = latent
        
        if self.inversion == "cfg":
            z = torch.cat([z, z], 0)
            landmark_img = torch.cat([landmark_img, torch.zeros_like(landmark_img)], 0)
            y = y + ("",)
            model_kwargs = dict(y=list(y), landmark=landmark_img, cfg_scale=5.)
            with torch.no_grad():
                samples = self.diffusion.ddim_sample_loop(
                    self.model.forward_with_cfg,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=self.device,
                    real_step=self.real_step)
        else:
            model_kwargs = dict(y=list(prompt), landmark=landmark_img)
            with torch.no_grad():
                samples = self.diffusion.ddim_sample_loop(
                    self.model.forward,
                    z.shape,
                    z,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    progress=True,
                    device=self.device,
                    real_step=self.real_step)
        
        samples = self.vae.decode(samples / 0.18215).sample
        if save_img:
            save_image(samples, f"sample_{prompt}.png", nrow=4, normalize=True, value_range=(-1, 1))
        return samples

    
    def edit(self, new_prompt:str, orig_prompt:str,  new_landmark_img, orig_landmark_img, latent, intermediates=[], cfg_scale=5., save_img=True):
        z = torch.cat([latent, latent], 0).to(self.device)
        editor = MutualSelfAttentionControl(model_type="DiT-L-Text", start_step=0, start_layer=0)
        register_attention_editor_dit(self.model, editor)
        y = [orig_prompt, new_prompt, orig_prompt, new_prompt]
        landmark_img = torch.cat([orig_landmark_img, 
                                  new_landmark_img, 
                                  orig_landmark_img,
                                  new_landmark_img], 0).to(self.device)
                                #   *[torch.zeros_like(orig_landmark_img).to(self.device)]*2], 0).to(self.device)
        model_kwargs = dict(y=y, landmark=landmark_img, cfg_scale=cfg_scale)
        with torch.no_grad():
            samples = self.diffusion.ddim_sample_loop(
                self.model.forward_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=self.device,
                real_step=self.real_step,
                intermediates=intermediates)
        samples, _ = samples.chunk(2, dim=0)
        samples = self.vae.decode(samples / 0.18215).sample
        if save_img:
            save_image(samples, f"edit_{new_prompt}.png", nrow=4, normalize=True, value_range=(-1, 1))

        return samples

    
    
   