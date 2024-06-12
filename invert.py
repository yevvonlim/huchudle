# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from models import TextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer
from dataset import HuchuDataset


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        # assert args.num_classes == 1000
    dataset = HuchuDataset(ann_path=args.ann_path, root_dir=args.data_path, istrain=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        exceptional_prompt=True if args.inversion == "exceptional" else False,
        # num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # exceptional prompt inversion
    
    for i, (x, landmark_img, y) in enumerate(dataloader):
        if args.inversion == "exceptional":
            model.exceptional_prompt = model.exceptional_prompt.to(device)        
        x = x.to(device)
        batch_size = x.shape[0]
        landmark_img = landmark_img.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)        
        real_step = 50
        if args.inversion == "exceptional":
            model_kwargs = dict(y="", landmark=landmark_img, text_emb=model.exceptional_prompt)
        elif args.inversion == "unconditional":
            model_kwargs = dict(y="", landmark=torch.zeros_like(landmark_img))
        elif args.inversion == "npi" or args.inversion == "cfg":
            model_kwargs = dict(y=list(y), landmark=landmark_img)
        z = diffusion.ddim_reverse_sample_loop(model.forward,(batch_size, 3, 256, 256), x, clip_denoised=False, model_kwargs=model_kwargs, device=device, real_step=real_step, progress=True)
        
        if args.inversion == "cfg":
            z = torch.cat([z, z], 0)
            landmark_img = torch.cat([landmark_img, torch.zeros_like(landmark_img)], 0)
            y = y + ("",)
            model_kwargs = dict(y=list(y), landmark=landmark_img, cfg_scale=args.cfg_scale)
            with torch.no_grad():
                samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, real_step=real_step
                )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        else:
            with torch.no_grad():
                samples = diffusion.ddim_sample_loop(
                model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, real_step=real_step
                )
            
        samples = vae.decode(samples / 0.18215).sample
        noise = vae.decode(z / 0.18215).sample
    
        # Save and display images:
        save_image(samples, f"{args.inversion}_inversion_{args.seed}_ppl_sample_{i}.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(noise, f"{args.inversion}_noise_{args.seed}_ppl_sample_{i}.png", nrow=4, normalize=True, value_range=(-1, 1))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2-Text")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--inversion", type=str, default="exceptional", choices=["exceptional", "unconditional", "npi", "cfg"])
    args = parser.parse_args()
    main(args)
