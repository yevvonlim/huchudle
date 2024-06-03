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
from dataset.HuchuDataset import HuchuDataset


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        # assert args.num_classes == 1000
    dataset = HuchuDataset(ann_path=args.ann_path, root_dir=args.data_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    # exceptional prompt inversion
    
    for x, landmark_img, y in dataloader:
        model.remove_pos_emb(device)
        x = x.to(device)
        batch_size = x.shape[0]
        landmark_img = landmark_img.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)        
        # Exceptional prompt
        tokens = torch.zeros(batch_size, 77).to(device, torch.int64) + 7788
        real_step = 500
        model_kwargs = dict(y="", landmark=torch.zeros_like(landmark_img).to(device), token=tokens)
        # model_kwargs = dict(y=y[0], landmark=landmark_img)
        z_tau = diffusion.ddim_reverse_sample_loop(model.forward,(batch_size, 3, 256, 256), x, clip_denoised=False, model_kwargs=model_kwargs, device=device, real_step=real_step)
        z = torch.cat([z_tau, z_tau], 0)
        landmark_img = torch.cat([landmark_img, landmark_img], 0)
        y = ("",) + ("",) #text discription 안준거 
        # y = y + ("", ) # 준거 
        model_kwargs = dict(y=list(y), landmark=landmark_img, cfg_scale=args.cfg_scale)
        model.retain_orig_pos_emb()
        with torch.no_grad():
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, real_step=real_step
            )
            samples,_ = samples.chunk(2, dim=0)
        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample
        noise = vae.decode(z_tau / 0.18215).sample
    

        # Save and display images:
        save_image(samples, f"inversion_{args.seed}.png", nrow=4, normalize=True, value_range=(-1, 1))
        save_image(noise, f"noise_{args.seed}.png", nrow=4, normalize=True, value_range=(-1, 1))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2-Text")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=12.0)
    parser.add_argument("--num-sampling-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
