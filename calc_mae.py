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
from tqdm.auto import tqdm 
from models import TextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer
from dataset import HuchuDataset, draw_landmarks
from masactrl import FACEPipeline
from torchvision.transforms import ToTensor



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
    
    pipeline = FACEPipeline(model, diffusion, vae, inversion=args.inversion, real_step=args.num_sampling_steps, device=device)
    MAE = 0.
    for i, (x, landmark_img, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x = x.to(device)
        landmark_img = landmark_img.to(device)
        z_T, intermediates = pipeline.invert(x, landmark_img, list(y), return_intermediate=False, progress=False)
        z_0 = pipeline.sample(list(y), landmark_img, latent=z_T, save_img=False, progress=False)
        x_0 = vae.decode(z_0 / 0.18215).sample 
        MAE += torch.nn.functional.l1_loss(x, x_0)
    
    MAE /= len(dataloader)
    print(f"Mean Absolute Error: {MAE} on {args.data_path} with {args.ann_path}")
    with open(f"mae_{args.inversion}_STEP{args.num_sampling_steps}_CELEBAHQ.txt", "w") as f:
        f.write(MAE)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ann-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/2-Text")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--inversion", type=str, default="exceptional", choices=["exceptional", "unconditional", "npi", "cfg"])
    args = parser.parse_args()
    main(args)
