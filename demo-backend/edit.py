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
from dataset import HuchuDataset, draw_landmarks
from torchvision.transforms import ToTensor
from masactrl.dit_utils import FACEPipeline


TARGET_LD = [[30,74],
            [
                30,
                101
            ],
            [
                33,
                128
            ],
            [
                39,
                154
            ],
            [
                48,
                178
            ],
            [
                60,
                199
            ],
            [
                78,
                217
            ],
            [
                104,
                223
            ],
            [
                128,
                223
            ],
            [
                154,
                223
            ],
            [
                178,
                220
            ],
            [
                199,
                208
            ],
            [
                214,
                193
            ],
            [
                223,
                169
            ],
            [
                229,
                145
            ],
            [
                232,
                119
            ],
            [
                232,
                92
            ],
            [
                47,
                50
            ],
            [
                62,
                38
            ],
            [
                80,
                35
            ],
            [
                98,
                38
            ],
            [
                113,
                47
            ],
            [
                151,
                47
            ],
            [
                166,
                41
            ],
            [
                184,
                41
            ],
            [
                199,
                44
            ],
            [
                214,
                56
            ],
            [
                130,
                74
            ],
            [
                128,
                98
            ],
            [
                125,
                119
            ],
            [
                125,
                139
            ],
            [
                107,
                148
            ],
            [
                116,
                151
            ],
            [
                125,
                157
            ],
            [
                136,
                154
            ],
            [
                145,
                151
            ],
            [
                65,
                74
            ],
            [
                77,
                68
            ],
            [
                92,
                71
            ],
            [
                104,
                83
            ],
            [
                89,
                83
            ],
            [
                74,
                80
            ],
            [
                160,
                86
            ],
            [
                172,
                74
            ],
            [
                187,
                74
            ],
            [
                199,
                80
            ],
            [
                187,
                86
            ],
            [
                172,
                86
            ],
            [
                89,
                178
            ],
            [
                101,
                172
            ],
            [
                116,
                169
            ],
            [
                125,
                175
            ],
            [
                136,
                172
            ],
            [
                154,
                175
            ],
            [
                169,
                178
            ],
            [
                154,
                193
            ],
            [
                139,
                199
            ],
            [
                125,
                202
            ],
            [
                113,
                199
            ],
            [
                101,
                193
            ],
            [
                95,
                178
            ],
            [
                113,
                181
            ],
            [
                125,
                184
            ],
            [
                136,
                181
            ],
            [
                166,
                181
            ],
            [
                136,
                184
            ],
            [
                125,
                187
            ],
            [
                113,
                184
            ]
        ]

TARGET_LANDMARK = draw_landmarks({"landmark": TARGET_LD})
TARGET_LANDMARK = ToTensor()(TARGET_LANDMARK)[0:1]

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
    # exceptional prompt inversion
    # new_prompt = "a picture of a woman with long blond hair."
    # new_landmark_img = dataset[0][1].to(device).unsqueeze(0)
    new_landmark_img = TARGET_LANDMARK.to(device).unsqueeze(0)
    for i, (x, landmark_img, y) in enumerate(dataloader):
        x = x.to(device)
        landmark_img = landmark_img.to(device)

        z, intermediates = pipeline.invert(x, landmark_img, list(y), return_intermediate=True)
        z = torch.cat([z]*2, 0).to(device)
        samples = pipeline.edit(y[0], y[0], new_landmark_img, landmark_img, z, cfg_scale=args.cfg_scale, intermediates=intermediates[::-1])
        # Save and display images:
        # save_image(samples, f"{args.inversion}_inversion_{args.seed}_ppl_sample_{i}.png", nrow=4, normalize=True, value_range=(-1, 1))
        
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