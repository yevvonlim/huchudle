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


from sanic import Sanic
from sanic.response import text
from sanic import Request, Sanic, response
import os
from sanic import json
from sanic import HTTPResponse
from sanic_ext import Extend


_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

app = Sanic("app")
# Fill in CORS headers
app.config.CORS_ORIGINS = "*"
app.config.CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
app.config.CORS_ALLOW_HEADERS = "*"
Extend(app)


@app.route("/sample", methods=["GET", "POST"])
async def sample(request):
    
    text_prompt= request.json.get("caption") 

    cfg_scale =12.0
    global device
    global latent_size

    
    # Create sampling noise:
    n = len(text_prompt)
    null_text_prompt = [""]*n
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    text_embedder = TextEmbedder(text_encoder,tokenizer,0)
    y= text_embedder(text_prompt,False)
    # y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)

    y_null = text_embedder(null_text_prompt, False)

    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, f"sample_{0}.png", nrow=4, normalize=True, value_range=(-1, 1))



def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))

    # args 설정
    # define as global
    global device
    global latent_size
    seed=0
    ckpt = "/workspace/austin/huchudle/results/018-DiT-L-2-Text/checkpoints/0000000.pt"
    image_size = 256
    model = "DiT-S/2-Text"
    num_sampling_steps =250
    vae = "ema"

    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt is None:
        assert model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert image_size in [256, 512]

    # Load model:
    latent_size = image_size // 8  
    model = DiT_models[model](
        input_size=latent_size,
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = ckpt or f"DiT-XL-2-{image_size}x{image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae}").to(device)

    
    app.run(host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()















######################################################################################################################################################

## OG code
def main(args):

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        # assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    text_prompt=["A male student who is nervous and tense before an important presentation", "A man whose head consists of a balloon."]
    

    # Create sampling noise:
    n = len(text_prompt)
    null_text_prompt = [""]*n
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    text_embedder = TextEmbedder(text_encoder,tokenizer,0)
    y= text_embedder(text_prompt,False)
    # y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)

    y_null = text_embedder(null_text_prompt, False)
    # y_null = torch.tensor([1000] * n, device=device)

    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, f"sample_{args.seed}.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-S/2-Text")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=12.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
