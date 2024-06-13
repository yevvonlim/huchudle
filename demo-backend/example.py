# Two server routes that OctoAI containers should have:
# a route for inference requests (e.g. ”/predict”). This route for inference requests must receive JSON inputs and JSON outputs.
# a route for health checks (e.g. ”/healthcheck”).
# Number of workers (not required). Typical best practice is to make this number some function of the # of CPU cores that the server has access to and should use.

"""HTTP Inference serving interface using sanic."""
import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import DiT_models
import argparse
from models import TextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer


from custommodel import CustomModel
from sanic import Request, Sanic, response
from sanic.response import text

_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

# Load and initialize the model on startup globally, so it can be reused.
model_instance = CustomModel()
"""Global instance of the model to serve."""

server = Sanic("server")
"""Global instance of the web server."""


@server.route("/sampling", methods=["GET"])
def sampling(_: Request) -> response.JSONResponse:
    """Responds to healthcheck requests.

    :param request: the incoming healthcheck request.
    :return: json responding to the healthcheck.
    
    text_prompt example : ["A male student who is nervous and tense before an important presentation", "A man whose head consists of a balloon."]

    """
    ## SETTING
    # Recieve request for 68landmark points and caption 
    text_prompt = request.json.get("text_prompt") # parsed body object
    landmark_points = request.json.get("landmark_points") # parsed body object

    # Load our model
    latent_size = 256 // 8 # image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
    ).to(device)

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict) #  initializes the model's parameters and architecture based on the saved checkpoint
    model.eval()  #  sets the model to evaluation mode
    # setting for diffusion and VAE
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    ## HANDLING CAPTION & LANDMARK 
    # Create sampling noise for text caption data
    n = len(text_prompt)
    null_text_prompt = [""]*n
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
    text_embedder = TextEmbedder(text_encoder,tokenizer,0)
    y= text_embedder(text_prompt,False)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)

    y_null = text_embedder(null_text_prompt, False)
    # y_null = torch.tensor([1000] * n, device=device)

    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Create sampling noise for 68 landmark points
    ## ...
    ## ...

    

    ## SAMPLING
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, f"sample_{args.seed}.png", nrow=4, normalize=True, value_range=(-1, 1))


    return response.json({"healthy": "yes"})


def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    server.run(host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    # set up PyTorch
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        # assert args.num_classes == 1000
    main()





# @server.route("/reconstruction/predict", methods=["POST"])
# def predict(request: Request) -> response.JSONResponse:
#     """Responds to inference/prediction requests.

#     :param request: the incoming request containing inputs for the model.
#     :return: json containing the inference results.
#     """

#     try:
#         inputs = request.json
#         output = model_instance.predict(inputs)
#         return response.json(output)
#     except Exception as e:
#         return response.json({'error': str(e)}, status=500)

# @server.route("/predict", methods=["POST"])
# def predict_wo_dire(request: Request) -> response.JSONResponse:
#     """Responds to inference/prediction requests.

#     :param request: the incoming request containing inputs for the model.
#     :return: json containing the inference results.
#     """

#     try:
#         inputs = request.json
#         output = model_instance.predict_wo_dire(inputs)
#         return response.json(output)
#     except Exception as e:
#         return response.json({'error': str(e)}, status=500)