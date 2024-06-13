#import for model
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models_dit import DiT_models
import argparse
from models_dit import TextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw


#import for Sanic
from sanic import Sanic
from sanic.response import text
from sanic import Request, Sanic, response
import os
from sanic import json
from sanic import HTTPResponse
from dataset import HuchuDataset
from sanic import file
from sanic_ext import Extend


# app setting
_DEFAULT_PORT = 8000
"""Default port to serve inference on."""

app = Sanic("app")
# Fill in CORS headers
app.config.CORS_ORIGINS = "*"
app.config.CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
app.config.CORS_ALLOW_HEADERS = "*"
Extend(app)



@app.route("/test", methods=["GET", "POST"])
async def test(request):
    try:
        caption = request.json.get("caption") 
        landmark = request.json.get("landmark") 

        # generate new url
        url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTq5H7JuSiIVSF7n7lbXm0-zNWmJoX71Bosww_oma7qrw&s"


        result= {
            "isSuccess": True,
            "img_url": url
        }
    except:
        result = {
            "isSuccess": False,
            "img_url": "error"
        }
    return json(result)




@app.route("/postwelltest", methods=["GET", "POST"])
async def post_well_test(request):
    try:
        caption = request.json.get("caption") 
        landmark = request.json.get("landmark") 

        result= {
            "isSuccess": True,
            "caption":caption,
            "lanmark": landmark
        }
    except:
        result = {
            "isSuccess": False,
            "lanmark": "error"
        }
    return json(result)


@app.route("/imagetest", methods=["GET", "POST"])
async def imagetest(request):
    try:
        caption = request.json.get("caption") 
        landmark = request.json.get("landmark") 

        image = await file("/workspace/project-root/demo-backend/sample_img.jpg", filename="super-awesome-incredible.jpg")

        result= {
            "isSuccess": True,
            "caption":caption,
            "lanmark": landmark,
            "image" : image
        }
    except:
        result = {
            "isSuccess": False,
            "lanmark": "error"
        }
    return await file("/workspace/project-root/demo-backend/sample_img.jpg", filename="sample.jpg")




def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))
    app.run(host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()