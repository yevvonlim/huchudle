# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess
import sys

#import for model
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
# from models_dit import DiT_models
from huchu_models.models import DiT_models
import argparse
from models_dit import TextEmbedder
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageDraw
import numpy as np
import torchvision.transforms as transforms
from putlandmark import get_initial_landmark
import aiohttp
from PIL import Image
from io import BytesIO
from dataset.HuchuDataset import HuchuDataset
from torchvision.transforms import ToTensor
from masactrl.dit_utils import FACEPipeline
import cv2





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
from sanic.worker.manager import WorkerManager

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)


app = Sanic("app")
# Fill in CORS headers
app.config.CORS_ORIGINS = "*"
app.config.CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
app.config.CORS_ALLOW_HEADERS = "*"
# app.config.CORS_EXPOSE_HEADERS="*,*"
WorkerManager.THRESHOLD = 1800

Sanic.start_method = "spawn"
Sanic.START_METHOD_SET = True
app.config.INSPECTOR = True
# torch.multiprocessing.set_start_method('spawn', force=True)  
Extend(app)

# apis below here are test api
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


# data에 landmark 배열 넣어주기
def draw_landmarks(data):
    # print("landmark data:", data)
    # black backgraound 
    draw_img = Image.new("RGB", (32, 32), color="black")
    
    draw = ImageDraw.Draw(draw_img)

    for dot in data:
        x, y = dot
        ratio = 32/256
        draw.point((x * ratio, y * ratio), fill="white")

    print(draw_img)    
    print(draw_img.size)
    return draw_img



# our apis

# setting args
_DEFAULT_PORT =8000
device = "cuda" if torch.cuda.is_available() else "cpu" 
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
image_size = 256
latent_size = image_size//8
num_sampling_steps = 50
diffusion = create_diffusion(str(num_sampling_steps))
seed=0
ckpt = "/workspace/project-root/demo-backend/018-DiT-L-2-Text/checkpoints/0333300.pt" # example
ckpt_path = ckpt 
vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
cfg_scale =5.0

inversion = "npi"  #default는 cfg로 주기
model = DiT_models["DiT-L/2-Text"](
        input_size=latent_size,
        exceptional_prompt=False,

    ).to(device)


state_dict = find_model(ckpt_path)
model.load_state_dict(state_dict, strict=False)
model.eval()  # important!


# fetch image from url
async def fetch_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                img_data = await resp.read()
                image = Image.open(BytesIO(img_data))
                return image
            else:
                raise Exception(f"Failed to fetch image, status code: {resp.status}")


# point landmark from image
@app.route("/pointlandmark",methods=["GET", "POST"])
async def pointlandmark(request):
    try:
        image_base64 = request.json.get("image_base64")
        print(image_base64)
 
        # get landmark
        encoded_image, landmark_list,resized_image = get_initial_landmark(image_base64, True)
        print(resized_image,landmark_list)

        # encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_BGR2RGB)
        

        result = {
            "isSuccess": True,
            "landmark": landmark_list,
            "resized_image": encoded_image
        }
    except Exception as e:
        result = {
            "isSuccess": False,
            "landmark": f"error_{e}"
        }
    return json(result)



# sample api for generating image
@app.route("/sample", methods=["GET", "POST"])
async def sample(request):
    global device
    global latent_size
    global model
    print(device, latent_size)
    
    text_prompt= request.json.get("caption") 
    landmarks_list = request.json.get("landmark")

    
    landmark_img = draw_landmarks(landmarks_list)
    print(isinstance(landmark_img, torch.Tensor))
    transform = transforms.ToTensor()
    landmark_img = transform(landmark_img)[0:1]
    print("landmark_img.shape:",landmark_img.shape)
    print("landmark_img:",landmark_img)
    print(isinstance(landmark_img, torch.Tensor))      
    if torch.rand(1) > 0.5:
        landmark_img = torch.flip(landmark_img, [2])

    landmark_img = landmark_img * 2 - 1
    landmark_img = landmark_img.unsqueeze(0).to(device)
    
    # Create sampling noise:
    text_prompt = [text_prompt]
    n = len(text_prompt)
    null_text_prompt = [""]*n
    z = torch.randn(2*n, 4, latent_size, latent_size, device=device)
    y = text_prompt + null_text_prompt
    landmarks = torch.cat([landmark_img, landmark_img], 0)

    model_kwargs = dict(y=y, cfg_scale=cfg_scale, landmark=landmarks)
    print(len(y), z.shape, landmarks.shape)

    # change code to use pipeline -later
    # inversion = "cfg"
    # pipeline = FACEPipeline(model, diffusion, vae, inversion=inversion, real_step=num_sampling_steps, device=device)
    # samples = pipeline.sample(text_prompt, landmark_img, z, progress=True)
    # save_image(samples, f"/workspace/project-root/demo-backend/temp/sample_{seed}.jpg", nrow=4, normalize=True, value_range=(-1, 1))
    # return await file(f"/workspace/project-root/demo-backend/temp/sample_{seed}.jpg", filename="sample.jpg")
    # ###

    # Sample images:
    with torch.no_grad():
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        samples = vae.decode(samples / 0.18215).sample
        samples, _ = samples.chunk(2)

    # Save and display images:
    save_image(samples, f"/workspace/project-root/demo-backend/temp/sample_{seed}.jpg", nrow=4, normalize=True, value_range=(-1, 1))
    save_image(landmarks, f"/workspace/project-root/demo-backend/temp/landmark_{seed}.jpg", nrow=4, normalize=True, value_range=(-1, 1))

    return await file(f"/workspace/project-root/demo-backend/temp/sample_{seed}.jpg", filename="sample.jpg")
    # return json({"isSuccess": True, "img_url": f"/workspace/project-root/demo-backend/temp/sample_{seed}.jpg"})

@app.route("/editlandmark", methods=["GET", "POST"])
async def editlandmark(request):
    global inversion
    global device
    global model
    global latent_size

    # target landmark
    TARGET_LD = request.json.get("landmark")
    TARGET_LANDMARK = draw_landmarks(TARGET_LD)
    TARGET_LANDMARK = ToTensor()(TARGET_LANDMARK)[0:1]
    print("TARGET_LANDMARK",TARGET_LANDMARK)

    # get request
    image_base64 = request.json.get("image_base64")
    text_prompt= request.json.get("caption") 

    # get landmark list from image
    encoded_image, og_landmark_list,resized_image = get_initial_landmark(image_base64)
    
    # draw landmark on image
    og_landmark_img = draw_landmarks(og_landmark_list)
    print(isinstance(og_landmark_img, torch.Tensor))
    transform = transforms.ToTensor()
    og_landmark_img = transform(og_landmark_img)[0:1]
    print("landmark_img.shape:",og_landmark_img.shape)
    print("landmark_img:",og_landmark_img)
    print(isinstance(og_landmark_img, torch.Tensor))      
    if torch.rand(1) > 0.5:
        og_landmark_img = torch.flip(og_landmark_img, [2])
    og_landmark_img = og_landmark_img * 2 - 1
    og_landmark_img = og_landmark_img.unsqueeze(0).to(device)
    
    inversion = "npi"
    pipeline = FACEPipeline(model, diffusion, vae, inversion=inversion, real_step=num_sampling_steps, device=device)
    new_landmark_img = TARGET_LANDMARK.to(device).unsqueeze(0)
    
    # inverting code
    resized_image = ToTensor()(resized_image).unsqueeze(0)
    
    resized_image = resized_image * 2 - 1
    x = resized_image.to(device)
    landmark_img = og_landmark_img.to(device)
    y = text_prompt
    print("resized_image shape",resized_image.shape)
    print("landmark_img shape",landmark_img.shape)
    print("text_prompt",text_prompt)

    z, intermediates = pipeline.invert(x, landmark_img, y, return_intermediate=True)
    z = torch.cat([z]*2, 0).to(device)
    samples = pipeline.edit(y[0], y[0], new_landmark_img, landmark_img, z, cfg_scale=cfg_scale, intermediates=intermediates[::-1])
    save_image(samples[1], f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_sample.jpg", nrow=4, normalize=True, value_range=(-1, 1))

    return await file(f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_sample.jpg", filename="edited_img.jpg")



@app.route("/editprompt", methods=["GET", "POST"])
async def editprompt(request):
    global inversion
    global device
    global model
    global latent_size

    # get request
    image_base64 = request.json.get("image_base64")
    og_text_prompt= request.json.get("og_caption") 
    new_text_prompt= request.json.get("new_caption") 

    # get landmark list from image
    encoded_image, og_landmark_list,resized_image = get_initial_landmark(image_base64)
    
    # draw landmark on image
    og_landmark_img = draw_landmarks(og_landmark_list)
    print(isinstance(og_landmark_img, torch.Tensor))
    transform = transforms.ToTensor()
    og_landmark_img = transform(og_landmark_img)[0:1]
    print("landmark_img.shape:",og_landmark_img.shape)
    print("landmark_img:",og_landmark_img)
    print(isinstance(og_landmark_img, torch.Tensor))      
    if torch.rand(1) > 0.5:
        og_landmark_img = torch.flip(og_landmark_img, [2])
    og_landmark_img = og_landmark_img * 2 - 1
    og_landmark_img = og_landmark_img.unsqueeze(0).to(device)
    
    inversion = "npi"
    pipeline = FACEPipeline(model, diffusion, vae, inversion=inversion, real_step=num_sampling_steps, device=device)
    
    # edit prompt code
    resized_image = ToTensor()(resized_image).unsqueeze(0)
    resized_image = resized_image * 2 - 1
    x = resized_image.to(device)
    landmark_img = og_landmark_img.to(device)
    
    y = og_text_prompt
    print("og text prompt", og_text_prompt)
    og_prompt = og_text_prompt
    # new_prompt = og_prompt.replace("black", "blue")   
    z, intermediates = pipeline.invert(x, landmark_img, y, return_intermediate=True)
    # z = torch.cat([z]*2, 0).to(device)
    # z = pipeline.invert(x, landmark_img, y)
    # print(z)
    print("z shape", z.shape)
    print("og prompt",og_text_prompt)
    print("new prompt",new_text_prompt)
    samples = pipeline.edit_prompt2prompt(og_prompt= og_text_prompt,new_prompt = new_text_prompt,orig_landmark_img = landmark_img,  latent =z, cfg_scale=cfg_scale)
    save_image(samples, f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_p2p.jpg", nrow=4, normalize=True, value_range=(-1, 1))
    
    return await file(f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_p2p.jpg", filename="edited_img.jpg")

    resized_image = ToTensor()(resized_image).unsqueeze(0)
    
    resized_image = resized_image * 2 - 1
    x = resized_image.to(device)
    landmark_img = og_landmark_img.to(device)
    y = text_prompt
    print("resized_image shape",resized_image.shape)
    print("landmark_img shape",landmark_img.shape)
    print("text_prompt",text_prompt)

    z, intermediates = pipeline.invert(x, landmark_img, y, return_intermediate=True)
    z = torch.cat([z]*2, 0).to(device)
    samples = pipeline.edit(y[0], y[0], new_landmark_img, landmark_img, z, cfg_scale=cfg_scale, intermediates=intermediates[::-1])
    save_image(samples, f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_sample.jpg", nrow=4, normalize=True, value_range=(-1, 1))

    return await file(f"/workspace/project-root/demo-backend/temp/{inversion}_inversion_{seed}_ppl_sample.jpg", filename="edited_img.jpg")






def main():
    """Entry point for the server."""
    port = int(os.environ.get("SERVING_PORT", _DEFAULT_PORT))

    # args 설정
    global device
    global latent_size
    global vae
    global diffusion
    global seed
    global model
    

    # Setup PyTorch:
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

    
    # Load model:
    latent_size = image_size // 8  
    

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    
    

    print("main-done")
    app.run(host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()