import dlib
from PIL import Image, ImageDraw
import json

def member():

    filepath = "/workspace/huchudle/HY/dataset/dataset_list.json"
    with open(filepath,"r") as file:
        data = json.load(file)

    print(data["landmark"])
    
    # black backgraound 
    image = Image.new("RGB", (64, 64), color="black")
    
    draw = ImageDraw.Draw(image)

    for dot in data["landmark"]:
         x, y = dot
         ratio = 64/256
         draw.point((x*ratio, y*ratio, x*ratio, y*ratio), fill="white")
      
    return image


