import dlib
from PIL import Image, ImageDraw
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import os

from tqdm.auto import tqdm
from torchvision.transforms import ToTensor

def draw_landmarks(data):
    
    # black backgraound 
    draw_img = Image.new("RGB", (64, 64), color="black")
    
    draw = ImageDraw.Draw(draw_img)

    for dot in data["landmark"]:
        x, y = dot
        ratio = 64/256
        draw.point((x * ratio, y * ratio), fill="white")
        
    return draw_img

class HuchuDataset(Dataset):

    def __init__(self, ann_path, root_dir):
        self.file_path = ann_path
        self.root_dir = root_dir

    def __getitem__(self, idx):    
        # open JSON file
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file) 
      
        # img
        img_id = data[idx]["image_id"]
        img = Image.open(os.path.join(self.root_dir, img_id))
        img = ToTensor()(img) 
        
        # caption
        caption = data[idx]["caption"]

        # landmark image
        draw_img = draw_landmarks(data[idx])
        draw_img = ToTensor()(draw_img)

        return img, draw_img, caption
    
if __name__ == "__main__":
    dataset = HuchuDataset('/workspace/huchudle/HY/dataset/dataset_list.json', '/workspace/huchudle/HY/dataset/cropped_img/') 
    




