from PIL import Image, ImageDraw
import json
from torch.utils.data import Dataset
import os
from torchvision.transforms import ToTensor
import torch
import cv2


def draw_landmarks(data):
    
    # black backgraound 
    draw_img = Image.new("RGB", (32, 32), color="black")
    
    draw = ImageDraw.Draw(draw_img)

    for dot in data["landmark"]:
        x, y = dot
        ratio = 32/256
        draw.point((x * ratio, y * ratio), fill="white")
        
    return draw_img

class HuchuDataset(Dataset):

    def __init__(self, ann_path, root_dir, istrain=True):
        self.istrain = istrain
        self.file_path = ann_path
        self.root_dir = root_dir
        
        with open(self.file_path, "r") as json_file:
            self.data = json.load(json_file) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):    
        # img
        img_id = self.data[idx]["image_id"]
        # img = Image.open(os.path.join(self.root_dir, img_id))
        img = cv2.imread(os.path.join(self.root_dir, img_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # caption
        caption = self.data[idx]["caption"]

        # landmark image
        draw_img = draw_landmarks(self.data[idx])

        img = ToTensor()(img)
        draw_img = ToTensor()(draw_img)[0:1]
        
        if torch.rand(1) > 0.5 and self.istrain:
            img = torch.flip(img, [2])
            draw_img = torch.flip(draw_img, [2])

        img = img * 2 - 1
        draw_img = draw_img * 2 - 1

        return img, draw_img, caption