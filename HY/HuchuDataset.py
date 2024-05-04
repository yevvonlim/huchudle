import dlib
from PIL import Image, ImageDraw
import json
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import os

from tqdm.auto import tqdm


class HuchuDataset(nn.Dataset):

    def member(data):
    
         # black backgraound 
        image = Image.new("RGB", (64, 64), color="black")
    
        draw = ImageDraw.Draw(image)

        for dot in data["landmark"]:
            x, y = dot
            ratio = 64/256
            draw.point((x * ratio, y * ratio), fill="white")
        
        return image
    
    def __getitem__(self, idx):
        
        # open caption JSON file
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file) 

        # image_id 받아서 image 저장된 경로에서 가져오기 
        img = data["image_id"] 

        draw_img = self.member(data)

        caption = data["caption"]

        return img, draw_img, caption
    
if __name__ == "__main__":
    dataset = HuchuDataset('/workspace/train2017') # 데이터 경로
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)
    # print(dataloader[0])
    for i, (img, fname) in enumerate(tqdm(dataloader)):
        assert (img.shape[1] == 3)

