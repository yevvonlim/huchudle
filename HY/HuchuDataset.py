import dlib
from PIL import Image, ImageDraw
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2

from tqdm.auto import tqdm

def member(data):
    
     # black backgraound 
    draw_img = Image.new("RGB", (64, 64), color="black")
    
    draw = ImageDraw.Draw(draw_img)

    for dot in data["landmark"]:
        x, y = dot
        ratio = 64/256
        draw.point((x * ratio, y * ratio), fill="white")
        
    return draw_img

class HuchuDataset(nn.Dataset):

    def __init__(self, data_dir):
        self.file_path = ("/workspace/huchudle/HY/dataset/dataset_list.json")
   
    def __getitem__(self, idx):
        #img, captions, draw_img 
    
        # open JSON file
        with open(self.file_path, "r") as json_file:
            data = json.load(json_file) 
      
        #image_id 받아서 img 가져오기 
        img_id = data[idx]["image_id"]
        img = cv2.imread('/workspace/huchudle/HY/dataset/cropped_img' + img_id)

        #caption
        caption = data[idx]["caption"]

        #landmark image
        draw_img = member(data[idx])

        return img, draw_img, caption
    
if __name__ == "__main__":
    dataset = HuchuDataset('/workspace/huchudle/HY/dataset/cropped_img') # 이미지 데이터 경로 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)




