import os
from PIL import Image 
from torchvision.transforms import ToTensor
import json
from torch.utils.data import Dataset
import torchvision.transforms.functional as fn
import cv2


class OurDataset(Dataset):
    
    def __init__(self, ann_path, root_dir) -> None:
        self.ann_path=ann_path
        self.root_dir=root_dir

        with open(self.ann_path,"r") as json_file:
            self.data=json.load(json_file)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        #img
        image_id = self.data[idx]["image_id"]
        img = cv2.imread(os.path.join(self.root_dir, image_id))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ToTensor()(img)

        #caption 
        cap = self.data[idx]["caption"]
        
        
        return img, cap




