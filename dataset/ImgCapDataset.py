from glob import glob
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import os
from PIL import Image
import torch
import json

from torchvision.transforms import Resize, PILToTensor
from tqdm.auto import tqdm
from torchvision.datasets import CocoCaptions


class HuchuCocoCaptions(CocoCaptions):
    def __init__(self, *args, **kwargs):
        super(HuchuCocoCaptions, self).__init__(*args, **kwargs)
    
    def __getitem__(self, idx):
        img, captions = super(HuchuCocoCaptions, self).__getitem__(idx)
        return img, (' '.join(captions))


class ImgCapDataset(Dataset):
    def __init__(self, root, ann_dir, transform=None):
        super(ImgCapDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.caption_file_path = ann_dir

    def __len__(self):
        return len(self.img_files)


    def __getitem__(self, idx):
        # img = cv2.cvtColor(cv2.imread(self.files[idx],cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        img = Image.open(self.img_files[idx])
        print(self.img_files[idx])
        # get id
        filename_without_extension = self.img_files[idx][-16:]
        filename_without_extension = os.path.splitext(filename_without_extension)[0]
        filename_without_extension = filename_without_extension.lstrip("0")
        id_number = filename_without_extension


        # open caption JSON file
        with open(self.caption_file_path, "r") as json_file:
            caption_data = json.load(json_file) 
        annotations = caption_data["annotations"]


        # join into one caption
        captions_list = [annotation["caption"] for annotation in annotations if str(annotation["image_id"]) == str(id_number)]

        caption = " ".join(captions_list) if captions_list else None
      
        if self.transform:
            img = torch.from_numpy(self.transform(img).pixel_values[0])
        else:
            img = PILToTensor()(img)
            img = Resize((680, 1024), antialias=True)(img)
            if (len(img.shape) == 2):
                img = img.unsqueeze(0)
                img = img.repeat((3, 1, 1))
            if (img.shape[0] == 1):
                img = img.repeat((3, 1, 1))
            if (img.shape[0] == 4):
                img = img[:3, ...]

        
        assert(img.shape[0]==3), f"got {img.shape}"
        fname = osp.basename(self.img_files[idx])
        
        return img, caption
    

if __name__ == "__main__":
    dataset = ImgCapDataset('/workspace/train2017')
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=False)
    # print(dataloader[0])
    for i, (img, fname) in enumerate(tqdm(dataloader)):
        assert (img.shape[1] == 3)