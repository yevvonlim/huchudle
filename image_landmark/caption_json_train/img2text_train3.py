import torch
import argparse
from torch.utils.data import DataLoader
from dataset import ImgCapDataset
from tqdm.auto import tqdm
import json
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    parser.add_argument('--world_size', type=int, default=3)
    return parser


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def main(dataloader):
    
    rank = dist.get_rank()
    setup_for_distributed(rank==0)
    torch.cuda.set_device(rank)
    result = {"annotations": []}    
    # opts.world_size = dist.get_world_size()
   
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map=f"cuda:{rank}")

    # model.cuda(rank)
    model = DDP(model).cuda(rank)
    device = torch.device(f"cuda:{rank}")

    try:
        with torch.no_grad():
            for idx, (img, fname) in enumerate(tqdm(dataloader)):
                # with torch.autocast('cuda'):
                prompt = ["A photography of "]*img.shape[0]
                print("Prompt1: ", prompt)
                inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids = model.module.generate(**inputs, max_new_tokens=20)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                print("Generated Text1: ", generated_text)
            
            
                prompt = [f"Question: What is this person's race? Answer: " for i in range(img.shape[0])]
                print("Prompt2: ", prompt)
                inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            # with torch.autocast('cuda'):
                generated_ids = model.module.generate(**inputs, max_new_tokens=15)
                generated_text = "This person's race is " + processor.batch_decode(generated_ids, skip_special_tokens=True)
                print("Generated Text2: ", generated_text)
            
                prompt = [f"{prompt[i]}{generated_text[i]}, Question: What is this person's gender? Answer in female or male. Answer: " for i in range(img.shape[0])]
                print("Prompt3: ", prompt)
                inputs = processor(img, text=prompt, return_tensors="pt", padding=True).to(device, torch.float16)
            # with torch.autocast('cuda'):
                generated_ids = model.module.generate(**inputs, max_new_tokens=15)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
                print("Generated Text3: ", generated_text)

            
                # prompt = ["This image caption is ", "This person's race is", "This person's gender is"]*img.shape[0]
                ret = [prompt[i] + generated_text[i] + "." for i in range(img.shape[0])]
                for i in range(img.shape[0]):
                    result[fname[i]] = ret[i]                
                
                captions = []
                captions.extend(ret)

                for idx in range(img.shape[0]):
                    result["annotations"].append({
                        "image_id": fname[idx], # fname[i].split('.')[0],
                        "caption": captions[idx]
                    })

               
                if (idx%50==0):
                    with open(f'/workspace/image_landmark/caption_json/3_output_cropped_json/caption_result{rank}.json', 'w') as fp:
                        json.dump(result, fp)
                        
    except Exception as e:
        print(e)
        with open(f'/workspace/image_landmark/caption_json/3_output_cropped_json/caption_result{rank}.json', 'w') as fp:
            json.dump(result, fp)

    with open(f'/workspace/image_landmark/caption_json/3_output_cropped_json/caption_result{rank}.json', 'w') as fp:
            json.dump(result, fp)




if __name__ == "__main__":
    parser = argparse.ArgumentParser('BLIP2 distributed inference for image-to-text', parents=[get_args_parser()])
    opts = parser.parse_args()
    dist.init_process_group(backend='nccl')
                            # init_method='tcp://127.0.0.1:9011',
                            # world_size=opts.world_size,
                            # rank=rank)
    dataset = ImgCapDataset('/workspace/image_landmark/cropped_image/3_output_cropped_img')
    sampler = DistributedSampler(dataset=dataset, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False, pin_memory=True, sampler=sampler, num_workers=4)
    
    # torch.multiprocessing.spawn(main, nprocs=4, args=(opts,))
    main(dataloader)