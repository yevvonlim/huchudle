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
               
                prompt1 = ["Question: What is the eye color of this person? Answer: "]*img.shape[0]
                inputs1 = processor(img, text=prompt1, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids1 = model.module.generate(**inputs1, max_new_tokens=5)
                generated_text1 = processor.batch_decode(generated_ids1, skip_special_tokens=True)
            
            
                # prompt2 = [f"Question: What is ethnicity of this person? Answer: " for i in range(img.shape[0])]
                # inputs2 = processor(img, text=prompt2, return_tensors="pt", padding=True).to(device, torch.float16)
                # generated_ids2 = model.module.generate(**inputs2, max_new_tokens=10)
                # generated_text2 = processor.batch_decode(generated_ids2, skip_special_tokens=True)
                
                prompt3 = [f"Question: What is this person's gender? Answer in female or male. Answer: " for i in range(img.shape[0])]
                inputs3 = processor(img, text=prompt3, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids3 = model.module.generate(**inputs3, max_new_tokens=5)
                generated_text3 = processor.batch_decode(generated_ids3, skip_special_tokens=True)
                ismale = ['female' not in generated_text3[i] for i in range(img.shape[0])]


                prompt4 = [f"Question: What is the hair color of this person? Answer: " for i in range(img.shape[0])]
                inputs4 = processor(img, text=prompt4, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids4 = model.module.generate(**inputs4, max_new_tokens=5)
                generated_text4 = processor.batch_decode(generated_ids4, skip_special_tokens=True)

                prompt5 = [f"Question: What is the hair length of this person? Answer in short or long. Answer: " for i in range(img.shape[0])]
                inputs5 = processor(img, text=prompt5, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids5 = model.module.generate(**inputs5, max_new_tokens=5)
                generated_text5 = processor.batch_decode(generated_ids5, skip_special_tokens=True)

                prompt6 = ["Question: Does this person have a beard? Answer in yes or no. Answer: " for i in range(img.shape[0])]
                inputs6 = processor(img, text=prompt6, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids6 = model.module.generate(**inputs6, max_new_tokens=5)
                generated_text6 = processor.batch_decode(generated_ids6, skip_special_tokens=True)

                prompt7 = ["Question: Does this person have a mustache? Answer in yes or no. Answer: " for i in range(img.shape[0])]
                inputs7 = processor(img, text=prompt7, return_tensors="pt", padding=True).to(device, torch.float16)
                generated_ids7 = model.module.generate(**inputs7, max_new_tokens=5)
                generated_text7 = processor.batch_decode(generated_ids7, skip_special_tokens=True)

                # prompt8 = ["Question: Does this person look old? Answer in yes or no. Answer: " for i in range(img.shape[0])]
                # inputs8 = processor(img, text=prompt8, return_tensors="pt", padding=True).to(device, torch.float16)
                # generated_ids8 = model.module.generate(**inputs8, max_new_tokens=5)
                # generated_text8 = processor.batch_decode(generated_ids8, skip_special_tokens=True)

                final_prompts = []
                for i in range(img.shape[0]):
                    if ismale[i]:
                        prompt = f"A picture of a man with {generated_text1[i]} eyes, {generated_text4[i]} hair, {generated_text5[i]} hair length. "
                        # if "old" in generated_text8[i]:
                        #     prompt += "He looks old, "
                        # else:
                        #     prompt += "He looks young, "
                        if "yes" in generated_text7[i]:
                            prompt += "He has a mustache, "
                        else:
                            prompt += "He does not have a mustache, "
                        if "yes" in generated_text6[i]:
                            prompt += "and has a beard."
                        else:
                            prompt += "and does not have a beard."

                        
                    else:
                        prompt = f"A picture of a woman with {generated_text1[i]} eyes, {generated_text4[i]} hair, {generated_text5[i]} hair length."
                        # if "old" in generated_text8[i]:
                        #     prompt += "She looks old."
                        # else:
                        #     prompt += "She looks young."

                    final_prompts.append(prompt)
            
                for i in range(img.shape[0]):
                    result["annotations"].append({
                        "image_id": fname[i], # fname[i].split('.')[0],
                        "caption": final_prompts[i]
                    })
               
                if (idx%50==0):
                    print("saving...")
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