# !/bin/bash

torchrun --nnodes=1 --nproc_per_node=3 train.py --data-path /workspace/austin/coco2017 --image-size 256 --global-batch-size 363 --ckpt-every 5_000