#! /bin/bash
# 假设是在2台机器上使用4个GPU
# 准备工作：两台机器上要有相同的代码和数据


# 第一台机器启一个terminal, 运行下面程序
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.10.10.10" --master_port=65500 demo_distributed_launch.py

# 第二台机器启一个terminal, 运行下面程序
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="10.10.10.10" --master_port=65500 demo_distributed_launch.py