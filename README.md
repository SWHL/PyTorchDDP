### PyTorchDDP
- PyTorch: 1.9.0
- Python: 3.7.10
- GPU: 两个机器上4个V100

### 单机多卡
```bash
# 使用第三和第四张卡的并行运行命令， torch v1.10以上
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 demo_single_gpus.py
```

### 多机多卡
#### torch.distributed.launch
- 假设是在2台机器上使用4个GPU
- 准备工作：两台机器上要有相同的代码和数据
    ```bash
    # 第一台机器启一个terminal, 运行下面程序
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="10.10.10.10" --master_port=65500 demo_distributed_launch.py

    # 第二台机器启一个terminal, 运行下面程序
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="10.10.10.10" --master_port=65500 demo_distributed_launch.py
    ```

#### torch.multiprocessing
- 假设是在2台机器上使用4个GPU
- 准备工作：两台机器上要有相同的代码和数据
    ```bash
    # 第一台机器启一个terminal, 运行下面程序
    CUDA_VISIBLE_DEVICES=0,1 python demo_multiprocessing.py --nproc_per_node 2 --nnodes 2 --node_rank 0 --master_addr 10.10.10.10 --master_port 65500

    # 第二台机器启一个terminal, 运行下面程序
    CUDA_VISIBLE_DEVICES=0,1 python demo_multiprocessing.py --nproc_per_node 2 --nnodes 2 --node_rank 1 --master_addr 10.10.10.10 --master_port 65500
    ```

#### 参数介绍
|参数名称|描述|备注|
|:---:|:---:|:---:|
|`nproc_per_node`|每个节点上有几个进程|一般一个显卡一个进程|
|`nnodes`|有几个节点数|也就是使用该机器上显卡数目|
|`node_rank`|进程开始的序号||
|`master_addr`|主节点ip地址||
|`master_port`|主节点端口号||

#### 参考资料
- [PyTorch 并行训练极简 Demo](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247638753&idx=3&sn=59705f871ebd57f3f85040527bdeeb0b&chksm=ec123318db65ba0ef2940499898f0ad0d1d3bf34167d89b880fabc22215512e2be0f35fe879a&mpshare=1&scene=23&srcid=0314wmvKMWBN8TQzyMPt3BJE&sharer_sharetime=1679562302804&sharer_shareid=d3bb90e2f60a69fc71407a2191356565#rd)