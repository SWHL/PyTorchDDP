# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms


def init_distributed_mode(args):
    if not torch.distributed.is_available():
        raise ValueError('This machine is not supported the DDP!')

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    print(args)
    args.distributed = True
    args.dist_url = 'env://'

    args.dist_backend = 'nccl'
    print(f'distributed init (rank {args.rank}): {args.dist_url}',
          flush=True)
    dist.init_process_group(backend=args.dist_backend,
    init_method='env://')
    dist.barrier()


def cleanup():
    dist.destroy_process_group()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    # 在整个训练过程中，模型必须启动并保持同步，这一点非常重要。
    # 否则，您将获得不正确的渐变，并且模型将无法收敛。
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    # ✧✧这里可以添加其他相关参数

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    init_distributed_mode(args)

    device = torch.device(args.device)
    torch.cuda.set_device(args.local_rank)

    args.rank = dist.get_rank()

    print(f"[init] == local rank: {args.local_rank}, global rank: {args.rank} ==")

    # rank = 0时，表明是主进程
    if args.rank == 0:
        # 实例化一些 tensorboard信息
        tensorboard_writer = SummaryWriter('tmp')

    # 学习率需要乘以总的GPU个数
    args.lr *= args.world_size

    # 单个GPU上
    batch_size = 128
    num_workers = 2

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True,
                                   transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=num_workers,
                                               prefetch_factor=3,
                                               persistent_workers=True)

    model = Net().to(device)

    # 加载预训练模型
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 当网络中具有sysncBN，使用
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    criterion = nn.CrossEntropyLoss()

    model.train()
    try:
        for epoch in range(100):
            train_loader.sampler.set_epoch(epoch)

            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                images = data.to(device)
                target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if args.rank == 0:
                    print(f'step: {batch_idx+1}/{len(train_loader)} | loss: {train_loss:6.3f}')

            if args.rank == 0:
                # 记录信息
                # 打印信息

                # 保存模型
                torch.save(model.state_dict(), 'model.pt')
    finally:
        cleanup()
