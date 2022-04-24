# demo.py
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms

# 在整个训练过程中，模型必须启动并保持同步，这一点非常重要。
# 否则，您将获得不正确的渐变，并且模型将无法收敛。
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


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


def main(local_rank, args):
    rank = args.nnodes * args.node_rank + local_rank

    device = torch.device('cuda')
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=args.backend,
                            init_method=f'tcp://{args.master_addr}:{args.master_port}',
                            rank=rank,
                            world_size=args.world_size)

    if rank == 0:
        tensorboard_writer = SummaryWriter('tmp')

    # 单个GPU上
    batch_size = 64
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
    model = DistributedDataParallel(model, device_ids=[local_rank])

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

                if rank == 0:
                    print(f'step: {batch_idx+1}/{len(train_loader)} | loss: {train_loss:6.3f}')

            if rank == 0:
                # 记录信息
                # 打印信息

                # 保存模型
                torch.save(model.state_dict(), 'model.pt')
    finally:
        cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc_per_node', type=int, default=1,
                        help='number of nodes (default: 1)')

    parser.add_argument('--nnodes', type=int, default=1,
                       help='number of gpus per node (default: 1)')

    parser.add_argument('--node_rank', type=int, default=0,
                        help='ranking within the nodes')

    parser.add_argument('--backend', type=str, help='distributed backend',
                        choices=[dist.Backend.GLOO, dist.Backend.NCCL],
                        default=dist.Backend.NCCL)

    parser.add_argument('--master_addr', type=str, default="localhost",
                        help='master address')

    parser.add_argument('--master_port', type=str, default="5000",
                        help='master port')

    args = parser.parse_args()

    args.world_size = args.nnodes * args.nproc_per_node
    mp.spawn(main, nprocs=args.nnodes, args=(args, ))
