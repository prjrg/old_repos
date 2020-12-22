import os
import tempfile
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from data.cocodataset import CocoDataset
from model.coco_densenet201 import coco_densenet201, save_model
from tvision import utils
from tvision.engine import train_one_epoch, evaluate


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size):
    setup(rank, world_size)
    train_dataset = CocoDataset('./models/coco/train', './models/coco/train.json')
    test_dataset = CocoDataset('./models/coco/val', './models/coco/val.json', False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, num_workers=0, pin_memory=True, sampler=train_sampler, collate_fn=utils.collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=8, num_workers=0, pin_memory=True, collate_fn=utils.collate_fn)

    model = coco_densenet201().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.00005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    num_epochs = 150

    for epoch in range(num_epochs):
        CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
        if rank == 0:
            # All processes should see same parameters as they all start from same
            # random parameters and gradients are synchronized in backward passes.
            # Therefore, saving it in one process is sufficient.
            torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        ddp_model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=map_location))
        train_one_epoch(ddp_model, optimizer, train_loader, rank, epoch, 200)
        lr_scheduler.step()

        evaluate(ddp_model, test_loader, device=rank)
        if rank == 0:
            save_model(epoch, ddp_model, optimizer, lr_scheduler)
        if rank == 0:
            os.remove(CHECKPOINT_PATH)

    cleanup()

def run_app(fn, world_size):
    mp.spawn(fn, args=(world_size,),
             nprocs=world_size,
             join=True)