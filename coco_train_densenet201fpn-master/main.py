import torch

from data.cocodataset import CocoDataset
from parallel.parallel import run_app, main

if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()

    run_app(main, n_gpus)

    # coco = CocoDataset('./models/coco/train', './models/coco/train.json')
    #
    # for i in range(3):
    #     print(coco[i][1]['boxes'])


