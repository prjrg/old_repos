import torch

from parallel.continue_training_parallel import run_app, main

if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()

    run_app(main, n_gpus)


