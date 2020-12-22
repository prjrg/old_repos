from pytorch_lightning import Trainer, seed_everything

from data.coco_data_module import CocoDataModule
from model.faster_rcnn import FasterRCNN

if __name__ == '__main__':
    seed_everything(42)
    data = CocoDataModule('./dataset', 8)
    task = FasterRCNN()

    trainer = Trainer(distributed_backend='ddp', deterministic=True, gpus=2, max_epochs=100, row_log_interval=500)
    trainer.fit(task, datamodule=data)
    trainer.save_checkpoint("training_loop.ckpt")


