import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.cocodataset import CocoDataset


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.root = data_dir

    def setup(self, stage=None):
        self.coco_train = CocoDataset(self.root + "/train", self.root + '/train.json')
        self.coco_val = CocoDataset(self.root + "/val", self.root + "/val.json")

    def train_dataloader(self):
        return DataLoader(self.coco_train, num_workers=4, batch_size=self.batch_size, collate_fn=lambda batch: tuple(zip(*batch)))

    def val_dataloader(self):
        return DataLoader(self.coco_val, num_workers=4, batch_size=self.batch_size, collate_fn=lambda batch: tuple(zip(*batch)))

    def test_dataloader(self):
        return DataLoader(self.coco_val, num_workers=4, batch_size=self.batch_size, collate_fn=lambda batch: tuple(zip(*batch)))

