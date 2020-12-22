import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import pytorch_lightning as pl
from torchvision.ops import box_iou

from optimizer.lookahead import Lookahead
from optimizer.radam import RAdam

backbone = resnet_fpn_backbone('resnet152', True)
backbone.out_channels = 256

anchor_sizes = ((16,), (32,), (64,), (128,), (256,), (512,),)
aspect_ratios = (0.5, 1.0, 2.0) * len(anchor_sizes)
anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=[7, 7], sampling_ratio=2)

resnet_model = FasterRCNN(backbone, num_classes=91, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=600)

torch.nn.init.kaiming_normal_(resnet_model.rpn.head.conv.weight)
torch.nn.init.kaiming_normal_(resnet_model.rpn.head.cls_logits.weight)
torch.nn.init.kaiming_normal_(resnet_model.rpn.head.bbox_pred.weight)
torch.nn.init.kaiming_normal_(resnet_model.roi_heads.box_head.fc6.weight)
torch.nn.init.kaiming_normal_(resnet_model.roi_heads.box_head.fc7.weight)
torch.nn.init.kaiming_normal_(resnet_model.roi_heads.box_predictor.cls_score.weight)
torch.nn.init.kaiming_normal_(resnet_model.roi_heads.box_predictor.bbox_pred.weight)

def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


class FasterRCNN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001
    ):
        """
        PyTorch Lightning implementation of `Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks <https://arxiv.org/abs/1506.01497>`_.
        Paper authors: Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
        Model implemented by:
            - `Teddy Koker <https://github.com/teddykoker>`
        During training, the model expects both the input tensors, as well as targets (list of dictionary), containing:
            - boxes (`FloatTensor[N, 4]`): the ground truth boxes in `[x1, y1, x2, y2]` format.
            - labels (`Int64Tensor[N]`): the class label for each ground truh box
        CLI command::
            # PascalVOC
            python faster_rcnn.py --gpus 1 --pretrained True
        Args:
            learning_rate: the learning rate
            num_classes: number of detection classes (including background)
            pretrained: if true, returns a model pre-trained on COCO train2017
            pretrained_backbone: if true, returns a model with backbone pre-trained on Imagenet
            trainable_backbone_layers: number of trainable resnet layers starting from final block
        """
        super().__init__()

        model = resnet_model

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True

        self.learning_rate = learning_rate

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = torch.stack([loss for loss in loss_dict.values()]).sum()
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        base_optim = RAdam(self.model.parameters(), lr=self.learning_rate)
        optimizer = Lookahead(base_optim, k=5, alpha=0.5)

        return optimizer



