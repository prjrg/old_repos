import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from model.trained_densenet201 import TrainedDensenet201


def coco_densenet201():
    m = TrainedDensenet201()
    m.out_channels = 256

    anchor_sizes = ((32,), (64,), (128,), (256,), (384,), (512,),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['a', 'b', 'c', 'd'], output_size=[7, 7], sampling_ratio=2)

    model = FasterRCNN(m, num_classes=91, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, min_size=800)

    torch.nn.init.kaiming_normal_(model.rpn.head.conv.weight)
    torch.nn.init.kaiming_normal_(model.rpn.head.cls_logits.weight)
    torch.nn.init.kaiming_normal_(model.rpn.head.bbox_pred.weight)
    torch.nn.init.kaiming_normal_(model.roi_heads.box_head.fc6.weight)
    torch.nn.init.kaiming_normal_(model.roi_heads.box_head.fc7.weight)
    torch.nn.init.kaiming_normal_(model.roi_heads.box_predictor.cls_score.weight)
    torch.nn.init.kaiming_normal_(model.roi_heads.box_predictor.bbox_pred.weight)


    for param in model.roi_heads.parameters():
        param.requires_grad = True
    for param in model.rpn.parameters():
        param.requires_grad = True
    for param in model.parameters():
        param.requires_grad = True

    return model


def save_model(epoch, model, optimizer, lr_scheduler=None):
    checkpoint_path = f"./models/coco_checkpoint_params{epoch}.pth"
    if lr_scheduler is not None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
        }, checkpoint_path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)




