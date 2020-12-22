import os

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
import numpy as np


class CocoDataset(Dataset):
    def __init__(self, root, annotation, train=True):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        if train:
            self.aug = A.Compose([
                A.Resize(800, 800),
                A.GaussianBlur(),
                A.RandomFog(),
                A.HueSaturationValue(),
                A.RandomBrightnessContrast(),
                A.GaussNoise(),
                A.ToFloat(max_value=255.0)
            ])
        else:
            self.aug = A.Compose([
                        A.Resize(800, 800),
                        A.ToFloat(max_value=255.0)
                    ])

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')

        bboxes, labels, areas, isCrowd = gettarget(coco_annotation)

        image1 = np.array(image, dtype=np.float32)

        image = self.aug(image=image1)['image']

        target = fix_target2(bboxes, index, areas, isCrowd, image1.shape[1], image1.shape[0])

        image = torch.from_numpy(image).permute(2, 0, 1)

        return image, target

    def __len__(self):
        return len(self.ids)


def gettarget(coco_annotation):
    num_objs = len(coco_annotation)

    bboxes = []
    labels = []
    areas = []
    isCrowd = []
    for i in range(num_objs):
        xmin = coco_annotation[i]['bbox'][0]
        ymin = coco_annotation[i]['bbox'][1]
        xmax = xmin + coco_annotation[i]['bbox'][2]
        ymax = ymin + coco_annotation[i]['bbox'][3]
        bboxes.append([xmin, ymin, xmax, ymax])
        areas.append(coco_annotation[i]['area'])
        labels.append(coco_annotation[i]['category_id'])
        isCrowd.append(coco_annotation[i]['iscrowd'])

    for i in range(num_objs):
        bboxes[i].append(labels[i])

    return bboxes, labels, areas, isCrowd


def fix_target(transformed, item, areas, isCrowd):
    target = {}
    if len(transformed['bboxes']) > 0:
        transformed['bboxes'] = np.array(transformed['bboxes'])
        labels = transformed['bboxes'][:, 4].tolist()
        transformed['bboxes'] = transformed['bboxes'][:, 0:4].tolist()
        target['boxes'] = []
        target['labels'] = []
        for label, bbox in zip(labels, transformed['bboxes']):
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                target['boxes'].append(bbox)
                target['labels'].append(label)

        if len(target['boxes']) > 0:
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.ones((1, 1), dtype=torch.int64)
    else:
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.ones((1, 1), dtype=torch.int64)

    target['image_id'] = torch.tensor([item], dtype=torch.int64)
    target['area'] = torch.as_tensor(areas, dtype=torch.float32)
    target['iscrowd'] = torch.tensor(isCrowd, dtype=torch.int64)

    return target


def fix_target2(bboxes, item, areas, isCrowd, width, height):
    target = {}
    if len(bboxes) > 0:
        bboxes = np.array(bboxes)
        labels = bboxes[:, 4].tolist()
        bboxes[:, 0] = bboxes[:, 0] * 800.0 / width
        bboxes[:, 2] = bboxes[:, 2] * 800.0 / width
        bboxes[:, 1] = bboxes[:, 1] * 800.0 / height
        bboxes[:, 3] = bboxes[:, 3] * 800.0 / height
        bboxes = bboxes[:, 0:4].tolist()
        target['boxes'] = []
        target['labels'] = []
        for label, bbox in zip(labels, bboxes):
            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                target['boxes'].append(bbox)
                target['labels'].append(label)

        if len(target['boxes']) > 0:
            target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.ones((1, 1), dtype=torch.int64)
    else:
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.ones((1, 1), dtype=torch.int64)

    target['image_id'] = torch.tensor([item], dtype=torch.int64)
    target['area'] = torch.as_tensor(areas, dtype=torch.float32)
    target['iscrowd'] = torch.tensor(isCrowd, dtype=torch.int64)

    return target