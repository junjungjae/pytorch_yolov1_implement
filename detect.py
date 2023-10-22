import argparse
import torch

import albumentations as A
import albumentations.pytorch.transforms as A_transform
import numpy as np

from PIL import Image, ImageDraw
from torchvision.ops import box_convert, clip_boxes_to_image

import conf as cfg

from yolov1_model import YOLOv1


def detect(img, transform_module, model):
    aug_img = np.array(img)
    aug_img = transform_module(image=aug_img)['image'].to(cfg.DEVICE)
    
    pred = model(aug_img.unsqueeze(0))
    res = model.inference(pred[0], 224, 224)
    pred = clip_boxes_to_image(res, size=(224, 224))
    
    bbox_pred = box_convert(pred[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
    label_pred = [cfg.IDX2CLASSES[label] for label in pred[:, 5].tolist()]
    
    draw = ImageDraw.Draw(img)
    
    for bbox, label in zip(bbox_pred, label_pred):
        bbox = bbox.tolist()
        print(bbox)
        draw.rectangle(bbox, outline=(0, 255, 0))
        draw.text(bbox[:2], label, (255, 255, 0))

    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", dest="img_path")
    parser.add_argument("--weights", dest="weights_path")
    args = parser.parse_args()
    
    
    img = Image.open(args.img_path)
    img = img.resize((224, 224))

    annot_image = ImageDraw.Draw(img)
    
    model = YOLOv1().to(cfg.DEVICE)
    model.load_state_dict(torch.load(args.weights_path))
    model.eval()
    A_inference = A.Compose([A.Resize(224, 224),
                            A.Normalize(),
                            A_transform.ToTensorV2()])
    
    detect(img, A_inference, model).resize((640, 480)).show()