import os
import json
import torch

import xml.etree.ElementTree as ET
import numpy as np

from torch.utils.data import Dataset
from torchvision.ops import box_convert
from PIL import Image

import conf as cfg

def make_json_file(vocpath):
    abspath = os.path.abspath(vocpath, )

    for data_split in ['train', 'trainval', 'val']:
        with open(f"{abspath}/ImageSets/Main/{data_split}.txt", "r") as f:
            file_path_list = f.read().splitlines()
        
        img_path_list = [f"{abspath}/JPEGImages/{i}.jpg" for i in file_path_list]
        xml_path_list = [f"{abspath}/Annotations/{i}.xml" for i in file_path_list]
        xml_parse_list = []

        for single_xmlpath in xml_path_list:
            tree = ET.parse(single_xmlpath)
            root = tree.getroot()
            bbox_dict = {"boxes": [], "labels": []}

            for _object in root.findall("object"):
                label = cfg.CLASSES_DICT[_object.find("name").text]
                
                xmin = int(_object.find("bndbox").find("xmin").text) - 1
                ymin = int(_object.find("bndbox").find("ymin").text) - 1
                xmax = int(_object.find("bndbox").find("xmax").text) - 1
                ymax = int(_object.find("bndbox").find("ymax").text) - 1

                bbox_dict['boxes'].append([xmin, ymin, xmax, ymax])
                bbox_dict['labels'].append(label)                
            
            xml_parse_list.append(bbox_dict)

        with open(f"./pre_defined_data/{data_split}_images.json", "w") as f:
            json.dump(img_path_list, f)
        
        with open(f"./pre_defined_data/{data_split}_annot.json", "w") as f:
            json.dump(xml_parse_list, f)
            

class VOCDataset(Dataset):
    def __init__(self, data_split, img_resize, apply_transform):
        with open(f"./pre_defined_data/{data_split}_images.json", "r") as f:
            self.img_path_list = json.load(f)
        
        with open(f"./pre_defined_data/{data_split}_annot.json", "r") as f:
            self.xml_parse_list = json.load(f)
        
        self.transform = apply_transform
        self.imgwidth, self.imgheight = img_resize

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, idx):
        gridinfo = torch.zeros((7, 7, 30), dtype=torch.float)

        img = np.array(Image.open(self.img_path_list[idx]))
        bbox = self.xml_parse_list[idx]['boxes']
        labels = self.xml_parse_list[idx]['labels']

        augmentation = self.transform(image=img, bboxes=bbox, class_labels=labels)

        img = augmentation['image']
        labels = torch.tensor(augmentation['class_labels'])
        
        boxes = torch.tensor(augmentation['bboxes'])
        boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='cxcywh')

        for box, label in zip(boxes, labels):
            box = box.float()
            grid_x = int(box[0] / 32)
            grid_y = int(box[1] / 32)

            gridinfo[grid_x, grid_y, [0, 5]] = box[0] / self.imgwidth
            gridinfo[grid_x, grid_y, [2, 7]] = box[2] / self.imgwidth

            gridinfo[grid_x, grid_y, [1, 6]] = box[1] / self.imgheight
            gridinfo[grid_x, grid_y, [3, 8]] = box[3] / self.imgheight
                     
            gridinfo[grid_x, grid_y, [4, 9]] = 1
            gridinfo[grid_x][grid_y][10 + label] = 1

        return img, gridinfo
    
if __name__ =="__main__":
    vocpath = '../dataset/voc/VOCdevkit/VOC2007'
    make_json_file(vocpath)
    
    print("prepare datset complete")
    