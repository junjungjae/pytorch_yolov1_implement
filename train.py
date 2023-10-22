import torch

import albumentations as A
import albumentations.pytorch.transforms as A_transform

from torch.utils.data import DataLoader
from tqdm import tqdm

import utils 

import conf as cfg

from custom_dataset import VOCDataset
from yolov1_model import YOLOv1
from yolov1_loss import YOLOv1Loss

import warnings
warnings.filterwarnings("ignore")

img_width, img_height = (224, 224)


A_train = A.Compose([A.Resize(img_width, img_height),
                     A.HorizontalFlip(),
                     A.Normalize(),
                     A_transform.ToTensorV2()
                     ],
                     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

A_valid = A.Compose([A.Resize(img_width, img_height),
                      A.Normalize(),
                      A_transform.ToTensorV2()
                      ],
                     bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

train_dataset = VOCDataset(data_split='train', img_resize=(img_width, img_height), apply_transform=A_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)

valid_dataset = VOCDataset(data_split='val', img_resize=(img_width, img_height), apply_transform=A_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, pin_memory=True)

model = YOLOv1().to(cfg.DEVICE)
criterion = YOLOv1Loss().to(cfg.DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9, weight_decay=0.0005)
earlystopper = utils.EarlyStopping(patience=50, verbose=True, delta=1e-6)

epochs = 200

for epoch in range(1, epochs+1):
    pbar = tqdm(train_dataloader, total=len(train_dataloader), ncols=200)
    train_epoch_loss, train_epoch_cls_loss, train_epoch_obj_loss, train_epoch_box_loss = 0, 0, 0, 0
    
    model.train()
    for train_minibatch, (img, gridinfo) in enumerate(pbar):
        img = img.to(cfg.DEVICE)
        gridinfo = gridinfo.to(cfg.DEVICE)
        
        res = model(img)

        calc_loss, cls_loss, box_loss, reg_loss = criterion(res, gridinfo)
        
        optimizer.zero_grad()
        calc_loss.backward()
        optimizer.step()

        train_epoch_loss = (train_epoch_loss * train_minibatch + calc_loss.item()) / (train_minibatch + 1)
        train_epoch_cls_loss = (train_epoch_cls_loss * train_minibatch + cls_loss.item()) / (train_minibatch + 1)
        train_epoch_obj_loss = (train_epoch_obj_loss * train_minibatch + box_loss.item()) / (train_minibatch + 1)
        train_epoch_box_loss = (train_epoch_box_loss * train_minibatch+ reg_loss.item()) / (train_minibatch + 1)

        loss_status = f"Epoch: {epoch}/{epochs}\tcls loss: {train_epoch_cls_loss:.6f}\tobj loss: {train_epoch_obj_loss:.6}\tbox loss: {train_epoch_box_loss:.6f}\ttrain batch loss: {train_epoch_loss:.6f}"
        pbar.set_description_str(loss_status)
    
    print()
    pbar = tqdm(valid_dataloader, total=len(valid_dataloader), ncols=200)
    valid_epoch_loss, valid_epoch_cls_loss, valid_epoch_obj_loss, valid_epoch_box_loss = 0, 0, 0, 0
    
    model.eval()
    with torch.no_grad():
        for valid_minibatch, (img, gridinfo) in enumerate(pbar):
            img = img.to(cfg.DEVICE)
            gridinfo = gridinfo.to(cfg.DEVICE)
            
            res = model(img)

            calc_loss, cls_loss, box_loss, reg_loss = criterion(res, gridinfo)
            
            valid_epoch_loss = (valid_epoch_loss * valid_minibatch + calc_loss.item()) / (valid_minibatch + 1)
            valid_epoch_cls_loss = (valid_epoch_cls_loss * valid_minibatch + cls_loss.item()) / (valid_minibatch + 1)
            valid_epoch_obj_loss = (valid_epoch_obj_loss * valid_minibatch + box_loss.item()) / (valid_minibatch + 1)
            valid_epoch_box_loss = (valid_epoch_box_loss * valid_minibatch+ reg_loss.item()) / (valid_minibatch + 1)

            loss_status = f"Epoch: {epoch}/{epochs}\tcls loss: {valid_epoch_cls_loss:.6f}\tobj loss: {valid_epoch_obj_loss:.6}\tbox loss: {valid_epoch_box_loss:.6f}\tvalid batch loss: {valid_epoch_loss:.6f}"
            pbar.set_description_str(loss_status)
    
    print()
    
    earlystopper(valid_epoch_loss, model=model)
    
    if earlystopper.early_stop:
        print("earlystopping activate")
        break