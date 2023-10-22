import torch

import torch.nn as nn
from torchvision.models import vgg16
from torchvision.ops import nms

import conf as cfg

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()

        self.backbone = vgg16(weights=True)

        if cfg.FREEZING:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        self.backbone = self.backbone.features[:-1]

        self.added_conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding = 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding = 1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 3, padding = 1),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(7* 7 * 1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7 * 7 * 30),
            nn.Sigmoid() # cx, cy, w, h, confidence, class probabilities 모두 0~1 범위 내에 해당하므로
        )


        for layer in self.added_conv.modules():
    	    if isinstance(layer, nn.Conv2d):
		        nn.init.normal_(layer.weight, mean=0, std=0.01)
                  
        for layer in self.fc.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)

    def inference(self, pred, img_width, img_height, conf_thres=0.25, iou_thres=0.45):
        with torch.no_grad():
            coord_info = pred[:, :, :10]
            coord_info = coord_info.view(7, 7, 2, 5)

            grid_max_conf, _ = pred[:, :, [4, 9]].max(dim=2)
            coord_info = coord_info[pred[:, :, [4, 9]] == grid_max_conf.unsqueeze(-1)].view(7, 7, 5)

            conf_mask = coord_info[:, :, 4] >= conf_thres

            class_info = pred[:, :, 10:]
            class_info = class_info[conf_mask].view(-1, 20)
            coord_info = coord_info[conf_mask]

            after_nms = torch.zeros((coord_info.size(0), 6))

            anchor_coord = coord_info[:, :4]
            anchor_conf = coord_info[:, 4]
            grid_class = class_info.argmax(dim=1)

            nms_ind = nms(anchor_coord, anchor_conf, iou_threshold=iou_thres)

            after_nms[:, :4] = anchor_coord[nms_ind]
            after_nms[:, 4] = anchor_conf[nms_ind]
            after_nms[:, 5] = grid_class

            after_nms[:, 0] *= img_width
            after_nms[:, 1] *= img_height
            after_nms[:, 2] *= img_width
            after_nms[:, 3] *= img_height


        return after_nms
    
    def forward(self, x):
         out = self.backbone(x)
         out = self.added_conv(out)
         out = self.fc(out)
         out = out.view(-1, 7, 7, 30)

         return out