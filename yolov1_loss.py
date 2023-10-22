import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import box_convert, box_iou

import conf as cfg

class YOLOv1Loss(nn.Module):
    def __init__(self):
        super(YOLOv1Loss, self).__init__()

    def forward(self, pred_tensor, target_tensor):
        """
        현재 no object, object loss와 같이 배치단위 처리가 필요없는 부분도 
        배치단위 처리하게 짜놔서 학습 시 속도저하 요인이 되는듯.

        coordinate regression 등 불가피한 부분은 배치단위 처리,
        나머지 부분은 벡터화를 통한 처리속도 개선을 목적으로 하기.
        """
        batch_size = pred_tensor.size(0)

        batch_obj_loss = torch.tensor(0).to(cfg.DEVICE)
        batch_coord_loss = torch.tensor(0).to(cfg.DEVICE)
        batch_cls_loss = torch.tensor(0).to(cfg.DEVICE)
        batch_loss = torch.tensor(0).to(cfg.DEVICE)

        for i in range(batch_size):            
            batch_target_tensor = target_tensor[i]
            batch_pred_tensor = pred_tensor[i].contiguous()
            
            noobj_mask = target_tensor[i, :, :, 4] == 0

            pred_noobj_info = batch_pred_tensor[noobj_mask].view(-1, 30)
            target_noobj_info = batch_target_tensor[noobj_mask].view(-1, 30)

            noobj_conf_mask = torch.ByteTensor(pred_noobj_info.size()).fill_(0).to(cfg.DEVICE)
            noobj_conf_mask[:, [4, 9]] = 1

            noobj_pred_conf = pred_noobj_info[noobj_conf_mask]
            noobj_target_conf = target_noobj_info[noobj_conf_mask]
            
            noobj_loss = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
            # ----------------------------- no object loss end -----------------------------
            obj_mask = target_tensor[i, :, :, 4] > 0
            
            pred_obj_info = batch_pred_tensor[obj_mask].view(-1, 30)
            target_obj_info = batch_target_tensor[obj_mask].view(-1, 30)

            bbox_pred = pred_obj_info[:, :10].contiguous().view(-1, 5)
            bbox_target = target_obj_info[:, :10].contiguous().view(-1, 5)
            
            pred_coord_xyxy = box_convert(bbox_pred[:, :4], in_fmt='cxcywh', out_fmt='xyxy')
            target_coord_xyxy = box_convert(bbox_target[:, :4], in_fmt='cxcywh', out_fmt='xyxy')

            max_iou, _ = box_iou(pred_coord_xyxy, target_coord_xyxy).max(dim=0)

            coord_obj_mask = torch.BoolTensor((bbox_pred.size(0))).to(cfg.DEVICE)
            coord_obj_mask[max_iou != 0] = True

            pred_coord_cxcy = bbox_pred[:, :2][coord_obj_mask]
            pred_coord_wh = bbox_pred[:, 2:4][coord_obj_mask]
            target_coord_cxcy = bbox_target[:, :2][coord_obj_mask]
            target_coord_wh = bbox_target[:, 2:4][coord_obj_mask]

            coord_loss_cxcy = F.mse_loss(pred_coord_cxcy, target_coord_cxcy, reduction='sum')
            coord_loss_wh = F.mse_loss(torch.sqrt(pred_coord_wh), torch.sqrt(target_coord_wh), reduction='sum')
            # ----------------------------- coordinate loss end ----------------------------            
            pred_iou = bbox_pred[:, 4][coord_obj_mask]
            target_iou = bbox_target[:, 4][coord_obj_mask]

            obj_loss = F.mse_loss(pred_iou, target_iou, reduction='sum')       
            # ----------------------------- objectness loss end ----------------------------
            class_target = target_obj_info[:, 10:]
            class_pred = pred_obj_info[:, 10:]
            
            cls_loss = F.mse_loss(class_pred, class_target, reduction='sum')
            # --------------------------- classification loss end --------------------------
            batch_cls_loss = batch_cls_loss + cls_loss
            batch_obj_loss = batch_obj_loss + obj_loss + (noobj_loss * cfg.NOOBJ_LAMBDA)
            batch_coord_loss = batch_coord_loss + (coord_loss_cxcy + coord_loss_wh) * cfg.COORD_LAMBDA
            # ------------------------------- batch loss end ------------------------------

        batch_cls_loss = batch_cls_loss / batch_size
        batch_obj_loss = batch_obj_loss / batch_size
        batch_coord_loss = batch_coord_loss / batch_size

        batch_loss = batch_cls_loss + batch_obj_loss + batch_coord_loss

        return batch_loss, batch_cls_loss, batch_obj_loss, batch_coord_loss