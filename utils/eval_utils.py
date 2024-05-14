import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes,threshold=0.5):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    #calculate surface area of each box
    pred_area = (pred_boxes[:,2]-pred_boxes[:,0]) * (pred_boxes[:,3]-pred_boxes[:,1])
    gt_area = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1])
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= threshold)
    return accu_num

def trans_vg_mean_iou(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    mean_iou = torch.mean(iou)

    return mean_iou
def trans_vg_cumulative_iou(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    
    # iou = bbox_iou(pred_boxes, gt_boxes)
    # cumulative_iou = torch.sum(iou)
    #calculate the sum of the intersection between the predicted and ground truth boxes
    sum_inter =  torch.sum(torch.min(pred_boxes[:,2:], gt_boxes[:,2:]) - torch.max(pred_boxes[:,:2], gt_boxes[:,:2]))
    #calculate the sum of the union between the predicted and ground truth boxes
    sum_union = torch.sum(torch.max(pred_boxes[:,2:], gt_boxes[:,2:]) - torch.min(pred_boxes[:,:2], gt_boxes[:,:2]))
    return  sum_inter/sum_union


def trans_vg_eval_test_iou(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)

    return accu_num, iou

def avg_size_difference(pred_boxes, gt_boxes):
    # Convert bounding boxes from (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
    pred_boxes = xywh2xyxy(pred_boxes)
    gt_boxes = xywh2xyxy(gt_boxes)

    # Calculate the width and height of each box
    pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]

    # Calculate the absolute differences in widths and heights
    width_diff = torch.abs(pred_widths - gt_widths)
    height_diff = torch.abs(pred_heights - gt_heights)

    # Calculate the average difference
    avg_width_diff = torch.mean(width_diff)
    avg_height_diff = torch.mean(height_diff)

    return avg_width_diff, avg_height_diff

def avg_position_difference(pred_boxes, gt_boxes):
    # Extract center coordinates from both predicted and ground truth boxes
    pred_x_centers = pred_boxes[:, 0]
    pred_y_centers = pred_boxes[:, 1]
    gt_x_centers = gt_boxes[:, 0]
    gt_y_centers = gt_boxes[:, 1]

    # Calculate the absolute differences in x and y coordinates
    x_diff = torch.abs(pred_x_centers - gt_x_centers)
    y_diff = torch.abs(pred_y_centers - gt_y_centers)

    # Calculate the average differences
    avg_x_diff = torch.mean(x_diff)
    avg_y_diff = torch.mean(y_diff)

    return avg_x_diff, avg_y_diff

def trans_vg_eval_test_iom(pred_boxes, gt_boxes):
    # Calculer l'intersection
    intersection = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:]) - torch.max(pred_boxes[:, :2], gt_boxes[:, :2])
    intersection = intersection.clamp(min=0).prod(dim=1)

    # Calculer l'aire de prédiction et de ground truth
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Calculer le minimum de l'aire de prédiction et de ground truth
    min_areas = torch.min(pred_area, gt_area)

    # Calculer l'IoM (Intersection over Minimum)
    iom = intersection / min_areas

    # Déterminer les prédictions correctes (IoM > seuil, généralement 0.5 pour la détection d'objets)
    correct = (iom > 0.5).sum().item()

    return correct
def bounding_box_regression_loss(pred_boxes, gt_boxes):
    # Calculer la perte L1 entre les prédictions et les ground truth
    loss = torch.abs(pred_boxes - gt_boxes).sum(dim=1).mean()
    return loss

def trans_vg_eval_test_giou(pred_boxes, gt_boxes):
    # Convertir les boîtes au format x1y1x2y2
    pred_boxes = xywh2xyxy(pred_boxes)
    gt_boxes = xywh2xyxy(gt_boxes)

    # Calculer l'intersection
    intersection = torch.min(pred_boxes[:, 2:], gt_boxes[:, 2:]) - torch.max(pred_boxes[:, :2], gt_boxes[:, :2])
    intersection = intersection.clamp(min=0).prod(dim=1)

    # Calculer l'aire de prédiction et de ground truth
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Calculer l'union
    union = pred_area + gt_area - intersection

    # Calculer les boîtes englobantes minimales (enclosing boxes)
    enclose_x1y1 = torch.min(pred_boxes[:, :2], gt_boxes[:, :2])
    enclose_x2y2 = torch.max(pred_boxes[:, 2:], gt_boxes[:, 2:])
    enclose_area = (enclose_x2y2[:, 0] - enclose_x1y1[:, 0]) * (enclose_x2y2[:, 1] - enclose_x1y1[:, 1])

    # Calculer GIoU
    giou = intersection / union - (enclose_area - union) / enclose_area

    # Déterminer les prédictions correctes (IoM > seuil, généralement 0.5 pour la détection d'objets)
    correct = (giou > 0.5).sum().item()

    return correct