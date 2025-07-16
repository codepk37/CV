import torch
import numpy as np
import cv2
import torchvision
import argparse
import random
import os
import yaml
from tqdm import tqdm
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import detection
from detection.faster_rcnn import FastRCNNPredictor
from detection.anchor_utils import AnchorGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import math
from shapely.geometry import Polygon

def get_rotated_box(x1, y1, x2, y2, angle):
    """
    Convert (x1, y1, x2, y2, angle) into a rotated Polygon.
    """
    # Center of the rectangle
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Dimensions of the rectangle
    w, h = x2 - x1, y2 - y1
    
    # Convert angle to radians
    angle = math.radians(angle)
    
    # Rectangle corners relative to center
    dx, dy = w / 2, h / 2
    corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    
    # Rotation matrix
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    # Rotate corners and translate to the center
    rotated_corners = np.dot(corners, rotation_matrix) + [xc, yc]

    # Return the rotated polygon
    return Polygon(rotated_corners)

def get_iou(det, gt): #to use
    """
    Compute IoU for rotated boxes.
    Args:
    - det: A tuple (x1, y1, x2, y2, angle) for the detected bounding box.
    - gt: A tuple (x1, y1, x2, y2, angle) for the ground truth bounding box.
    Returns:
    - IoU value for the rotated boxes.
    """
    # Get the rotated polygons
    det_polygon = get_rotated_box(*det)
    gt_polygon = get_rotated_box(*gt)
    
    # Compute the intersection and union of the polygons
    intersection = det_polygon.intersection(gt_polygon)
    union = det_polygon.union(gt_polygon)
    
    # Compute the areas of intersection and union
    area_intersection = intersection.area
    area_union = union.area
    
    # Compute IoU
    if area_union == 0:
        return 0.0
    iou = area_intersection / area_union
    return iou



def load_model_and_dataset(filename):
    # Read the config file #
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st = SceneTextDataset('test', root_dir=dataset_config['root_dir'])
    test_dataset = DataLoader(st, batch_size=1, shuffle=False)

    faster_rcnn_model = detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                            min_size=600,
                                                            max_size=1000,
                                                            box_score_thresh=0.7,
                                                            angle_binsize=train_config['angle_binsize'],
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'],angle_binsize=train_config['angle_binsize'])

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)
    faster_rcnn_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']),
                                                    map_location=device))

    return faster_rcnn_model, st, test_dataset


    
def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    mean_precisions = [0]
    mean_recalls = [0]
    # average precisions for ALL classes
    aps = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        # Compute the average precision (mean of all precision values)
        mean_precision = np.mean(precisions)
        mean_recall = np.mean(recalls)

        # Store the mean precision and recall for this class
        mean_precisions.append(mean_precision)
        mean_recalls.append(mean_recall)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan


    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)

    # Check if mean_precisions is not empty
    if len(mean_precisions) > 0:
        overall_precision = np.mean(mean_precisions)
    else:
        overall_precision = 0  # or some other default value (e.g., np.nan)

    # Check if mean_recalls is not empty
    if len(mean_recalls) > 0:
        overall_recall = np.mean(mean_recalls)
    else:
        overall_recall = 0  # or some other default value (e.g., np.nan)

    return mean_ap, all_aps, overall_precision, overall_recall





def evaluate_map(filename):
    faster_rcnn_model, voc, test_dataset = load_model_and_dataset(filename)
    gts = []
    preds = []
    for im, target, fname in tqdm(test_dataset):
        im_name = fname
        im = im.float().to(device)
        target_angles = target['angles'].float().to(device)[0]
        target_boxes = target['bboxes'].float().to(device)[0]
        target_labels = target['labels'].long().to(device)[0]
        frcnn_output = faster_rcnn_model(im, None)[0]

        boxes = frcnn_output['boxes']
        angles = frcnn_output['angles']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']

        pred_boxes = {}
        gt_boxes = {}
        for label_name in voc.label2idx:
            pred_boxes[label_name] = []
            gt_boxes[label_name] = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = labels[idx].detach().cpu().item()
            score = scores[idx].detach().cpu().item()
            angle = angles[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            pred_boxes[label_name].append([x1, y1, x2, y2, angle,score])
        for idx, box in enumerate(target_boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            label = target_labels[idx].detach().cpu().item()
            label_name = voc.idx2label[label]
            angle = target_angles[idx].detach().cpu().item()
            gt_boxes[label_name].append([x1, y1, x2, y2,angle])

        gts.append(gt_boxes)
        preds.append(pred_boxes)

    
    mean_ap, all_aps, overall_precision, overall_recall = compute_map(preds, gts, method='interp')
    return mean_ap


if __name__ == '__main__':

    a= evaluate_map('config/st_reg.yaml')
    print(a)