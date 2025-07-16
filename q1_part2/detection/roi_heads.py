from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops, roi_align

from . import _utils as det_utils


def fastrcnn_loss(class_logits, box_regression,angle_regression, labels, regression_targets,angle_targets,angle_binsize): #add angles
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor] # type: ignore
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    # print(classification_loss ,"classification_loss",labels,class_logits)
    # print(labels ,"labels ") 0 0 0 1 1
    # print("logits ",class_logits)
                        # torch.Size([2048]) ,torch.Size([2048, 2])

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    angle_targets = torch.cat(angle_targets, dim=0) #add angles
    
    # angle_targets  torch.Size([2048, 1])
    # angle_regression  torch.Size([2048, 2, 1])
   
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape #2048,2

  
    sample_cnt= box_regression.size(-1) // 4
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    
    # num_bins = angle_regression.size(-1)
    if len(angle_regression.shape) == 2:
        angle_regression = angle_regression.reshape(N, sample_cnt, 1)
    

    box_loss = F.smooth_l1_loss( #uses diff x,y,w,h like 4 box info elements
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    # print("angle_reg ",angle_regression[0], " ",angle_targets[0])

    if angle_binsize==0: #do regression
        angle_loss = F.smooth_l1_loss(
            angle_regression[sampled_pos_inds_subset, labels_pos],
            angle_targets[sampled_pos_inds_subset],
            reduction="sum",
        )
        angle_loss = angle_loss / labels.numel()
        angle_loss = torch.clamp(angle_loss, min=0, max=1000)


    else:  #see dimension once 
        compute_pred = angle_regression[sampled_pos_inds_subset, labels_pos]

        compute_gt = angle_targets[sampled_pos_inds_subset] # list of 0-180 angles

        num_bins = angle_binsize

        gt_bin = (compute_gt / num_bins).long().squeeze(-1) #correct bin class [0,1,28,..]
        num_bins = 180 // angle_binsize  # Number of bins
        gt_bin = torch.clamp(gt_bin, 0, num_bins - 1)

        angle_loss = F.cross_entropy(compute_pred, gt_bin)

    return classification_loss, box_loss, angle_loss #is there need to give box_loss seperately ?

"""Ground Truth and Predicted Bounding Boxes:

    Ground Truth Box: (x_gt, y_gt, w_gt, h_gt) = (100, 100, 50, 50)
    Predicted Box: (x_pred, y_pred, w_pred, h_pred) = (105, 110, 55, 45)

Step 1: Compute Errors (Differences)

    x error: 105−100=5105−100=5
    y error: 110−100=10110−100=10
    w error: 55−50=555−50=5
    h error: 45−50=−545−50=−5

Step 2: Apply Smooth L1 Loss

Using the Smooth L1 loss formula:

    For errors 5, 10, 5, and -5, we compute:
    SmoothL1(5)=5−0.5=4.5,SmoothL1(10)=10−0.5=9.5,SmoothL1(5)=4.5,SmoothL1(−5)=4.5
    SmoothL1(5)=5−0.5=4.5,SmoothL1(10)=10−0.5=9.5,SmoothL1(5)=4.5,SmoothL1(−5)=4.5

Step 3: Sum the Losses

The total box loss is:
Total Box Loss=4.5+9.5+4.5+4.5=23
Total Box Loss=4.5+9.5+4.5+4.5=23"""
class RoIHeads(nn.Module):
    __annotations__ = {
        "box_coder": det_utils.BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        angle_binsize#, add here bin size later
    ):
        super().__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool  
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        self.angle_binsize=angle_binsize

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
        return matched_idxs, labels

    def subsample(self, labels):
        # type: (List[Tensor]) -> List[Tensor]
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        # type: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]

        return proposals

    def check_targets(self, targets):
        # type: (Optional[List[Dict[str, Tensor]]]) -> None
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["boxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_angles = [t["angles"].to(dtype).unsqueeze(-1) for t in targets] #add angles

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        matched_gt_angles = [] #add angles
        
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            gt_angles_in_image = gt_angles[img_id] #add angles

            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                gt_angles_in_image = torch.zeros((1, 1), dtype=dtype, device=device) #add angles

            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
            matched_gt_angles.append(gt_angles_in_image[matched_idxs[img_id]]) #add angles

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets ,matched_gt_angles #add angles

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        box_regression,  # type: Tensor
        angle_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        # add angles
        if len(angle_regression.shape) == 2:
            pred_angles = angle_regression
            num_bins = 1
        else:
            pred_angles = F.softmax(angle_regression, -1)
            num_bins = angle_regression.size(-1)

        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_angles_list = pred_angles.split(boxes_per_image, 0) #add angles

        all_boxes = []
        all_scores = []
        all_labels = []
        all_angles = [] #add angles

        # for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        for boxes, scores, angles, image_shape in zip(pred_boxes_list, pred_scores_list, pred_angles_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            angles = angles[:, 1:] #add angles

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # print(num_bins,"num_bins") #1 reg , 180/binsize for classification

            if num_bins == 1: #add angles
                angles = angles.reshape(-1)
            else:
                angles = angles.reshape(-1, num_bins)
                angles = torch.argmax(angles, dim=-1, keepdim=True)
                bin_val = 180/num_bins
                angles = (angles)*bin_val
                angles = angles.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            # boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            boxes, scores, labels, angles = boxes[inds], scores[inds], labels[inds],angles[inds] #add angles

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            boxes, scores, labels, angles = boxes[keep], scores[keep], labels[keep], angles[keep] #add angles

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            boxes, scores, labels, angles = boxes[keep], scores[keep], labels[keep], angles[keep]   #add angles

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_angles.append(angles) #add angles

        return all_boxes, all_scores, all_labels ,all_angles #add angles

    def forward(
        self,
        features,  # type: Dict[str, Tensor]
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
        targets=None,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets,angle_targets = self.select_training_samples(proposals, targets) #add angles
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression ,angle_regression= self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            
            loss_classifier, loss_box_reg, loss_angle_reg = fastrcnn_loss(class_logits, box_regression, angle_regression, labels, regression_targets, angle_targets,self.angle_binsize) #add angles
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_angle": loss_angle_reg}
        else:
            boxes, scores, labels ,angles = self.postprocess_detections(class_logits, box_regression,angle_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "angles": angles[i], #add angles
                    }
                )

        return result, losses