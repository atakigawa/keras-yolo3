#! /usr/bin/env python
# coding=utf-8

import numpy as np
import tensorflow as tf


# Discard all boxes with low scores and high IOU
def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on gpu ---------------------------------------*/

    Arguments:
            boxes  -- tensor of shape [1, 10647, 4] # 10647 boxes
            scores -- tensor of shape [1, 10647, num_classes], scores of boxes
            classes -- the return value of function `read_coco_names`
    Note:Applies Non-max suppression (NMS) to set of boxes. Prunes away boxes that have high
    intersection-over-union (IOU) overlap with previously selected boxes.

    max_boxes -- integer, maximum number of predicted boxes you'd like, default is 20
    score_thresh -- real value, if [ highest class probability score < score_threshold]
                       then get rid of the corresponding box
    iou_thresh -- real value, "intersection over union" threshold used for NMS filtering
    """

    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1,4])  # '-1' means we don't know the exact number of boxes
    # confs = tf.reshape(confs, [-1,1])
    score = tf.reshape(scores, [-1,num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    score_thresh = tf.constant(score_thresh, dtype=score.dtype)
    mask = tf.greater_equal(score, score_thresh)
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(
               boxes=tf.cast(filter_boxes, tf.float32),
               scores=tf.cast(filter_score, tf.float32),
               max_output_size=max_boxes,
               iou_threshold=iou_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label


def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes => shape of [-1, 4], the value of '-1' means that dont know the
                        exact number of boxes
               scores => shape of [-1,]
               max_boxes => representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh => representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep[:max_boxes]


def cpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.3, iou_thresh=0.5):
    """
    /*----------------------------------- NMS on cpu ---------------------------------------*/
    Arguments:
        boxes ==> shape [1, 10647, 4]
        scores ==> shape [1, 10647, num_classes]
    """

    boxes = boxes.reshape(-1, 4)
    scores = scores.reshape(-1, num_classes)
    # Picked bounding boxes
    picked_boxes, picked_score, picked_label = [], [], []

    for i in range(num_classes):
        indices = np.where(scores[:,i] >= score_thresh)
        filter_boxes = boxes[indices]
        filter_scores = scores[:,i][indices]
        if len(filter_boxes) == 0:
            continue
        # do non_max_suppression on the cpu
        indices = py_nms(filter_boxes, filter_scores,
                         max_boxes=max_boxes, iou_thresh=iou_thresh)
        picked_boxes.append(filter_boxes[indices])
        picked_score.append(filter_scores[indices])
        picked_label.append(np.ones(len(indices), dtype='int32')*i)
    if len(picked_boxes) == 0:
        return [], [], []

    boxes = np.concatenate(picked_boxes, axis=0)
    score = np.concatenate(picked_score, axis=0)
    label = np.concatenate(picked_label, axis=0)

    return boxes, score, label


def resize_image_correct_bbox(image, boxes, image_h, image_w):

    origin_image_size = tf.to_float(tf.shape(image)[0:2])
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_w / origin_image_size[1]
    yy1 = boxes[:, 1] * image_h / origin_image_size[0]
    xx2 = boxes[:, 2] * image_w / origin_image_size[1]
    yy2 = boxes[:, 3] * image_h / origin_image_size[0]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes


def bbox_iou(A, B):
    intersect_mins = np.maximum(A[:, 0:2], B[:, 0:2])
    intersect_maxs = np.minimum(A[:, 2:4], B[:, 2:4])
    intersect_wh = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    A_area = np.prod(A[:, 2:4] - A[:, 0:2], axis=1)
    B_area = np.prod(B[:, 2:4] - B[:, 0:2], axis=1)

    iou = intersect_area / (A_area + B_area - intersect_area)

    return iou
