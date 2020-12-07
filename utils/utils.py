import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont


def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def py_nms(boxes, scores, max_boxes=50, iou_thresh=0.5):
    """
    Pure Python NMS baseline.

    Arguments: boxes: shape of [-1, 4], the value of '-1' means that dont know the
                      exact number of boxes
               scores: shape of [-1,]
               max_boxes: representing the maximum of boxes to be selected by non_max_suppression
               iou_thresh: representing iou_threshold for deciding to keep boxes
    """
    assert boxes.shape[1] == 4 and len(scores.shape) == 1

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
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

def image_preporcess(image, target_size, means, gt_boxes=None):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = image - means

    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_h = ih
    pad_w = iw
    
    pad_size = [[0,pad_h-nh], [0,pad_w-nw], [0,0]]
    img_pad = np.pad(image_resized, pad_size, 'constant')

    if gt_boxes is None:
        return img_pad

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale
        return img_pad, gt_boxes

def post_process(detections, org_img_shape, input_size, down_ratio, score_threshold):
    bboxes = detections[0, :, 0:4]
    scores = detections[0, :, 4]
    classes = detections[0, :, 5]
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size[1] / org_w, input_size[0] / org_h)

    bboxes = 1.0 * (bboxes * down_ratio) / resize_ratio

    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, org_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, org_h)
    
    score_mask = scores >= score_threshold
    bboxes, socres, classes = bboxes[score_mask], scores[score_mask], classes[score_mask]
    return np.concatenate([bboxes, socres[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

# [b,k,4+1+34+1]
def post_process_hp(detections, org_img_shape, input_size, down_ratio, score_threshold):
    bboxes = detections[0, :, 0:4]
    scores = detections[0, :, 4]
    kps = detections[0, :, 5:-1]
    classes = detections[0, :, -1]
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size[1] / org_w, input_size[0] / org_h)

    bboxes = 1.0 * (bboxes * down_ratio) / resize_ratio

    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, org_w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, org_h)

    kps = 1.0 * (kps * down_ratio) / resize_ratio

    kps[:, 0::2] = np.clip(kps[:, 0::2], 0, org_w)
    kps[:, 1::2] = np.clip(kps[:, 1::2], 0, org_h)
    
    score_mask = scores >= score_threshold
    bboxes, socres, kps, classes = bboxes[score_mask], scores[score_mask], kps[score_mask], classes[score_mask]
    return np.concatenate([bboxes, socres[:, np.newaxis], kps, classes[:, np.newaxis]], axis=-1)

def bboxes_draw_on_img(img, classes_id, scores, bboxes, class_names, thickness=2):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors_tableau[int(classes_id[i])]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, thickness)
        # Draw text
        s = '%s: %.2f' % (class_names[int(classes_id[i])], scores[i])
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (y1_src - text_size[1], x1_src)

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img

def poses_draw_on_img(img, classes_id, scores, bboxes, kps, thickness=2):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        color = colors_tableau[int(classes_id[i])]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, thickness)
        # Draw text
        s = '%.2f' % scores[i]
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (y1_src - text_size[1], x1_src)

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

        kpts = kps[i]
        kpts = np.array(kpts).reshape(-1, 2)
        #kpts = kpts + np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        color = colors_tableau[2]
        show_skelenton(img, kpts, color)

    return img


def show_skelenton(img, kpts, color = (255,0,0)):

    for i in range(kpts.shape[0]):
        x,y = kpts[i][0], kpts[i][1]

        cv2.circle(img, (x,y), 3, color)

    skelenton = [[0, 1], [0, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], 
                     [6, 12], [5, 11], [11, 12], [12, 14], [14, 16], [11, 13], [13, 15]]

    for sk in skelenton:

        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)

    return img

def cv2ImgAddText(img, text, left, top, textColor=(255, 255, 255), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "utils/font/Lantinghei.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def lp_draw_on_img(img, classes_id, scores, bboxes, kps, thickness=2, desc=[]):
    colors_tableau = [(158, 218, 229), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207)]
    scale = 0.8
    text_thickness = 2
    line_type = 8
    for i in range(bboxes.shape[0]):
        if scores[i] < 0.5:
            continue
        if i > 0:
            break

        bbox = bboxes[i]
        color = colors_tableau[0]
        # Draw bounding boxes
        x1_src = int(bbox[0])
        y1_src = int(bbox[1])
        x2_src = int(bbox[2])
        y2_src = int(bbox[3])

        cv2.rectangle(img, (x1_src, y1_src), (x2_src, y2_src), color, thickness)
        # Draw text
        #s = '%.2f' % scores[i]
        s = desc[i]
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (y1_src - text_size[1], x1_src)
        #print(text_size)
        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        #cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)
        #img = cv2ImgAddText(img, s, p1[1], p1[0] + baseline, textColor=(255, 0, 0), textSize=30)

        kpts = kps[i]
        kpts = np.array(kpts).reshape(-1, 2)
        #kpts = kpts + np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
        show_lp(img, kpts)
        img = cv2ImgAddText(img, s, p1[1], p1[0]-10, textColor=(255, 0, 0), textSize=25)

    return img


def show_lp(img, kpts, color = (0,255,0)):

    skelenton = [[0, 1], [1, 2], [2, 3], [3, 0]]

    for sk in skelenton:

        pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
        pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))

        if pos1[0]>0 and pos1[1] >0 and pos2[0] >0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)

    return img