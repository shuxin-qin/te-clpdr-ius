import random
import numpy as np
import cv2
import math

def scale_crop(image, bboxes, kpts, size):
    
    h0, w0 = size
    h, w, _ = image.shape
    factor = w0*1.0/w
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    bboxes = bboxes * factor
    kpts = kpts * factor

    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_u = max_bbox[1]
    max_d = h - max_bbox[3]

    dis = min(max_u, max_d)
    dis = min(dis, h-h0)
    crop = int(random.uniform(0, dis))
    if max_u < max_d:
        x0 = crop
    else:
        x0 = h - h0 - crop 

    x0 = min(x0, h-h0)
    x0 = max(0, x0)

    image = image[x0:x0+h0, :, :]

    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - x0
    kpts[:, :, 1] = kpts[:, :, 1] - x0

    return image, bboxes, kpts


def random_scale(image, bboxes, kpts):

    factor = random.uniform(0.5, 0.9)
    h, w, _ = image.shape
    image = cv2.resize(image, (0, 0), fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    bboxes = bboxes * factor
    kpts = kpts * factor
    h1, w1, _ = image.shape
    pad_size = [[0,h-h1], [0,w-w1], [0,0]]
    image = np.pad(image, pad_size, 'constant') 

    return image, bboxes, kpts

def random_crop(image, bboxes, kpts):

    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
    crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
    crop_xmax = min(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
    crop_ymax = min(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

    if (crop_xmax-crop_xmin<200) or (crop_ymax-crop_ymin<200):
        return image, bboxes, kpts

    image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax, :]

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    kpts[:, :, 0] = kpts[:, :, 0] - crop_xmin
    kpts[:, :, 1] = kpts[:, :, 1] - crop_ymin

    return image, bboxes, kpts

def random_translate(image, bboxes, kpts):

    h, w, _ = image.shape
    max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

    max_l_trans = max_bbox[0]
    max_u_trans = max_bbox[1]
    max_r_trans = w - max_bbox[2]
    max_d_trans = h - max_bbox[3]

    tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
    ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

    M = np.array([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h))

    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    kpts[:, :, 0] = kpts[:, :, 0] + tx
    kpts[:, :, 1] = kpts[:, :, 1] + ty

    return image, bboxes, kpts


def random_rotate(image, bboxes, kpts, angle=30, scale=1):

    h, w, _ = image.shape

    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    
    #---------------------- 矫正bbox坐标 ----------------------
    # rot_mat是最终的旋转矩阵
    # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
    rot_bboxes = list()
    rot_kpts = list()
    for bbox in bboxes:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
        point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
        # 合并np.array
        concat = np.vstack((point1, point2, point3, point4))
        # 改变array类型
        concat = concat.astype(np.int32)
        # 得到旋转后的坐标
        rx, ry, rw, rh = cv2.boundingRect(concat)
        rx_min = rx
        ry_min = ry
        rx_max = rx+rw
        ry_max = ry+rh
        # 加入list中
        rot_bboxes.append([rx_min, ry_min, rx_max, ry_max])
        rot_kpts.append(concat)

    rot_bboxes = np.asarray(rot_bboxes)
    rot_kpts = np.asarray(rot_kpts)

    return rot_img, rot_bboxes, rot_kpts


def random_color_distort(img, brightness_delta=32, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img
