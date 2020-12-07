import os
import cv2
import numpy as np
import io
import random, math
from utils.data_aug import random_crop, random_translate, random_scale, scale_crop
from utils.lpr_util import sparse_tuple_from, CHARS_DICT, decode_sparse_tensor

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class DataReader(object):
    def __init__(self, img_dir, config=None):
        '''Load a subset of the COCO dataset.
        '''
        self.config = config
        self.img_dir = img_dir
        self.max_objs = 1
        self.num_classes = 1
        self.num_joints = 4

        self.images = self.get_img_list(self.img_dir)
        self.num_samples = len(self.images)
        self.shuffle()

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        
        img_id = self.images[idx]
        bboxes, kpts, lpnumber = self.parse_lp(img_id)
        img_path = os.path.join(self.img_dir, img_id)

        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # image resize and cut to 512x512
        size = (self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        img, bboxes, kpts = scale_crop(img, bboxes, kpts, size)
        #print(img.shape)
        # data augmentation np.random.randint(5) 0,1,2,3,4
        flag = np.random.randint(5)
        if flag < 2:
            img, bboxes, kpts = random_scale(img, bboxes, kpts)
        elif flag == 2:
            img, bboxes, kpts = random_crop(img, bboxes, kpts)
        elif flag == 3:
            img, bboxes, kpts = random_translate(img, bboxes, kpts)
        
        # 处理图像，缩放到指定大小
        img, scale_factor = self.imrescale_wh(img, self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        img = self.imnormalize(img, self.config.MEAN_PIXEL, self.config.STD_PIXEL)
        img = self.impad_to_wh(img, self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT)
        bboxes = bboxes.astype(np.float32)
        kpts = kpts.astype(np.float32)

        labels = np.ones((1, 13), dtype=np.float32)
        labels[:, :4] = bboxes
        kpts_f = np.reshape(kpts, (1, 8))
        labels[:, 5:] = kpts_f
        labels = labels*scale_factor / 4

        output_h = self.config.IMAGE_HEIGHT//4
        output_w = self.config.IMAGE_WIDTH//4
        hm = np.zeros((output_h, output_w, self.num_classes), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.float32)

        hm_hp = np.zeros((output_h, output_w, self.num_joints), dtype=np.float32)
        hp_offset = np.zeros((self.max_objs * self.num_joints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * self.num_joints), dtype=np.float32)
        hp_mask = np.zeros((self.max_objs * self.num_joints), dtype=np.float32)

        kps = np.zeros((self.max_objs, self.num_joints*2), dtype=np.float32)
        kps_mask = np.zeros((self.max_objs, self.num_joints*2), dtype=np.float32)

        gt_det = []
        for k in range(self.max_objs):
            bbox = bboxes[k]
            cls_id = 0
            pts = kpts[k]
            # process bbox
            bbox = bbox * scale_factor / 4 #缩放 scale and 1/4 
            bbox = np.clip(bbox, 0, output_h - 1)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                rh, rw = self.gaussian_radius((math.ceil(h),math.ceil(w)))
                rh = max(0, int(rh))
                rw = max(0, int(rw))
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                self.draw_umich_gaussian(hm[:,:, cls_id], ct_int, rw, rh)

                hp_radius = self.gaussian_radius_c((math.ceil(h), math.ceil(w)))
                hp_radius = max(0, int(hp_radius)) 
                for j in range(self.num_joints):
                    pts[j] = pts[j] * scale_factor/4  #缩放 scale and 1/4 
                    if pts[j, 0] >= 0 and pts[j, 0] < output_w and pts[j, 1] >= 0 and pts[j, 1] < output_h:
                        kps[k, j * 2: j * 2 + 2] = pts[j] - ct_int
                        kps_mask[k, j * 2: j * 2 + 2] = 1
                        pt_int = pts[j].astype(np.int32)
                        hp_offset[k * self.num_joints + j] = pts[j] - pt_int
                        hp_ind[k * self.num_joints + j] = pt_int[1] * output_w + pt_int[0]
                        hp_mask[k * self.num_joints + j] = 1

                        self.draw_umich_gaussian_c(hm_hp[..., j], pt_int, hp_radius)

        return img, hm, wh, reg, reg_mask, ind, hm_hp, hp_offset, \
               hp_ind, hp_mask, kps, kps_mask, lpnumber, labels


    def shuffle(self):
        random.shuffle(self.images)

    def get_img_list(self, img_path, exts=['jpg', 'png', 'jpeg', 'JPG']):
        
        img_list = os.listdir(img_path)
        new_list = []
        for img_name in img_list:
            for ext in exts:
                if img_name.endswith(ext):
                    new_list.append(img_name)
                    break
        return new_list

    def parse_lp(self, img_name):

        fn, _ = os.path.splitext(img_name)
        #print(fn)
        plist = fn.split('-')
        bbox = plist[2].split('_')
        box = [[int(pt.split('&')[0]), int(pt.split('&')[1])] for pt in bbox]
        box = sum(box, [])  #[178, 467, 410, 539]
        box = [box,]  #[[178, 467, 410, 539]] 矩形框 x1,y1, x2,y2
        box = np.array(box)
        #print(box)

        pt4 = plist[3].split('_')
        pt4 = np.array(pt4)[[2, 3, 0, 1]]
        pts = np.array([[int(pt.split('&')[0]), int(pt.split('&')[1])] for pt in pt4])
        pts = pts[np.newaxis, ...]

        # lpnumber
        lpn7 = plist[4].split('_')
        #lpnum = ''.join(lpn7)
        #0_9_19_30_29_33_29  皖kv6595
        pro = provinces[int(lpn7[0])]
        lpnumber = []
        lpnumber.append(pro)
        for i in range(6):
            lpnumber.append(ads[int(lpn7[i+1])])
        lp_code = []
        for nu in lpnumber:
            lp_code.append(CHARS_DICT[nu])

        lp_code = np.array(lp_code)

        return box, pts, lp_code


    def imrescale(self, img, scale):
        h, w = img.shape[:2]
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
        rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return rescaled_img, scale_factor

    def imrescale_wh(self, img, width, height):
        h, w = img.shape[:2]

        scale_factor = min(width*1.0/w, height*1.0/h)
        new_size = (int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5))
        rescaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
        return rescaled_img, scale_factor


    def imnormalize(self, img, mean, std):
        img = (img - mean) / std    
        return img.astype(np.float32)

    def impad_to_square(self, img, pad_size):
        h,w = img.shape[:2]
        if len(img.shape) == 2:
            pad_size = [[0,pad_size-h], [0,pad_size-w]]
        else:
            pad_size = [[0,pad_size-h], [0,pad_size-w], [0,0]]
        pad = np.pad(img, pad_size, 'constant')
        return pad

    def impad_to_wh(self, img, width, height):
        h,w = img.shape[:2]
        pad_size = [[0,height-h], [0,width-w], [0,0]]
        pad = np.pad(img, pad_size, 'constant')
        return pad

    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size
        ra = 0.1155*height
        rb = 0.1155*width
        return ra, rb

    def gaussian2D(self, shape, sigmah=1, sigmaw=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x / (2*sigmaw*sigmaw) + y * y / (2*sigmah*sigmah)))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian(self, heatmap, center, rw, rh, k=1):
        diameterw = 2 * rw + 1
        diameterh = 2 * rh + 1
        gaussian = self.gaussian2D((diameterh, diameterw), sigmah=diameterh/6, sigmaw=diameterw/6)

        x, y = center

        height, width = heatmap.shape[0:2]

        left, right = min(x, rw), min(width - x, rw + 1)
        top, bottom = min(y, rh), min(height - y, rh + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[rh - top:rh + bottom, rw - left:rw + right]
        if min(masked_gaussian.shape)>0 and min(masked_heatmap.shape)>0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

        return heatmap

    def gaussian_radius_c(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 - sq1) / (2 * a1)

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 - sq2) / (2 * a2)

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def gaussian2D_c(self, shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_umich_gaussian_c(self, heatmap, center, radius, k=1):

        diameter = 2 * radius + 1
        gaussian = self.gaussian2D_c((diameter, diameter), sigma=diameter / 6)

        x, y = center

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)