import tensorflow as tf
import loss
from net import resnet
from net.layers import _conv, upsampling
import numpy as np
from utils.decode import decode_hp
from utils.transform_tf import four_point_transform_2D_batch
from utils.lpr_util import NUM_CHARS

class LpdrNet():
    def __init__(self, inputs, heads, is_training, cfgs, labels=None): # heads={'hm':1, 'wh':2, 'offset':2, 'hm_hp':4, 'hp_kp':8, 'hp_offset':2}
        self.is_training = is_training
        self.heads = heads
        self.cfgs = cfgs
        self.k = self.cfgs.k
        self.backbone = self.cfgs.backbone
        #self.seq_len = tf.constant(np.ones(self.cfgs.BATCH_SIZE, dtype=np.int32) * 24)
        
        ch, cw = self.cfgs.roi_h, self.cfgs.roi_w
        batch = self.cfgs.BATCH_SIZE
        self.tgt_pts = tf.convert_to_tensor([[0, 0], [0, cw], [ch, cw], [ch, 0]])
        self.tgt_pts = tf.broadcast_to(self.tgt_pts, shape=[batch, 4, 2])

        try:
            self.pred_hm, self.pred_wh, self.pred_reg, self.pred_hm_hp, self.pred_hp_kp, \
                self.pred_hp_off, self.pred_logits, self.pred_det, self.features = self._build_model(inputs, labels)
        except:
            raise NotImplementedError("Can not build up network!")


    def _build_model(self, inputs, labels=None):
        with tf.variable_scope('resnet'):
            if self.backbone == 'resnet18':
                c2, c3, c4, c5 = resnet.resnet18(is_training=self.is_training).forward(inputs)
            elif self.backbone == 'resnet50':
                c2, c3, c4, c5 = resnet.resnet50(is_training=self.is_training).forward(inputs)

            p5 = _conv(c5, 128, [1,1], is_training=self.is_training)

            up_p5 = upsampling(p5, method='resize')
            reduce_dim_c4 = _conv(c4, 128, [1,1], is_training=self.is_training)
            p4 = 0.5*up_p5 + 0.5*reduce_dim_c4

            up_p4 = upsampling(p4, method='resize')
            reduce_dim_c3 = _conv(c3, 128, [1,1], is_training=self.is_training)
            p3 = 0.5*up_p4 + 0.5*reduce_dim_c3

            up_p3 = upsampling(p3, method='resize')
            reduce_dim_c2 = _conv(c2, 128, [1,1], is_training=self.is_training)
            p2 = 0.5*up_p3 + 0.5*reduce_dim_c2

            features = _conv(p2, 128, [3,3], is_training=self.is_training)

        with tf.variable_scope('detector'):
            hm = _conv(features, 64, [3,3], is_training=self.is_training)
            hm = tf.layers.conv2d(hm, self.heads['hm'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm')
            # wh
            wh = _conv(features, 64, [3,3], is_training=self.is_training)
            wh = tf.layers.conv2d(wh, self.heads['wh'], 1, 1, padding='valid', activation = None, name='wh')
            # offset
            reg =  _conv(features, 64, [3,3], is_training=self.is_training)
            reg = tf.layers.conv2d(reg, self.heads['offset'], 1, 1, padding='valid', activation = None, name='reg')

            # pose heat map
            hm_hp = _conv(features, 64, [3,3], is_training=self.is_training)
            hm_hp = tf.layers.conv2d(hm_hp, self.heads['hm_hp'], 1, 1, padding='valid', activation = tf.nn.sigmoid, 
                                  bias_initializer=tf.constant_initializer(-np.log(99.)), name='hm_hp')
            # pose keypoints
            hp_kp = _conv(features, 64, [3,3], is_training=self.is_training)
            hp_kp = tf.layers.conv2d(hp_kp, self.heads['hp_kp'], 1, 1, padding='valid', activation = None, name='hp_kp')
            # keypoints offset
            hp_off = _conv(features, 64, [3,3], is_training=self.is_training)
            hp_off = tf.layers.conv2d(hp_off, self.heads['hp_offset'], 1, 1, padding='valid', activation = None, name='hp_off')

        with tf.variable_scope('recognizer'):
            #for training, using labels
            if self.is_training:
                det = labels
            else:
                det = decode_hp(hm, wh, reg, hm_hp, hp_kp, hp_off, K=self.k)

            ch, cw = self.cfgs.roi_h, self.cfgs.roi_w
            batch = self.cfgs.BATCH_SIZE
            img_shape = tf.shape(features)
            #det = decode_hp(hm, wh, reg, hm_hp, hp_kp, hp_off, K=1)
            # get cropped rois as the lp area
            boxes = det[:,:,0:4]
            boxes = tf.reshape(boxes, (-1,4))
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h
            N = img_shape[0]
            normalized_rois = tf.transpose(tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]))

            #features = tf.stop_gradient(features)
            k = self.k
            ind = tf.cast(tf.range(0, N*k)/k, tf.int32)
            cropped_roi_features = tf.image.crop_and_resize(features, normalized_rois,
                                                            box_ind=ind,
                                                            crop_size=[ch, cw],
                                                            name='CROP_AND_RESIZE')
            # affine transform by detected four kpts of lp
            kpts = det[:,:,5:13]
            kpts = tf.reshape(kpts, (-1,8))
            kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4 = \
                    kpts[:,0],kpts[:,1],kpts[:,2],kpts[:,3],kpts[:,4],kpts[:,5],kpts[:,6],kpts[:,7]
            #normalize kpts to box position
            kx1 = (kx1-x1)*cw/(x2-x1+1)
            kx2 = (kx2-x1)*cw/(x2-x1+1)
            kx3 = (kx3-x1)*cw/(x2-x1+1)
            kx4 = (kx4-x1)*cw/(x2-x1+1)
            ky1 = (ky1-y1)*ch/(y2-y1+1)
            ky2 = (ky2-y1)*ch/(y2-y1+1)
            ky3 = (ky3-y1)*ch/(y2-y1+1)
            ky4 = (ky4-y1)*ch/(y2-y1+1)
            kpts = tf.transpose(tf.stack([kx1, ky1, kx2, ky2, kx3, ky3, kx4, ky4]))
            kpts = tf.reshape(kpts, (batch*k, 4, 2))
            kpts = kpts[:,:,::-1]
            #tgt_pts = tf.convert_to_tensor([[0, 0], [0, cw], [ch, cw], [ch, 0]])
            #tgt_pts = tf.broadcast_to(tgt_pts, shape=[batch, 4, 2])
            # lp feature after transform
            tgt_image = four_point_transform_2D_batch(cropped_roi_features, kpts, self.tgt_pts)

            # recognition net
            # N*36*96*128
            con1 = _conv(tgt_image, 128, [3,3], is_training=self.is_training)
            # N*36*48*128
            pool1 = tf.nn.max_pool(con1, ksize=[1, 3, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
            # N*36*48*128
            con2 = _conv(pool1, 128, [3,3], is_training=self.is_training)
            # N*36*24*128
            pool2 = tf.nn.max_pool(con2, ksize=[1, 3, 3, 1], strides=[1, 1, 2, 1], padding='SAME')
            # N*36*24*128
            con3 = _conv(pool2, 128, [3,3], is_training=self.is_training)
            # N*36*24*128
            con4 = _conv(con3, 128, [3,3], is_training=self.is_training)
            # N*36*24*67
            con5 = _conv(con4, NUM_CHARS+1, [1,1], is_training=self.is_training)
            # N*24*67
            logits = tf.reduce_mean(con5, axis=1)
            # 24*N*67
            logits = tf.transpose(logits, (1, 0, 2), name='logits')

        return hm, wh, reg, hm_hp, hp_kp, hp_off, logits, det, features


    def detect_loss(self, true_hm, true_wh, true_reg, reg_mask, ind, true_hm_hp, 
                     true_kpt, kpt_mask, true_hp_off, hp_mask, hp_ind):

        hm_loss = loss.focal_loss(self.pred_hm, true_hm)
        wh_loss = 0.05*loss.reg_l1_loss(self.pred_wh, true_wh, ind, reg_mask)
        reg_loss = loss.reg_l1_loss(self.pred_reg, true_reg, ind, reg_mask)

        #pose hm_hp
        hm_hp_loss = loss.focal_loss(self.pred_hm_hp, true_hm_hp)
        kpt_loss = 0.05*loss.reg_l1_loss_kpt(self.pred_hp_kp, true_kpt, ind, kpt_mask)
        hm_off_loss = loss.reg_l1_loss(self.pred_hp_off, true_hp_off, hp_ind, hp_mask)

        return hm_loss, wh_loss, reg_loss, hm_hp_loss, kpt_loss, hm_off_loss

    def recog_loss(self, targets):

        #print('######################')
        #print(self.logits)
        seq_len = tf.constant(np.ones(self.cfgs.BATCH_SIZE*self.k, dtype=np.int32) * 24)
        loss = tf.nn.ctc_loss(labels=targets, inputs=self.pred_logits, sequence_length=seq_len)
        loss = tf.reduce_mean(loss)

        return loss

    def predict(self):
        
        return self.pred_hm, self.pred_wh, self.pred_reg, \
                self.pred_hm_hp, self.pred_hp_kp, self.pred_hp_off, self.pred_logits

    def logit(self):

        return self.pred_logits

    def det(self):

        return self.pred_det

    def feat(self):

        return self.features


