import tensorflow as tf

def nms(heat, kernel=3):
    hmax = tf.layers.max_pooling2d(heat, kernel, 1, padding='same')
    keep = tf.cast(tf.equal(heat, hmax), tf.float32)
    return heat*keep

def topk(hm, K=100):
    batch, height, width, cat = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    #[b,h*w*c]
    scores = tf.reshape(hm, (batch, -1))
    #[b,k]
    topk_scores, topk_inds = tf.nn.top_k(scores, k=K)
    #[b,k]
    topk_clses = topk_inds % cat
    topk_xs = tf.cast(topk_inds // cat % width, tf.float32)
    topk_ys = tf.cast(topk_inds // cat // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs 

def topk_channel(hm, K=100):
    hm = tf.transpose(hm, (0, 3, 1, 2))
    batch, cat, height, width = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    scores = tf.reshape(hm, (batch, 17, -1))

    topk_scores, topk_inds = tf.nn.top_k(scores, k=K)

    topk_xs = tf.cast(topk_inds % width, tf.float32)
    topk_ys = tf.cast(topk_inds // width, tf.float32)
    topk_inds = tf.cast(topk_ys * tf.cast(width, tf.float32) + topk_xs, tf.int32)

    return topk_scores, topk_inds, topk_ys, topk_xs

def decode(heat, wh, reg=None, K=100):
    batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
    heat = nms(heat)
    scores, inds, clses, ys, xs = topk(heat, K=K)

    if reg is not None:
        reg = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
        #[b,k,2]
        reg = tf.batch_gather(reg, inds)
        xs = tf.expand_dims(xs, axis=-1) + reg[..., 0:1]
        ys = tf.expand_dims(ys, axis=-1) + reg[..., 1:2]
    else:
        xs = tf.expand_dims(xs, axis=-1) + 0.5
        ys = tf.expand_dims(ys, axis=-1) + 0.5
    
    #[b,h*w,2]
    wh = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    #[b,k,2]
    wh = tf.batch_gather(wh, inds)

    clses = tf.cast(tf.expand_dims(clses, axis=-1), tf.float32)
    scores = tf.expand_dims(scores, axis=-1)

    xmin = xs - wh[...,0:1] / 2
    ymin = ys - wh[...,1:2] / 2
    xmax = xs + wh[...,0:1] / 2
    ymax = ys + wh[...,1:2] / 2

    bboxes = tf.concat([xmin, ymin, xmax, ymax], axis=-1)

    #[b,k,6]
    detections = tf.concat([bboxes, scores, clses], axis=-1)
    return detections

# 使用tf.map_fn
def decode_hp(heat, wh, reg, hm_hp, hp_kp, hp_off, K=100):
    
    batch, height, width, cat = tf.shape(heat)[0], tf.shape(heat)[1], tf.shape(heat)[2], tf.shape(heat)[3]
    
    heat = nms(heat)
    hm_flat = tf.reshape(heat, (batch, -1))
    reg_flat = tf.reshape(reg, (batch, -1, tf.shape(reg)[-1]))
    wh_flat = tf.reshape(wh, (batch, -1, tf.shape(wh)[-1]))
    kps_flat = tf.reshape(hp_kp, (batch, -1, tf.shape(hp_kp)[-1]))

    hm_hp = nms(hm_hp)
    hm_hp_flat = tf.reshape(hm_hp, (batch, -1, tf.shape(hm_hp)[-1]))
    hp_off_flat = tf.reshape(hp_off, (batch, -1, tf.shape(hp_off)[-1]))

    def _process_sample(args):
        _hm, _reg, _wh, _kps, _hm_hp, _hp_offset = args
        _scores, _inds = tf.nn.top_k(_hm, k=K)
        _classes = tf.cast(_inds % cat, tf.float32)
        _inds = tf.cast(_inds / cat, tf.int32)
        _xs = tf.cast(_inds % width, tf.float32)
        _ys = tf.cast(tf.cast(_inds / width, tf.int32), tf.float32)
        _wh = tf.gather(_wh, _inds)
        _reg = tf.gather(_reg, _inds)
        _kps = tf.gather(_kps, _inds)

        # shift keypoints by their center
        _kps_x = _kps[:, ::2]
        _kps_y = _kps[:, 1::2]
        _kps_x = _kps_x + tf.expand_dims(_xs, axis=-1)  # k x J
        _kps_y = _kps_y + tf.expand_dims(_ys, axis=-1)  # k x J
        _kps = tf.stack([_kps_x, _kps_y], axis=-1)  # k x J x 2

        _xs = _xs + _reg[..., 0]
        _ys = _ys + _reg[..., 1]

        _x1 = _xs - _wh[..., 0] / 2
        _y1 = _ys - _wh[..., 1] / 2
        _x2 = _xs + _wh[..., 0] / 2
        _y2 = _ys + _wh[..., 1] / 2

        # snap center keypoints to the closest heatmap keypoint
        def _process_channel(args):
            __kps, __hm_hp = args
            thresh = 0.1
            __hm_scores, __hm_inds = tf.nn.top_k(__hm_hp, k=K)
            __hm_xs = tf.cast(__hm_inds % width, tf.float32)
            __hm_ys = tf.cast(tf.cast(__hm_inds / width, tf.int32), tf.float32)
            __hp_offset = tf.gather(_hp_offset, __hm_inds)
            __hm_xs = __hm_xs + __hp_offset[..., 0]
            __hm_ys = __hm_ys + __hp_offset[..., 1]
            mask = tf.cast(__hm_scores > thresh, tf.float32)
            __hm_scores = (1. - mask) * -1. + mask * __hm_scores
            __hm_xs = (1. - mask) * -10000. + mask * __hm_xs
            __hm_ys = (1. - mask) * -10000. + mask * __hm_ys
            __hm_kps = tf.stack([__hm_xs, __hm_ys], axis=-1)  # k x 2
            __broadcast_hm_kps = tf.expand_dims(__hm_kps, axis=1)  # k x 1 x 2
            #__broadcast_hm_kps = tf.tile(__broadcast_hm_kps, (1, K, 1))
            __broadcast_kps = tf.expand_dims(__kps, axis=0)  # 1 x k x 2
            #__broadcast_kps = tf.tile(__broadcast_kps, (K, 1, 1))
            dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(__broadcast_kps, __broadcast_hm_kps), 2))  # k, k
            min_dist = tf.reduce_min(dist, 0)
            min_ind = tf.argmin(dist, 0)
            __hm_scores = tf.gather(__hm_scores, min_ind)
            __hm_kps = tf.gather(__hm_kps, min_ind)
            mask = (tf.cast(__hm_kps[..., 0] < _x1, tf.float32) + tf.cast(__hm_kps[..., 0] > _x2, tf.float32) +
                    tf.cast(__hm_kps[..., 1] < _y1, tf.float32) + tf.cast(__hm_kps[..., 1] > _y2, tf.float32) +
                    tf.cast(__hm_scores < thresh, tf.float32) +
                    tf.cast(min_dist > 0.3 * (tf.maximum(_wh[..., 0], _wh[..., 1])), tf.float32))
            mask = tf.expand_dims(mask, -1)
            mask = tf.cast(mask > 0, tf.float32)
            __kps = (1. - mask) * __hm_kps + mask * __kps
            return __kps

        _kps = tf.transpose(_kps, (1, 0, 2))  # J x k x 2
        _hm_hp = tf.transpose(_hm_hp, (1, 0))  # J x -1
        _kps = tf.map_fn(_process_channel, [_kps, _hm_hp], dtype='float32')
        _kps = tf.reshape(tf.transpose(_kps, (1, 0, 2)), (K, -1))  # k x J * 2

        _boxes = tf.stack([_x1, _y1, _x2, _y2], axis = -1)
        _scores = tf.expand_dims(_scores, -1)
        _classes = tf.expand_dims(_classes, -1)
        _detection = tf.concat([_boxes, _scores, _kps, _classes], -1)
        return _detection

    detections = tf.map_fn(_process_sample,
                    [hm_flat, reg_flat, wh_flat, kps_flat, hm_hp_flat, hp_off_flat], dtype=tf.float32)

    return detections

