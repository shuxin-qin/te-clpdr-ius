import tensorflow as tf
import numpy as np
import os
import glob
import cv2
import time
import cfg
from lpdr_net import LpdrNet
from utils.decode import decode_hp
from utils.image import get_affine_transform, affine_transform
from utils.utils import image_preporcess, py_nms, post_process_hp, lp_draw_on_img
from utils.lpr_util import decode_sparse_tensor, is_legal_lpnumber

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def parse_lp(img_name):

    fn, _ = os.path.splitext(img_name)
    plist = fn.split('-')
    lpn7 = plist[4].split('_')
    pro = provinces[int(lpn7[0])]
    lpnumber = []
    lpnumber.append(pro)
    for i in range(6):
        lpnumber.append(ads[int(lpn7[i+1])])
    lpnumber = ''.join(lpnumber)
    return lpnumber

def save_txt(slist, fname):
 
    # 写之前，先检验文件是否存在，存在就删掉
    if os.path.exists(fname):
        os.remove(fname)
    # 以写的方式打开文件，如果文件不存在，就会自动创建
    with open(fname, 'w') as file:
        for var in slist:
            file.writelines(var)
            file.write('\n')

#查找所有解码序列，寻找符合规则且概率最高的序列作为识别结果
def pl_regularization(bs_decodes):
    #识别结果序列
    detected_list = []
    length = len(decode_sparse_tensor(bs_decodes[0]))
    detec_lists = [[] for i in range(length)]

    #解码所有beam search序列
    for i in range(len(bs_decodes)):
        d_batch = decode_sparse_tensor(bs_decodes[i])
        for j in range(len(d_batch)):
            detec_lists[j].append(d_batch[j])
            
    for i in range(len(detec_lists)):
        detects = detec_lists[i]
        detect = detec_lists[i][0]
        for j in range(len(detects)):
            d = detects[j]
            if is_legal_lpnumber(d, [7]):
                detect = d
                break
        detected_list.append(detect)
        #detected_list_prob.append(detect_prob)
    return detected_list

def inference():

    ckpt_path='./weights50/lpdr-mix-180530'
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config = config)
    cfgs = cfg.Config()
    cfgs.set_batch(1)

    heads={'hm':1, 'wh':2, 'offset':2, 'hm_hp':4, 'hp_kp':8, 'hp_offset':2}
    inputs = tf.placeholder(shape=[None, None, None, 3],dtype=tf.float32)
    model = LpdrNet(inputs, heads, is_training=False, cfgs=cfgs)
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    det = model.det()
    logits = model.logit()

    # decode lpnumber
    seq_len = tf.constant(np.ones(cfgs.k, dtype=np.int32) * 24)
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len, beam_width=10, merge_repeated=False)


    img_folder = 'data/test'
    out_path = 'data/test_out'
    img_names = os.listdir(img_folder)
    tp = 0
    false_list = []
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_folder, img_name)
        original_image = cv2.imread(img_path)
        original_image_size = original_image.shape[:2]
        image_data = image_preporcess(np.copy(original_image), [1024, 640], cfgs.MEAN_PIXEL)
        image_data = image_data[np.newaxis, ...]

        t0 = time.time()
        detections, decode = sess.run([det, decoded], feed_dict={inputs: image_data})
        detections = post_process_hp(detections, original_image_size, [1024, 640], cfgs.down_ratio, cfgs.score_threshold)
        t1 = time.time()
        lp_list = pl_regularization(decode)
        #lp_list = decode_sparse_tensor(decode)
        print('Inferencce took %.1f ms %.1f ms (%.2f fps)' % ((time.time()-t0)*1000, (time.time()-t1)*1000, 1/(time.time()-t0)))
        
        if cfgs.use_nms:
            results = []
            lp_results = []
            classified_bboxes = detections[:, :4]
            classified_scores = detections[:, 4]
            inds = py_nms(classified_bboxes, classified_scores, max_boxes=2, iou_thresh=0.5)
            results.extend(detections[inds])
            results = np.asarray(results)
            for j in inds:
                lp_results.append(lp_list[j]) 
            
            if len(results) != 0:
                bboxes = results[:,0:4]
                scores = results[:,4]
                classes = results[:, -1]
                kps = results[:, 5:-1]
                lp_list = lp_results
                num = bboxes.shape[0]
            else:
                num = 0
        else:
            bboxes = detections[:,0:4]
            scores = detections[:,4]
            classes = detections[:,-1]
            kps = detections[:, 5:-1]
            num = bboxes.shape[0]
        
        rec_lp_list = []
        flag = 'False'
        for k in range(num):
            rec_lp = ''.join(lp_list[k])
            rec_lp_list.append(rec_lp)

            gt_lp = parse_lp(img_name)
            msg = rec_lp+' ----- '+gt_lp
            if rec_lp == gt_lp:
                flag = 'True'
                tp += 1
            else:
                flag = 'False'
                print(img_name)
                false_list.append(img_name+msg)
            print(i+1, tp, k, flag, msg)

        if num > 0:
            original_image = lp_draw_on_img(original_image, classes, scores, bboxes, kps, desc=rec_lp_list)

        cv2.imwrite(os.path.join(out_path, img_name), original_image)
    
    print(tp, len(img_names), tp/len(img_names))
    save_txt(false_list, os.path.join(out_path, 'result18.txt'))


if __name__ == '__main__':

    inference()