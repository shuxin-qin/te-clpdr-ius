import os
import numpy as np
import tensorflow as tf
import cv2
import math
import time
import shutil
import cfg
from lpdr_net import LpdrNet
from utils import data_reader, dataset
from net.resnet import load_weights

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train():
    # define dataset
    configs = cfg.Config()
    configs.set_k(1)

    heads={'hm':1, 'wh':2, 'offset':2, 'hm_hp':4, 'hp_kp':8, 'hp_offset':2}

    img_dir = '/home/qinshuxin/datasets/ccpd/ccpd_mix'
    data_source = data_reader.DataReader(img_dir, config=configs)
    
    datasets = dataset.Dataset(data_source, batch_size=configs.BATCH_SIZE)

    in_imgs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name='in_imgs')
    batch_hm = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None], name='batch_hm')
    batch_wh = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='batch_wh')
    batch_reg = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='batch_reg')
    batch_reg_mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_reg_mask')
    batch_ind = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_ind')

    batch_hm_hp = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 4], name='batch_hm_hp')
    batch_hp_off = tf.placeholder(dtype=tf.float32, shape=[None, None, 2], name='batch_hp_off')
    batch_hp_ind = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_hp_ind')
    batch_hp_mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='batch_hp_mask')
    batch_kps = tf.placeholder(dtype=tf.float32, shape=[None, None, 8], name='batch_kps')
    batch_kps_mask = tf.placeholder(dtype=tf.float32, shape=[None, None, 8], name='batch_kps_mask')

    batch_labels = tf.placeholder(dtype=tf.float32, shape=[None, None, 13], name='batch_labels')
    targets = tf.sparse_placeholder(dtype=tf.int32, name='targets')

    # define model and loss
    model = LpdrNet(in_imgs, heads, is_training=True, cfgs=configs, labels=batch_labels)
    with tf.variable_scope('loss'):
        hm_loss, wh_loss, reg_loss, hm_hp_loss, kpt_loss, hm_off_loss = \
            model.detect_loss(batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_hm_hp, 
                batch_kps, batch_kps_mask, batch_hp_off, batch_hp_mask, batch_hp_ind)
        det_loss = hm_loss + wh_loss + reg_loss + hm_hp_loss + kpt_loss + hm_off_loss

        rec_loss = model.recog_loss(targets)
        total_loss = det_loss + 10*rec_loss

    global_step = tf.train.create_global_step()
    training_variables = tf.trainable_variables()
    learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=training_variables)
    clip_grad_var = [(g, v) if g is None else (tf.clip_by_norm(g, 10.), v) for g, v in grads_and_vars]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step, name='train_op')

    saver  = tf.train.Saver(max_to_keep=1)
    saver_best = tf.train.Saver(max_to_keep=1)
    print(len(data_source))

    # calculate edit distance between two sequences
    #seq_len = tf.constant(np.ones(configs.BATCH_SIZE, dtype=np.int32) * 24)
    #decoded, _ = tf.nn.ctc_beam_search_decoder(model.logit(), seq_len, beam_width=10, merge_repeated=False)
    #dis = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with tf.name_scope('summary'):
            tf.summary.scalar("learning_rate", learning_rate)
            tf.summary.scalar("det_loss", det_loss)
            tf.summary.scalar("rec_loss", rec_loss)
            tf.summary.scalar("total_loss", total_loss)

            logdir = "./log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            write_op = tf.summary.merge_all()
            summary_writer  = tf.summary.FileWriter(logdir, graph=sess.graph)
        
        # train 
        sess.run(tf.global_variables_initializer())
        #load_weights(sess,'./pretrained_weights/resnet50.npy')
        #print('load pretrained weights resnet50!')
        saver.restore(sess, './weights18/lpdr-125000')
        print('load pretraned weights!')

        print('Global Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.global_variables()]))
        print('Trainable Variables: ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        print('\n----------- start to train -----------\n')
        best_loss = 1
        for epoch in range(1, 1+configs.epochs):
            epoch_loss = []
            start = time.time()
            step_start = time.time()
            for data in datasets:
                imgs, hms, whs, regs, reg_masks, inds, hm_hps, hp_offsets, \
                    hp_inds, hp_masks, kpss, kps_masks, sparse_labels, labels = data
                feed_dict = {in_imgs:imgs, 
                             batch_hm:hms, 
                             batch_wh:whs, 
                             batch_reg:regs, 
                             batch_reg_mask:reg_masks, 
                             batch_ind:inds,
                             batch_hm_hp:hm_hps,
                             batch_hp_off:hp_offsets,
                             batch_hp_ind:hp_inds,
                             batch_hp_mask:hp_masks,
                             batch_kps:kpss,
                             batch_kps_mask:kps_masks,
                             batch_labels:labels,
                             targets:sparse_labels}
                _, summary, step_loss, det_los, rec_los, hm_los, wh_los, reg_los, hm_hp_los, kpt_los, hm_off_los, step, lr = \
                    sess.run([train_op, write_op, total_loss, det_loss, rec_loss, hm_loss, wh_loss, reg_loss, hm_hp_loss, \
                              kpt_loss, hm_off_loss, global_step, learning_rate], feed_dict=feed_dict)

                epoch_loss.append(step_loss)

                if step % 10 == 0 and step > 0:
                    summary_writer.add_summary(summary, step)
                    step_time = time.time() - step_start
                    step_start = time.time()
                    print(('Epoch:{}, Step:{}, loss:{:.3f}, lr:{:.6f}, det:{:.3f}, rec:{:.5f}, hm:{:2f}, wh:{:.2f}, ' + \
                        'reg:{:.2f}, hp:{:.2f}, kpt:{:.2f}, hm_off:{:.2f}, time:{:.2f}').format(epoch, step, step_loss, lr, 
                            det_los, rec_los, hm_los, wh_los, reg_los, hm_hp_los, kpt_los, hm_off_los, step_time))

            epoch_loss = np.mean(epoch_loss)
            print('Epoch:{}, average loss:{:.3f}, time:{:.2f}'.format(epoch, epoch_loss, time.time()-start))
            saver.save(sess, "weights18/lpdr-mix", global_step=global_step)

            if epoch_loss < best_loss:
                saver_best.save(sess, 'weights18/lpdr-best-mix')
                best_loss = epoch_loss

if __name__ == '__main__': 

    train()