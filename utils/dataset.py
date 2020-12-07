#! /usr/bin/env python
# coding=utf-8

import os
import random
import numpy as np
from utils.lpr_util import sparse_tuple_from

class Dataset(object):
    """implement Dataset here"""
    def __init__(self, dataset, batch_size=4, shuffle=True):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_count = 0
        self.num_samples = len(self.dataset)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        return self

    def __next__(self):

        imgs, hms, whs, regs, reg_masks, inds, hm_hps, hp_offsets, hp_inds, hp_masks, kpss, \
            kps_masks, lpnums, labels = [], [], [], [], [], [], [], [], [], [], [], [], [], []

        num = 0
        if self.batch_count < self.num_batchs:
            while num < self.batch_size:
                index = self.batch_count * self.batch_size + num
                if index >= self.num_samples: 
                    index -= self.num_samples
                
                img, hm, wh, reg, reg_mask, ind, hm_hp, hp_offset, hp_ind, hp_mask, kps, \
                    kps_mask, lpnum, label = self.dataset[index]

                imgs.append(img)
                hms.append(hm)
                whs.append(wh)
                regs.append(reg)
                reg_masks.append(reg_mask)
                inds.append(ind)

                hm_hps.append(hm_hp)
                hp_offsets.append(hp_offset)
                hp_inds.append(hp_ind)
                hp_masks.append(hp_mask)
                kpss.append(kps)
                kps_masks.append(kps_mask)
                lpnums.append(lpnum)
                labels.append(label)

                num += 1

            self.batch_count += 1
            imgs = np.asarray(imgs)
            hms = np.asarray(hms)
            whs = np.asarray(whs)
            regs = np.asarray(regs)
            reg_masks = np.asarray(reg_masks)
            inds = np.asarray(inds)

            hm_hps = np.asarray(hm_hps)
            hp_offsets = np.asarray(hp_offsets)
            hp_inds = np.asarray(hp_inds)
            hp_masks = np.asarray(hp_masks)
            kpss = np.asarray(kpss)
            kps_masks = np.asarray(kps_masks)
            lpnums = np.asarray(lpnums)
            labels = np.asarray(labels)

            sparse_labels = sparse_tuple_from(lpnums)

            return imgs, hms, whs, regs, reg_masks, inds, hm_hps, hp_offsets, \
                   hp_inds, hp_masks, kpss, kps_masks, sparse_labels, labels
        
        else:
            self.batch_count = 0
            if self.shuffle:
                self.dataset.shuffle()
            raise StopIteration

    def __len__(self):
        return self.num_batchs

    