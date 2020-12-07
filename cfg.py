
import numpy as np

class Config(object):
    
    ############################################################
    # train
    GPU_COUNT = 1
    IMAGES_PER_GPU = 16

    # dataset
    IMAGE_WIDTH = 512
    IMAGE_HEIGHT = 512
    MAX_GT_INSTANCES = 1

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    STD_PIXEL = np.array([1., 1., 1.])

    num_classes = 1
    score_threshold = 0.3
    use_nms = False
    nms_thresh = 0.4
    epochs = 10
    down_ratio = 4

    roi_w = 96
    roi_h = 32
    k = 1 #instances per image
    backbone = 'resnet18'

    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

    def set_batch(self, batch_size):
        self.BATCH_SIZE = batch_size

    def set_k(self, k):
        self.k = k
 


