import torch
import numpy as np


# https://blog.csdn.net/weixin_43654653/article/details/107972147?ops_request_misc=&request_id=&biz_id=102&utm_term=pixel%20accuracy%E4%BB%A3%E7%A0%81&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-6-107972147.142%5Ev10%5Epc_search_result_control_group,157%5Ev8%5Econtrol&spm=1018.2226.3001.4187
# https://zhuanlan.zhihu.com/p/518797698

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def pixel_accuracy(self):
        """The ratio of correctly classified pixels to all pixels
        Returns:
            acc = (TP + TN) / (TP + TN + FP + TN)
        """
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc

    def pixel_accuracy_class(self):
        """Calculate the ratio of the correctly classified pixels of each class to
        all the pixels of that class and then calculate the average
        Returns:
            acc = (TP) / TP + FP
        """
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nanmean(acc)
        return acc

    def mean_intersection_over_union(self):
        """Calculate the IoU of each category and then average (Intersection and Union ratio).
        IoU is generally calculated based on classes, and it is also calculated on pictures;
        The IoU calculated based on the class is accumulated after the IoU calculation of each class,
        and then averaged, and the obtained is the global evaluation mIoU.
        Returns:
            mIoU
        """
        mIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        mIoU = np.nanmean(mIoU)
        return mIoU

    def frequency_weighted_intersection_over_union(self):
        """It is understood that the weighted summation of the iou of each category is performed according
        to the frequency of occurrence of each category.
        Returns:
            fwIOU = [(TP+FN)/(TP+FP+TN+FN)] * [TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def precision(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        return tp / (tp + fp + 1e-7)

    def recall(self):
        tp = np.diag(self.confusion_matrix)
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        return tp / (tp + fn + 1e-7)

    def f1(self):
        tp = np.diag(self.confusion_matrix)
        fp = np.sum(self.confusion_matrix, axis=0) - tp
        fn = np.sum(self.confusion_matrix, axis=1) - tp
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return (2.0 * precision * recall) / (precision + recall + 1e-7)

    def _generate_matrix(self, gt_image, pre_image):
        """confusion matrix
        Args:
            gt_image: Segmentation map label (numpy array form)
            pre_image: Prediction segmentation map (numpy array format)
        Returns:
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        for lp, lt in zip(pre_image, gt_image):
            self.confusion_matrix += self._generate_matrix(lt.flatten(), lp.flatten())

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)