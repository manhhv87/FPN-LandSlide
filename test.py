from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import argparse

import torch

from mypath import Path
from utils.metrics import Evaluator
from dataset import make_data_loader

from model.FPN import FPN


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')

    parser.add_argument('--dataset', dest='dataset', type=str, default='Landslide4Sense',
                        help='training dataset')
    parser.add_argument('--net', dest='net', type=str, default='resnet101',
                        help='resnet101, res152, etc')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=1,
                        help='starting epoch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=2000,
                        help='number of iterations to train')
    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        default="D:\\disk\\midterm\\experiment\\code\\semantic\\fpn\\fpn\\run",
                        help='directory to save models')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='number of worker to load dataset')

    # cuda
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true',
                        help='whether use multiple GPUs')

    # batch size
    parser.add_argument('--bs', dest='batch_size', type=int, default=5,
                        help='batch_size')

    # config optimization
    parser.add_argument('--o', dest='optimizer', type=str, default='sgd',
                        help='training optimizer')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                        help='starting learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', type=int, default=500,
                        help='step to do learning rate decay, uint is epoch')
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', type=float, default=0.1,
                        help='learning rate decay ratio')

    # set training session
    parser.add_argument('--s', dest='session', type=int, default=1,
                        help='training session')

    # resume trained model
    parser.add_argument('--r', dest='resume', type=bool, default=False,
                        help='resume checkpoint or not')
    parser.add_argument('--checksession', dest='checksession', type=int, default=1,
                        help='checksession to load model')
    parser.add_argument('--checkepoch', dest='checkepoch', type=int, default=1,
                        help='checkepoch to load model')
    parser.add_argument('--checkpoint', dest='checkpoint', type=int, default=0,
                        help='checkpoint to load model')

    # log and display
    parser.add_argument('--use_tfboard', dest='use_tfboard', type=bool, default=True,
                        help='whether use tensorflow tensorboard')

    # configure validation
    parser.add_argument('--no_val', dest='no_val', type=bool, default=False,
                        help='not do validation')
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=2,
                        help='iterval to do evaluate')

    parser.add_argument('--checkname', dest='checkname', type=str, default=None,
                        help='checkname')

    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    # test confit
    parser.add_argument('--plot', dest='plot', type=bool, default=False,
                        help='wether plot test result image')
    parser.add_argument('--exp_dir', dest='experiment_dir', type=str,
                        help='dir of experiment')

    parser.add_argument("--num_class", type=int, default=2,
                        help="number of classes.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.net == 'resnet101':
        blocks = [2, 4, 23, 3]
        model = FPN(blocks, args.num_class, back_bone=args.net)

    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    evaluator = Evaluator(args.num_class)

    # Trained model path and name
    experiment_dir = args.experiment_dir
    load_name = os.path.join(experiment_dir, 'checkpoint.pth.tar')

    # Load trained model
    if not os.path.isfile(load_name):
        raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))

    print('====>loading trained model from ' + load_name)
    checkpoint = torch.load(load_name)
    # checkepoch = checkpoint['epoch']

    if args.cuda:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    if args.dataset == "Landslide4Sense":
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
        _, _, test_loader = make_data_loader(args, **kwargs)
    else:
        raise RuntimeError("dataset {} not found.".format(args.dataset))

    for _, batch in enumerate(test_loader):
        if args.dataset == 'Landslide4Sense':
            image, target, _, _ = batch
        else:
            raise NotImplementedError

        if args.cuda:
            image, target, model = image.cuda(), target.cuda(), model.cuda()

        with torch.no_grad():
            output = model(image)

        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        target = target.cpu().numpy()
        evaluator.add_batch(target, pred)

    acc = evaluator.pixel_accuracy()
    acc_class = evaluator.pixel_accuracy_class()
    mIoU = evaluator.mean_intersection_over_union()
    fwIoU = evaluator.frequency_weighted_intersection_over_union()

    print('Mean evaluate result on dataset {}'.format(args.dataset))
    print('acc:{:.3f}\tacc_class:{:.3f}\nmIoU:{:.3f}\tfwIoU:{:.3f}'.format(acc, acc_class, mIoU, fwIoU))


if __name__ == "__main__":
    main()
