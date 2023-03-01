from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable

from dataset import make_data_loader

from utils.metrics import Evaluator
from utils.saver import Saver
from utils.loss import SegmentationLosses

from model.FPN import FPN


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')

    parser.add_argument('--dataset', dest='dataset', type=str, default='Landslide4Sense',
                        help='training dataset')
    parser.add_argument('--net', dest='net', type=str, default='resnet101',
                        help='resnet18, resnet34, resnet50, etc.')
    parser.add_argument('--start_epoch', dest='start_epoch', type=int, default=1,
                        help='starting epoch')
    parser.add_argument('--epochs', dest='epochs', type=int, default=35,
                        help='number of iterations to train')
    parser.add_argument('--save_dir', dest='save_dir', default=None, nargs=argparse.REMAINDER,
                        help='directory to save models')
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=0,
                        help='number of worker to load dataset')

    # cuda
    parser.add_argument('--cuda', dest='cuda', type=bool, default=True,
                        help='whether use CUDA')

    # multiple GPUs
    parser.add_argument('--mGPUs', dest='mGPUs', type=bool, default=False,
                        help='whether use multiple GPUs')
    parser.add_argument('--gpu_ids', dest='gpu_ids', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')

    # batch size
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                        help='batch_size')

    # config optimization
    parser.add_argument('--o', dest='optimizer', type=str, default='sgd',
                        help='training optimizer')
    parser.add_argument('--lr', dest='lr', type=float, default=0.01,
                        help='starting learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-5,
                        help='weight_decay')
    parser.add_argument('--lr_decay_step', dest='lr_decay_step', type=int, default=50,
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
    parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=1,
                        help='iterval to do evaluate')

    parser.add_argument('--checkname', dest='checkname', type=str, default=None,
                        help='checkname')

    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    parser.add_argument("--num_class", type=int, default=2,
                        help="number of classes.")

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        if args.dataset == 'Landslide4Sense':
            kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
            self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)

        # Define network
        if args.net == 'resnet101':
            blocks = [2, 4, 23, 3]
            fpn = FPN(blocks, args.num_class, back_bone=args.net)

        # Define Optimizer
        self.lr = self.args.lr

        if args.optimizer == 'adam':
            self.lr = self.lr * 0.1
            opt = torch.optim.Adam(fpn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            opt = torch.optim.SGD(fpn.parameters(), lr=args.lr, momentum=0, weight_decay=args.weight_decay)

        # Define criterion
        if args.dataset == 'Cityscapes':
            self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode='ce')

        self.model = fpn
        self.optimizer = opt

        # Define Evaluator
        self.evaluator = Evaluator(self.num_class)

        # multiple mGPUs
        if args.mGPUs:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        self.lr_stage = [68, 93]
        self.lr_stage_ind = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()

        if self.lr_stage_ind > 1 and epoch % (self.lr_stage[self.lr_stage_ind]) == 0:
            adjust_learning_rate(self.optimizer, self.args.lr_decay_gamma)
            self.lr *= self.args.lr_decay_gamma
            self.lr_stage_ind += 1

        for batch_id, batch in enumerate(self.train_loader):
            if self.args.dataset == 'Landslide4Sense':
                image, target, _, _ = batch
            else:
                raise NotImplementedError

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            self.optimizer.zero_grad()
            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long())
            # loss_train = loss.item()
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            train_loss += loss.item()

            if batch_id % 10 == 0:
                print("Epoch[{}]({}/{}):Loss:{:.4f}, learning rate={}".format(epoch, batch_id, len(self.train_loader),
                                                                              loss.data, self.lr))
        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_id * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        # save checkpoint every epoch
        if self.args.no_val:
            is_best = False

            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred
                }, is_best)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0

        for batch_id, batch in enumerate(self.val_loader):
            if self.args.dataset == 'Landslide4Sense':
                image, target, _, _ = batch
            else:
                raise NotImplementedError

            if self.args.cuda:
                image, target = image.cuda(), target.cuda()

            with torch.no_grad():
                output = self.model(image)

            loss = self.criterion(output, target)
            test_loss += loss.item()
            print('Test Loss:%.3f' % (test_loss / (batch_id + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        acc = self.evaluator.pixel_accuracy()
        acc_class = self.evaluator.pixel_accuracy_class()
        mIoU = self.evaluator.mean_intersection_over_union()
        fwIoU = self.evaluator.frequency_weighted_intersection_over_union()

        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, batch_id * self.args.batch_size + image.shape[0]))
        print("Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(acc, acc_class, mIoU, fwIoU))
        print('Loss: %.3f' % test_loss)

        new_pred = mIoU

        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred
                }, is_best)


def main():
    args = parse_args()

    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), 'run')

    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    if args.cuda and args.mGPUs:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integer only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {
            'Landslide4Sense': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)

    trainer = Trainer(args)

    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoch:', trainer.args.epochs)

    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)

        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)


if __name__ == '__main__':
    main()
