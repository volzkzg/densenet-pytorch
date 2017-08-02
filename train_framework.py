import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import densenet as dn

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--batchnorm-decay', '--bd', default=1e-4, type=float,
                    help='batchnorm decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=100, type=int,
                    help='total number of layers (default: 100)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--reg', default=0, type=float,
                    help='regularization parameter')
parser.add_argument('--reg-method', default=0, type=int,
                    help='regularization method')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)


def main():
    global args, best_acc, suffix

    datasize = [1000, 2000, 4000, 8000, 16000, 32000, 50000]
    args = parser.parse_args()
    if args.tensorboard: configure("runs/%s"%(args.name))
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    if args.augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    dataset_train = datasets.CIFAR10('/home/gh349/bicheng/data', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('/home/gh349/bicheng/data/', train=False, transform=transform_test)

    for size in datasize:
        suffix = " - " + str(size)
        tmp_train = random.sample(dataset_train, size)
        tmp_test = dataset_test
        kwargs = {'num_workers': 12, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            tmp_train,
            batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            tmp_test,
            batch_size=args.batch_size, shuffle=True, **kwargs)

        # create model
        model = dn.DenseNet3(args.layers, 10, args.growth, reduction=args.reduce,
                             bottleneck=args.bottleneck, dropRate=args.droprate)
        # get the number of model parameters
        print('Number of model parameters: {}'.format(
            sum([p.data.nelement() for p in model.parameters()])))
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        model = model.cuda()

        cudnn.benchmark = True

        # define loss function (criterion) and pptimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

        best_acc = 0
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            acc_train = train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            acc_val = validate(val_loader, model, criterion, epoch)

            if args.tensorboard:
                log_value("generalization error" + suffix, acc_train - acc_val, epoch)

            # remember best precision and save checkpoint
            is_best = acc_val > best_acc
            best_acc = max(acc_val, best_acc)

        print('Best accuracy' + suffix + ': ', best_acc)
        if args.tensorboard:
            log_value('dataset accuracy', best_acc, size)


def train(train_loader, model, criterion, optimizer, epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        # print(top1.avg, top1.count)
        # print(prec1[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        add_regularization(model, args.reg_method, args.reg)
        #add_regularization(model, args.batchnorm_decay, 1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss' + suffix, losses.avg, epoch)
        log_value('train_acc' + suffix, top1.avg, epoch)

    return top1.avg


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data[0], input.size(0))
        print(top1.avg, top1.count)
        print(prec1[0], input.size(0))

        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def add_regularization(model, reg_method, param):
    {0: add_combined_reg(model, param),
     1: add_separate_reg(model, param)}[reg_method]


def add_combined_reg(model, param):
    # pass

    init_features = len(model.block1.layer[0].bn1.weight.data)
    for j in range(1, model.nblayer):
        reg = reg_param
        for i in range(j):
            reg = ((i+1)*reg_param/model.nblayer)
            weight = model.block1.layer[j].bn1.weight
            feature = model.block1.layer[j].conv1.weight
            print feature.data
            # take the square of each element in the feature maps
            feature_sum = feature**2
            # sum all 48 feature maps of the convolution layer
            feature_sum = feature_sum.sum(0)
            feature_sum = (feature_sum.view_as(weight.data)) # expand the feature sum into a tensor with same dimensions as BN weights
            
            print weight.grad.data
            weight.grad.data[init_features+(i)*12:init_features+(i+1)*12]+= reg*(feature_sum.data[init_features+i*12:init_features+(i+1)*12])*(weight.data[init_features+(i)*12:init_features+(i+1)*12])
            print weight.grad.data
            print '##'

            for k in range(48):
                feature.grad.data[k][init_features+i*12:init_features+(i+1)*12]+=reg*feature.data[k][init_features+i*12:init_features+(i+1)*12]*(weight.data[init_features+(i)*12:init_features+(i+1)*12])**2
    exit(0)

    init_features = len(model.block2.layer[0].bn1.weight.data)
    for j in range(1, model.nblayer):
        reg = reg_param
        for i in range(j):
            reg = ((i+1)*reg_param/model.nblayer)
            weight = model.block2.layer[j].bn1.weight
            feature = model.block2.layer[j].conv1.weight
            
            # take the square of each element in the feature maps
            feature_sum = feature**2
            # sum all 48 feature maps of the convolution layer
            feature_sum = feature_sum.sum(0)
            feature_sum = (feature_sum.view_as(weight.data)) # expand the feature sum into a tensor with same dimensions as BN weights
            weight.grad.data[init_features+(i)*12:init_features+(i+1)*12]+= reg*(feature_sum.data[init_features+i*12:init_features+(i+1)*12])*(weight.data[init_features+(i)*12:init_features+(i+1)*12])

            for k in range(48):
                feature.grad.data[k][init_features+i*12:init_features+(i+1)*12]+=reg*feature.data[k][init_features+i*12:init_features+(i+1)*12]*(weight.data[init_features+(i)*12:init_features+(i+1)*12])**2

    init_features = len(model.block3.layer[0].bn1.weight.data)
    for j in range(1, model.nblayer):
        reg = reg_param
        for i in range(j):
            reg = ((i+1)*reg_param/model.nblayer)
            weight = model.block3.layer[j].bn1.weight
            feature = model.block3.layer[j].conv1.weight
            
            # take the square of each element in the feature maps
            feature_sum = feature**2
            # sum all 48 feature maps of the convolution layer
            feature_sum = feature_sum.sum(0)
            feature_sum = (feature_sum.view_as(weight.data)) # expand the feature sum into a tensor with same dimensions as BN weights
            weight.grad.data[init_features+(i)*12:init_features+(i+1)*12]+= reg*(feature_sum.data[init_features+i*12:init_features+(i+1)*12])*(weight.data[init_features+(i)*12:init_features+(i+1)*12])

            for k in range(48):
                feature.grad.data[k][init_features+i*12:init_features+(i+1)*12]+=reg*feature.data[k][init_features+i*12:init_features+(i+1)*12]*(weight.data[init_features+(i)*12:init_features+(i+1)*12])**2

    



def add_separate_reg(model, param):
    pass


if __name__ == '__main__':
    main()
