import time
import argparse
import random

import torch
import torchvision.transforms as transforms
import torchvision
import apex

from resnet import resnet50
from optimal_grad_checkpointing.solver import optimal_grad_checkpointing


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='data', help='Path to dataset')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
parser.add_argument('--n-exp', default=1, type=int, help='Number of experiment')
parser.add_argument('--epochs', default=15, type=int, help='Nnumber of total epochs to run')
parser.add_argument('--batch-size', default=16, type=int, help='Batch size (default: 16)')
parser.add_argument('--accumulation-steps', default=1, type=int, help='Batch accumulation parameter')
parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='Weight decay (default: 1e-4)')
parser.add_argument('--pretrained', action='store_true', help='Load pretrained model')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
parser.add_argument('--use-cudnn-benchmark', action='store_true', help='Enable cudnn benchmark')
parser.add_argument('--use-amp', action='store_true', help='Enable mixed precision')
parser.add_argument('--use-apex-amp', default="O0", type=str, choices=["O0", "O1", "O2", "O3"], help='Enable Apex mixed precision')
parser.add_argument('--use-checkpointing', default='0', type=str, choices=['0', '1', '2', '3', 'optimal'], help='Choose checkpointing')


def main():
    args = parser.parse_args()

    seed = 2021
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = args.use_cudnn_benchmark

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    # create model
    model = resnet50(num_classes=2, pretrained=args.pretrained, use_checkpointing=args.use_checkpointing)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Data loading code
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    train_data = torchvision.datasets.ImageFolder('data/dog vs cat/dataset/training_set', transform=train_transform)
    valid_data = torchvision.datasets.ImageFolder('data/dog vs cat/dataset/test_set', transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(valid_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    elapsed_time = AverageMeter('Time', ':.2f')
    
    run_segment = None
    if not args.use_apex_amp == "O0":
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level=args.use_apex_amp, verbosity=0)
    if args.use_checkpointing == 'optimal':
        input_size = (70, 3, 224, 224)
        inp = torch.randn(*input_size).cuda(args.gpu)
        run_segment = optimal_grad_checkpointing(model, inp)
    
    best_acc, best_epoch = 0, 0
    for epoch in range(args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, criterion, optimizer, args, run_segment)
        elapsed_time.update(time.time() - start_time)
        # evaluate on validation set
        acc = validate(val_loader, model, args)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
    
    title = '| n exp | bs | use cudnn benchmark | use amp | apex amp | checkpointing | average epoch time, sec | val acc | best epoch | GPU memory, Gb |'
    delimiter = '| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |'
    values = f'| {args.n_exp} | {args.batch_size * args.accumulation_steps} ' \
             f'| {args.use_cudnn_benchmark} ' \
             f'| {args.use_amp} | {args.use_apex_amp} | {args.use_checkpointing} ' \
             f'| {elapsed_time.avg:.2f} | {best_acc:.4f} | {best_epoch} ' \
             f'| {torch.cuda.max_memory_allocated(args.gpu) / (1024 * 1024 * 1024):.2f} |'

    print(title)
    print(delimiter)
    print(values)


def train(train_loader, model, criterion, optimizer, args, run_segment):
    # switch to train mode
    model.train()
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    for batch_idx, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            if args.use_checkpointing == 'optimal':
                images.requires_grad = True
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.use_amp:
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.use_checkpointing == 'optimal':
                    output = run_segment(images)
                else:
                    output = model(images)
                loss = criterion(output, target)
                loss = loss / args.accumulation_steps
        else:
            if args.use_checkpointing == 'optimal':
                output = run_segment(images)
            else:
                output = model(images)
            loss = criterion(output, target)

        # compute gradient
        if args.use_amp:
            scaler.scale(loss).backward()
        elif args.use_apex_amp in ("O1", "O2", "O3"):
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # do SGD step
        if (batch_idx + 1) % args.accumulation_steps == 0 or batch_idx + 1 == len(train_loader):
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()  
                optimizer.zero_grad()


def validate(val_loader, model, args):
    top1 = AverageMeter('Acc@1', ':.4f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for images, target in val_loader:
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            # measure accuracy
            acc = accuracy(output, target, topk=(1,))[0]
            top1.update(acc[0], images.size(0))
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
