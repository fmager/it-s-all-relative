import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

import data
import models
import utils

parser = argparse.ArgumentParser(description='Ensemble evaluation')

parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                help='result directory (default: /tmp/eval)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')

parser.add_argument('--model', type=str, default=None, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--seeds', type=int, nargs='+', default=[0], metavar='N',
                    help='seeds for random number generators (default: [0])')
parser.add_argument('--ckpt', type=str, default='', metavar='CKPT',
                    help='checkpoint to evaluate')

args = parser.parse_args()


args.dir = os.path.join(args.dir, args.dataset.lower(), args.model.lower())

print(f'seeds: {args.seeds}')
print(f'ckpt: {args.ckpt}')
print(f'dir: {args.dir}')

checkpoints = [os.path.join(args.dir, f'seed_{seed}', args.ckpt) for seed in args.seeds]

print(f'Checkpoints: {checkpoints}')

torch.backends.cudnn.benchmark = True

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test
)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

model.cuda()

predictions_sum = np.zeros((len(loaders['test'].dataset), num_classes))
model_err = np.zeros(len(checkpoints))
ens_err = np.zeros(len(checkpoints))

for i, path in enumerate(checkpoints):

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])

    predictions, targets = utils.predictions(loaders['test'], model)
    err = 100.0 * (1- np.mean(np.argmax(predictions, axis=1) == targets))

    predictions_sum += predictions
    ens_err[i] = 100.0 * (1-np.mean(np.argmax(predictions_sum, axis=1) == targets))

    model_err[i] = err

    print('Model accuracy: %8.4f. Ensemble accuracy: %8.4f' % (model_err[i] , ens_err[i]))

path = os.path.join(args.dir, f'performance_seeds_{args.seeds[0]}-{args.seeds[-1]}.npz')
print(f'Saving results to {path}')
np.savez(
    path,
    model_err=model_err,
    ens_err=ens_err,
)

