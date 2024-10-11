
import argparse
import numpy as np
import os
import tabulate
import torch
import torch.nn.functional as F
import tqdm

import data
import models
import curves
import utils
import measures

parser = argparse.ArgumentParser(description='Relative representation evaluation')
parser.add_argument('--dir', type=str, default='/tmp/eval', metavar='DIR',
                help='result directory (default: /tmp/eval)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--curve', type=str, default='PolyChain', metavar='CURVE',
                    help='curve type to use (default: PolyChain)')
parser.add_argument('--seed_from_to', type=str, default='0-1', metavar='SEED',
                    help='seed range (default: 0-1)')
parser.add_argument('--num_bends', type=int, default=3, metavar='N',
                    help='number of curve bends (default: 3)')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--data_path', type=str, default='./data', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_test', action='store_true',
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=2, metavar='N',
                    help='number of workers (default: 2)')
parser.add_argument('--num_points', type=int, default=61, metavar='N',
                    help='number of points on the curve (default: 61)')
parser.add_argument('--num_anchors', type=int, default=512, metavar='N',
                    help='number of anchors (default: 512)')
parser.add_argument('--projection', type=str, default='cosine', metavar='TYPE',
                    help='relative projection type (default: cosine)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--sampling_method', type=str, default='linear', metavar='METHOD',
                    help='sampling method for t (default: linear)')
parser.add_argument('--layer_name', type=str, default='', metavar='LAYER',
                    help='layer to evaluate')
parser.add_argument('--center', action='store_true', default=False,
                    help='center the latent space before projection')
parser.add_argument('--standardize', action='store_true', default=False,
                    help='standardize latent space before projection')
parser.add_argument('--n_batches', type=int, default=1, metavar='N',
                    help='number of batches to use for evaluation (default: 1)')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.dir = os.path.join(args.dir, args.dataset.lower(), args.model.lower(), f'seed_{args.seed_from_to.replace('-','_to_')}', args.curve.lower())
args.ckpt = os.path.join(args.dir, args.ckpt)
os.makedirs(args.dir, exist_ok=True)

# Print command line arguments nicely
print('Command line arguments:')
for arg in vars(args):
    print(f'{arg}: {getattr(args, arg)}')

loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    args.use_test,
    shuffle_train=False,
    drop_last=False
)

targets = []
for _, t in loaders['test']:
    targets.append(t)
targets = torch.cat(targets, dim=0)

indices = torch.randperm(len(loaders['test'].dataset))[:args.num_anchors]
subset = torch.utils.data.Subset(loaders['test'].dataset, indices)

# Create the DataLoader
loaders['anchors'] = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

architecture = getattr(models, args.model)
curve = getattr(curves, args.curve)
model = curves.CurveNet(
    num_classes,
    curve,
    architecture.curve,
    args.num_bends,
    architecture_kwargs=architecture.kwargs,
)
model.to(device)
checkpoint = torch.load(args.ckpt)
model.load_state_dict(checkpoint['model_state'])

criterion = F.cross_entropy
regularizer = curves.l2_regularizer(args.wd)

if args.projection == 'cosine':
    proj_fn = utils.cosine_projection
elif args.projection == 'euclidean':
    proj_fn = utils.euclidean_projection
elif args.projection == 'basis_norm':
    proj_fn = utils.basis_norm_projection
else:
    raise ValueError('Unknown projection type: %s' % args.projection)

rel_proj = utils.RelProjector(proj_fn, center=args.center, standardize=args.standardize)

def sample_t(num_points, method: str = 'linear'):
    '''
    Sample t values from the interval [0, 1]
    '''
    if method == 'linear':
        return np.linspace(0.0, 1.0, num_points)
    elif method == 'normal':
        grid = np.sort(np.random.beta(2, 2, num_points))  # Beta distribution
        grid[0] = 0.0
        grid[-1] = 1.0
        return grid
    elif method == 'uniform':
        return np.sort(np.random.uniform(0.0, 1.0, num_points))
    else:
        raise ValueError('Unknown method: %s' % method)

T = args.num_points
ts = sample_t(T, args.sampling_method)
results = utils.get_cumulative_alignment(loaders, model, args.layer_name, rel_proj, ts, N_batches=args.n_batches)

path = os.path.join(args.dir, f'rho_cumulative_{args.sampling_method}_{args.projection}.npz')
print(f'Saving results to {path}')
np.savez(
    path,
    ts=ts,
    dl_0 = results['dl_0'],
    dl_1 = results['dl_1'],
    absolute_alignment = results['alignment_absolute'],
    relative_alignment = results['alignment_relative'],
)

