
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
parser.add_argument('--num_anchors', type=int, default=512, metavar='N',
                    help='number of anchors (default: 512)')
parser.add_argument('--projection', type=str, default='cosine', metavar='TYPE',
                    help='relative projection type (default: cosine)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--layer_name', type=str, default='', metavar='LAYER',
                    help='layer to evaluate')
parser.add_argument('--center', action='store_true', default=False,
                    help='center the latent space before projection')
parser.add_argument('--standardize', action='store_true', default=False,
                    help='standardize latent space before projection')
parser.add_argument('--seeds', type=int, nargs='+', default=[0], metavar='N',
                    help='seeds for random number generators (default: [0])')
parser.add_argument('--ckpt', type=str, default='', metavar='CKPT',
                    help='checkpoint to evaluate')
parser.add_argument('--num_test_points', type=int, default=20, metavar='N',
                    help='number of test points (default: 20)')


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args.dir = os.path.join(args.dir, args.dataset.lower(), args.model.lower())
checkpoints = [os.path.join(args.dir, f'seed_{seed}', args.ckpt) for seed in args.seeds]

print(f'Checkpoints: {checkpoints}')

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

indices = torch.randperm(len(loaders['test'].dataset))
subset = torch.utils.data.Subset(loaders['test'].dataset, indices[:args.num_anchors])

loaders['anchors'] = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

subset = torch.utils.data.Subset(loaders['test'].dataset, indices[args.num_anchors:args.num_anchors + args.num_test_points])

loaders['test'] = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

architecture = getattr(models, args.model)


model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model.to(device)

if args.projection == 'cosine':
    proj_fn = utils.cosine_projection
elif args.projection == 'euclidean':
    proj_fn = utils.euclidean_projection
elif args.projection == 'basis_norm':
    proj_fn = utils.basis_norm_projection
else:
    raise ValueError('Unknown projection type: %s' % args.projection)

rel_proj = utils.RelProjector(proj_fn, center=args.center, standardize=args.standardize)

latent = torch.Tensor()

def forward_hook(module, input, output):
    global latent
    latent = input[0].clone().detach()
    while latent.dim() > 2:
        latent = latent.mean(dim=-1)

layer = getattr(model, args.layer_name)
while isinstance(layer, (torch.nn.ModuleList, torch.nn.Sequential)):
    layer = layer[-1]

print(f'Layer: {layer}')

handle = layer.register_forward_hook(forward_hook)

ckpt_absolute = []
ckpt_relative = []

absolute_alignment = np.zeros((len(checkpoints)))
relative_alignment = np.zeros((len(checkpoints)))
acc = np.zeros((len(checkpoints)))

columns = ['Checkpoint', 'Acc', 'Absolute Align.', 'Relative Align.']

for i, checkpoint in enumerate(checkpoints):

    model.load_state_dict(torch.load(checkpoint)['model_state'])
    model.eval()

    if rel_proj.center is True or rel_proj.standardize is True:
        norm_layer = utils.train_norm_layer(model, None, loaders['train'], args.layer_name, norm='bn')
        rel_proj.update_stats(norm_layer.running_mean, torch.sqrt(norm_layer.running_var + 1e-5))

    anchors = []
    for input, target in loaders['anchors']:
        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        with torch.no_grad():
            _ = model(input)
        anchors.append(latent.clone().detach())

    anchors = torch.cat(anchors, dim=0)

    absolute = []
    relative = []
    targets = []
    preds = []
    correct = 0
    total = 0

    for input, target in loaders['test']:
        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)
        with torch.no_grad():
            pred = model(input)
        absolute.append(latent.clone().detach().cpu())
        relative.append((rel_proj.project(latent.clone().detach(), anchors)).cpu())  

        # Count the total number of predictions
        total += target.size(0)

        # Count the number of correct predictions
        correct += (pred.argmax(dim=1) == target).sum().item()

    # Calculate the accuracy
    acc[i] = correct / total

    ckpt_absolute.append(torch.cat(absolute, dim=0).unsqueeze(-1))
    ckpt_relative.append(torch.cat(relative, dim=0).unsqueeze(-1))

    absolute_alignment[i], _, _ = measures.sample_align(torch.cat(ckpt_absolute, dim=-1))
    relative_alignment[i], _, _ = measures.sample_align(torch.cat(ckpt_relative, dim=-1))

    values = [i, acc[i], absolute_alignment[i], relative_alignment[i]]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')

    if i % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

path = os.path.join(args.dir, f'alignment_seeds_{args.projection}_{args.seeds[0]}-{args.seeds[-1]}.npz')
print(f'Saving results to {path}')
np.savez(
    path,
    absolute_alignment=absolute_alignment,
    relative_alignment=relative_alignment
)

