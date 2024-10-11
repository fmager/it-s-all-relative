
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
parser.add_argument('--num_test_points', type=int, default=20, metavar='N',
                    help='number of test points (default: 20)')
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
    batch_size=args.num_test_points,
    shuffle=False,
    num_workers=args.num_workers
)

targets = []
for _, t in loaders['test']:
    targets.append(t)
targets = torch.cat(targets, dim=0)


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
# results = utils.get_cumulative_alignment(loaders, model, args.layer_name, rel_proj, ts, N_batches=args.n_batches)

latent = torch.Tensor()

def forward_hook(module, input, output):
    global latent
    latent = input[0].clone().detach()
    while latent.dim() > 2:
        latent = latent.mean(dim=-1)

layer = getattr(model.net, args.layer_name)
while isinstance(layer, torch.nn.ModuleList):
    layer = layer[-1]

handle = layer.register_forward_hook(forward_hook)


model.eval()
t = torch.FloatTensor([0.0]).cuda()

rel_fld_sample = np.zeros(len(ts))
abs_fld_sample = np.zeros(len(ts))
dl_0 = np.zeros(len(ts))
dl_1 = np.zeros(len(ts))

t_absolute = []
t_relative = []

columns = ['i', 't', 'rel. alig', 'abs. alig', 'rel. mean', 'rel. std', 'abs. mean', 'abs. std']

for i, t_value in enumerate(ts):


    if t_value < 0.0 or t_value > 1.0:
        continue

    t.data.fill_(t_value)

    utils.update_bn(loaders['train'], model, t=t)

    if rel_proj.center is True or rel_proj.standardize is True:
        norm_layer = utils.train_norm_layer(model, t_value, loaders['train'], args.layer_name, norm='bn')
        rel_proj.update_stats(norm_layer.running_mean, torch.sqrt(norm_layer.running_var + 1e-5))

    anchors = []
    for input, target in loaders['anchors']:
        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        _ = model(input, t=t)
        anchors.append(latent.clone().detach())

    anchors = torch.cat(anchors, dim=0)
    
    absolute = []
    relative = []


    for input, target in loaders['test']:

        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        _ = model(input, t=t)

        absolute.append(latent.clone().detach())
        relative.append(rel_proj.project(latent.clone().detach(), anchors))  
        
    t_absolute.append(torch.cat(absolute, dim=0).unsqueeze(-1))
    t_relative.append(torch.cat(relative, dim=0).unsqueeze(-1))

handle.remove()

absolute_alignment, abs_within, abs_between = measures.sample_align(torch.cat(t_absolute, dim=-1))
relative_alignment, rel_within, rel_between = measures.sample_align(torch.cat(t_relative, dim=-1))
    

path = os.path.join(args.dir, f'within_between_{args.sampling_method}_{args.projection}.npz')
print(f'Saving results to {path}')
np.savez(
    path,
    abs_latent=torch.cat(t_absolute, dim=-1).cpu().numpy(),
    rel_latent=torch.cat(t_relative, dim=-1).cpu().numpy(),
    target=target.cpu().numpy(),
    absolute_alignment=absolute_alignment,
    relative_alignment=relative_alignment,
    absolute_within=abs_within.cpu().numpy(),
    absolute_between=abs_between.cpu().numpy(),
    relative_within=rel_within.cpu().numpy(),
    relative_between=rel_between.cpu().numpy()
)

