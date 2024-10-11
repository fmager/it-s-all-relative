import numpy as np
import os
import torch
import torch.nn.functional as F
import tabulate
import torch.optim as optim
import curves
import measures
from torch.nn import ModuleList



def l2_regularizer(weight_decay):
    def regularizer(model):
        l2 = 0.0
        for p in model.parameters():
            l2 += torch.sqrt(torch.sum(p ** 2))
        return 0.5 * weight_decay * l2
    return regularizer


def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, '%s-%d.pt' % (name, epoch))
    torch.save(state, filepath)


def train(train_loader, model, optimizer, criterion, regularizer=None, lr_schedule=None):
    loss_sum = 0.0
    correct = 0.0

    num_iters = len(train_loader)
    model.train()
    for iter, (input, target) in enumerate(train_loader):
        if lr_schedule is not None:
            lr = lr_schedule(iter / num_iters)
            adjust_learning_rate(optimizer, lr)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        if regularizer is not None:
            loss += regularizer(model)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test(test_loader, model, criterion, regularizer=None, **kwargs):
    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    preds = []

    model.eval()

    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        nll = criterion(output, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        'nll': nll_sum / len(test_loader.dataset),
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
        'predictions': np.vstack(preds),
    }

def test_relative(test_loader, model, criterion, layer_name: str, rel_proj, regularizer=None, **kwargs):

    outputs_abs = []
    outputs_rel = []

    latent = torch.Tensor()

    def forward_hook(module, input, output):
        nonlocal latent
        latent = input[0].clone().detach()
    
    layer = getattr(model.net, layer_name)
    handle = layer.register_forward_hook(forward_hook)

    loss_sum = 0.0
    nll_sum = 0.0
    correct = 0.0
    preds = []

    model.eval()

    for input, target in test_loader['test']:

        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        out = model(input, **kwargs)

        outputs_abs.append(latent.clone().detach().cpu())

        probs = F.softmax(out, dim=1)
        preds.append(probs.cpu().data.numpy())
        nll = criterion(out, target)
        loss = nll.clone()
        if regularizer is not None:
            loss += regularizer(model)

        nll_sum += nll.item() * input.size(0)
        loss_sum += loss.item() * input.size(0)
        pred = out.data.argmax(1, keepdim=True)
        correct += pred.eq(target.data.view_as(pred)).sum().item()

    abs_latent = torch.cat(outputs_abs, dim=0)
    
    for input, target in test_loader['anchors']:
        input = input.cuda(non_blocking=True).requires_grad_(False)
        target = target.cuda(non_blocking=True).requires_grad_(False)

        _ = model(input, **kwargs)
        anchors = latent.clone().detach().cpu()
    
        rel_latent = rel_proj.project(abs_latent, anchors)
        outputs_rel.append(rel_latent)
    
    handle.remove()

    return {
        'nll': nll_sum / len(test_loader['test'].dataset),
        'loss': loss_sum / len(test_loader['test'].dataset),
        'accuracy': correct * 100.0 / len(test_loader['test'].dataset),
        'predictions': np.vstack(preds),
        'output_abs': abs_latent,
        'output_rel': torch.cat(outputs_rel, dim=1),
    }

def get_cumulative_alignment(loaders, model, layer_name: str, rel_proj, ts, N_batches):
    

    weights_0 = model.weights(torch.FloatTensor([0.0]).cuda())
    weights_1 = model.weights(torch.FloatTensor([1.0]).cuda())

    latent = torch.Tensor()
    
    def forward_hook(module, input, output):
        nonlocal latent
        latent = input[0].clone().detach()
        while latent.dim() > 2:
            latent = latent.mean(dim=-1)

    layer = getattr(model.net, layer_name)
    while isinstance(layer, ModuleList):
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

        weights = model.weights(t.data.fill_(t_value))

        dl_0[i] = np.sqrt(np.sum(np.square(weights - weights_0)))
        dl_1[i] = np.sqrt(np.sum(np.square(weights - weights_1)))


        if t_value < 0.0 or t_value > 1.0:
            continue

        t.data.fill_(t_value)

        update_bn(loaders['train'], model, t=t)

        if rel_proj.center is True or rel_proj.standardize is True:
            norm_layer = train_norm_layer(model, t_value, loaders['train'], layer_name, norm='bn')
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

        for _ in range(N_batches):

            input, target = next(iter(loaders['test']))

            input = input.cuda(non_blocking=True).requires_grad_(False)
            target = target.cuda(non_blocking=True).requires_grad_(False)

            _ = model(input, t=t)

            absolute.append(latent.clone().detach())
            relative.append(rel_proj.project(latent.clone().detach(), anchors))  
        
        t_absolute.append(torch.cat(absolute, dim=0).unsqueeze(-1))
        t_relative.append(torch.cat(relative, dim=0).unsqueeze(-1))


        abs_fld_sample[i], _, _ = measures.sample_align(torch.cat(t_absolute, dim=-1))
        rel_fld_sample[i], _, _ = measures.sample_align(torch.cat(t_relative, dim=-1))

        abs_mean = torch.cat(t_absolute, dim=-1).mean(dim=-1).mean()
        abs_std = torch.cat(t_absolute, dim=-1).mean(dim=-1).std(dim=0).mean().item()

        rel_mean = torch.cat(t_relative, dim=-1).mean()
        rel_std = torch.cat(t_relative, dim=-1).mean(dim=-1).std(dim=0).mean().item()

        values = [i, t_value, rel_fld_sample[i], abs_fld_sample[i], rel_mean, rel_std, abs_mean, abs_std]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
        
    handle.remove()

    return {
        'ts': ts,
        'alignment_absolute': abs_fld_sample,
        'alignment_relative': rel_fld_sample,
        'dl_0': dl_0,
        'dl_1': dl_1,
    }
     

def get_alignment_in_window(loaders, model, layer_name: str, rel_proj, ts, dt, N_batches, N_samples_in_window):
    

    weights_0 = model.weights(torch.FloatTensor([0.0]).cuda())
    weights_1 = model.weights(torch.FloatTensor([1.0]).cuda())

    latent = torch.Tensor()
    
    def forward_hook(module, input, output):
        nonlocal latent
        latent = input[0].clone().detach()
        while latent.dim() > 2:
            latent = latent.mean(dim=-1)

    layer = getattr(model.net, layer_name)
    while isinstance(layer, ModuleList):
        layer = layer[-1]

    handle = layer.register_forward_hook(forward_hook)


    model.eval()
    t = torch.FloatTensor([0.0]).cuda()

    rel_align_sample = np.zeros(len(ts))
    abs_align_sample = np.zeros(len(ts))
    dl_0 = np.zeros(len(ts))
    dl_1 = np.zeros(len(ts))

    columns = ['i', 't', 'rel. alig', 'abs. alig', 'rel. mean', 'rel. std', 'abs. mean', 'abs. std']

    for i, t_mid in enumerate(ts):

        weights = model.weights(t.data.fill_(t_mid))

        dl_0[i] = np.sqrt(np.sum(np.square(weights - weights_0)))
        dl_1[i] = np.sqrt(np.sum(np.square(weights - weights_1)))

        t_absolute = []
        t_relative = []


        for _, t_value in enumerate(np.linspace(t_mid - dt/2, t_mid + dt/2, N_samples_in_window)):

            if t_value < 0.0 or t_value > 1.0:
                # Skip this iteration
                continue

            if rel_proj.center is True or rel_proj.standardize is True:
                norm_layer = train_norm_layer(model, t_mid, loaders['train'], layer_name, norm='bn')
                rel_proj.update_stats(norm_layer.running_mean, torch.sqrt(norm_layer.running_var + 1e-5))

            
            t.data.fill_(t_value)

            update_bn(loaders['train'], model, t=t)

            anchors = []
            for input, target in loaders['anchors']:
                input = input.cuda(non_blocking=True).requires_grad_(False)
                target = target.cuda(non_blocking=True).requires_grad_(False)

                _ = model(input, t=t)
                anchors.append(latent.clone().detach())

            anchors = torch.cat(anchors, dim=0)
            
            absolute = []
            relative = []

            for _ in range(N_batches):

                input, target = next(iter(loaders['test']))

                input = input.cuda(non_blocking=True).requires_grad_(False)
                target = target.cuda(non_blocking=True).requires_grad_(False)

                _ = model(input, t=t)

                absolute.append(latent.clone().detach())
                relative.append(rel_proj.project(latent.clone().detach(), anchors))  
            
            t_absolute.append(torch.cat(absolute, dim=0).unsqueeze(-1))
            t_relative.append(torch.cat(relative, dim=0).unsqueeze(-1))

        t_absolute = torch.cat(t_absolute, dim=-1)
        t_relative = torch.cat(t_relative, dim=-1)

        abs_align_sample[i], _, _ = measures.sample_aligngn(t_absolute)
        rel_align_sample[i], _, _ = measures.sample_aligngn(t_relative)

        abs_mean = t_absolute.mean()
        abs_std = t_absolute.permute(0,2,1).flatten(0,1).std(dim=0).mean().item()

        rel_mean = t_relative.mean()
        rel_std = t_relative.permute(0,2,1).flatten(0,1).std(dim=0).mean().item()

        values = [i, t_mid, rel_align_sample[i], abs_align_sample[i], rel_mean, rel_std, abs_mean, abs_std]
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='10.4f')

        if i % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)
    
    handle.remove()

    return {
        'ts': ts,
        'alignment_absolute': abs_align_sample,
        'alignment_relative': rel_align_sample,
        'dl_0': dl_0,
        'dl_1': dl_1,
    }

def train_norm_layer(model, t, loader, layer_name, norm: str = 'bn', eps=1e-4):

    latent = torch.Tensor()
    
    def forward_hook(module, input, output):
        nonlocal latent
        latent = input[0].clone().detach()
        while latent.dim() > 2:
            latent = latent.mean(dim=-1)

    if hasattr(model, 'net'):
        layer = getattr(model.net, layer_name)
    else:
        layer = getattr(model, layer_name)
    while isinstance(layer, (torch.nn.ModuleList, torch.nn.Sequential)):
        layer = layer[-1]

    handle = layer.register_forward_hook(forward_hook)

    # run a single forward pass to get the feature dimension
    input, _ = next(iter(loader))
    input = input.cuda(non_blocking=True).requires_grad_(False)[0:1]
    with torch.no_grad():
        _ = model(input)

    feature_dim = latent.size(-1)

    if norm == 'bn':
        norm_layer = torch.nn.BatchNorm1d(feature_dim, affine=False)
    elif norm == 'ln':
        norm_layer = torch.nn.LayerNorm(feature_dim, elementwise_affine=False)
    elif norm == 'in':
        norm_layer = torch.nn.InstanceNorm1d(feature_dim, affine=False)

    norm_layer.train()
    norm_layer = norm_layer.cuda()

    prev_running_mean = torch.zeros(feature_dim).cuda()
    prev_running_var = torch.zeros(feature_dim).cuda()
    t = torch.FloatTensor([t]).cuda() if t is not None else None
    for i, (input, _) in enumerate(loader):
        input = input.cuda(non_blocking=True).requires_grad_(False)

        with torch.no_grad():
            if t == None:
                _ = model(input)
            else:
                _ = model(input, t=t)
            _ = norm_layer(latent)


        mean_diff = prev_running_mean - norm_layer.running_mean
        var_diff = prev_running_var - norm_layer.running_var

        if torch.norm(mean_diff) < eps and torch.norm(var_diff) < eps:
            break
        else:
            prev_running_mean = norm_layer.running_mean.clone()
            prev_running_var = norm_layer.running_var.clone()

    handle.remove()
    norm_layer.eval()

    return norm_layer


def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda(non_blocking=True)
        output = model(input, **kwargs)
        probs = F.softmax(output, dim=1)
        preds.append(probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)


def isbatchnorm(module):
    return issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm) or \
           issubclass(module.__class__, curves._BatchNorm)


def _check_bn(module, flag):
    if isbatchnorm(module):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isbatchnorm(module):
        module.reset_running_stats()


def _get_momenta(module, momenta):
    if isbatchnorm(module):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isbatchnorm(module):
        module.momentum = momenta[module]


def update_bn(loader, model, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    num_samples = 0

    for input, _ in loader:
        input = input.cuda(non_blocking=True)
        batch_size = input.data.size(0)

        momentum = batch_size / (num_samples + batch_size)
        for module in momenta.keys():
            module.momentum = momentum

        model(input, **kwargs)
        num_samples += batch_size

    model.apply(lambda module: _set_momenta(module, momenta))


def highlight_anchors(fig):
    for trace in fig.data:
        if trace.name == 'anchor':
            trace.marker.symbol = 'star'  # Set the marker type to 'star' for 'anchor'
            trace.marker.size = 5
            trace.marker.color = 'black' 
            trace.marker.line.width = .05  # Set the marker line width to 2 
        else:
            trace.marker.symbol = 'circle'
            trace.marker.line.width= .05


    for frame in fig.frames:
      for trace in frame.data:
        if trace.name == 'anchor':
            trace.marker.symbol = 'star'  # Set the marker type to 'star' for 'anchor'
            trace.marker.size = 5
            trace.marker.color = 'black'
            trace.marker.line.width = .05  # Set the marker line width to 2
        else:
            trace.marker.symbol = 'circle'
            trace.marker.line.width= .05
            
    return fig

class RelProjector(object):
    import torch.cuda

    def __init__(self, proj_fn, center: bool = False, standardize: bool = False):
        self.proj_fn = proj_fn
        self.center = center
        self.standardize = standardize
        self.mean = torch.Tensor([0.0])
        self.std = torch.Tensor([1.0])
        if torch.cuda.is_available():
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()

    def update_stats(self, mean, std):
        if self.center is True:
            self.mean = mean
        if self.standardize is True:
            self.std = std

    def project(self, input, anchors):
        input -= self.mean
        anchors -= self.mean
        input /= self.std
        anchors /= self.std
        return self.proj_fn(input, anchors)

def cosine_projection(x, y):
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)
    return torch.einsum('ik,jk->ij', x, y)

def basis_norm_projection(x, y):
    return torch.einsum('ik,jk->ij', x, y) / (y**2).sum(dim=1)

def euclidean_projection(x, y):
    return dist_projection(x, y, 2)

def dist_projection(x, y, p):
    # y = y.permute(2, 0, 1)
    # x = x.permute(2, 0, 1)
    return torch.cdist(x, y, p=p) #.permute(1, 2, 0)