# It's All Relative: Relative Uncertainty in Latent Spaces using Relative Representations

This is the official code base of [It's All Relative: Relative Uncertainty in Latent Spaces using Relative Representations](https://openreview.net/group?id=NeurIPS.cc/2024/Workshop/UniReps/Authors&referrer=%5BHomepage%5D(%2F)).
It is based on [this github repository](https://github.com/timgaripov/dnn-mode-connectivity) from the authors of [Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs). In this work, we show that ensembles are confounded by reparameterization. By transforming the embeddings to a space of relative proximity, we show that uncertainty in the latent space decreases.


## Getting Started
In this example, we use the VGG16 as an example. To train an endpoint, call

```bash
python3 python3 train.py --dir=<path/to/logging/dir> --dataset=CIFAR100 --data_path=./data --model=VGG16 --epochs=200 --lr=0.05 --wd=5e-4 --use_test --transform=VGG --seed= --seed=>SEED<
```

In the paper, we train 11 different seeds. The ensemble of all 11 seeds are used as baseline. For the curve fitting, we fit 3 curves from seed 0 -> 1, 1 -> 2, and 0 -> 2. Fitting a curve can be achieved by calling

```bash
python3 train.py --dir=<path/to/logging/dir> --dataset=CIFAR100 --use_test --transform=VGG --data_path=./data --model=VGG16BN --curve=Bezier --num_bends=3  --init_start=<START_SEED>/checkpoint-200.pt --init_end=<END_SEED>/checkpoint-200.pt --fix_start --fix_end --epochs=200 --lr=0.015 --wd=5e-4
```

We recommend checking out the original repository for additional information on the curve fitting experiments.

## Relative Representations and Uncertainty

The curve models finds the path with minimal increase in loss between two modes. We extract the embeddings along the curve and compare the alignment of these ensembles, before and after transforming the embeddings to a space of relative proximity. See [Relative Representations](https://arxiv.org/abs/2209.15430) for details.
For example, to compare the cumulative alignment for the aboslute space and the Relative Representations using a cosine similarity measure for the last layer before the classification head, run

```bash
python3 eval_relative_curve_cumulative.py --dir=<path/to/logging/dir> --dataset=CIFAR100 --use_test --transform=VGG --data_path=./data --model=VGG16 --curve=<path/to/curve> --seed_from_to=<START_SEED>-<END_SEED> --num_bends=3 --ckpt=checkpoint-200.pt --sampling_method=linear --layer_name=fc3 --num_anchors=512 --batch_size=512 --n_batches=3 --num_points=21 --projection=cosine --center
```
