# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

base_lr = 1e-3
min_lr = 0
warmup_epochs = 40
epochs = n_epochs


def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs
        lr = base_lr * epoch / warmup_epochs 
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
