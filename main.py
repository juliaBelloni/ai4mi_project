#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import warnings
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

from sympy import factor
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images,
                   estimate_flops,
                   save_flops_count)

from losses import (CrossEntropy, Dice, DiceCE, Balance)
import json
import random

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    
def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)    
    
def make_scheduler(args: argparse.Namespace, optimizer):
    if args.scheduler == "none":
        return None

    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, 
                                               gamma=args.scheduler_gamma)

    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode="max", 
                factor=args.scheduler_gamma, patience=args.scheduler_patience)

    if args.scheduler == "cosine":
        t_max = args.scheduler_t_max if args.scheduler_t_max > 0 else args.epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    raise ValueError(f"Invalid scheduler {args.scheduler}")
    
def compute_invfreq_weights(loader: DataLoader, K: int, idk: list[int], alpha: float = 1.0) -> list[float]:
    """These are the inverse frequency weights as given by
    Sugino et al. (2021) in eq. 3.

        w_k = (N / (K * n_k)) ** alpha
    where 
        N = total pixel count, K = number of classes,
        n_k = pixel count of class k.

    alpha is the same power parameter), but with an added
    constant factor (counts.sum() / K) ** alpha, and the 
    requirement alpha > 0, so it can be interpreted that: 

    w = 1 for a class with an average number of pixels.
    w > 1 means the class is rarer (organs).
    w < 1 means the class is more common (background). 
    
    The constant factor doesn't matter for the loss, as the
    cross-entropy we use is normalized.

    Sugino et al. (2021): ./papers/healthcare-09-00938.pdf"""

    assert alpha > 0, f"alpha must be > 0 to keep rarer classes at w > 1, got {alpha}"

    counts = torch.zeros(K)
    for data in loader:
        counts += data["gts"].sum(dim=(0, 2, 3))
    weights = (counts.sum() / (K * (counts + 1e-8))) ** alpha
    return [weights[k].item() for k in idk]

def save_config(args: argparse.Namespace) -> None:
    args.dest.mkdir(parents=True, exist_ok=True)

    payload = {key: str(value) if isinstance(value,Path) else value
        for key, value in vars(args).items()}

    with open(args.dest / "config.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    
def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]

def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler + adding mps support
    gpu: bool = args.gpu and torch.cuda.is_available()
    mps: bool = args.mps and torch.backends.mps.is_available()
    device = torch.device("cuda") if gpu else torch.device("mps") if mps else torch.device("cpu")

    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    in_channels = 2 * args.context_slices + 1
    net = datasets_params[args.dataset]['net'](in_channels, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = args.lr # default is  lr = 0.0005
        
    if args.opt == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Invalid optimizer {args.opt}")

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    generator = None
    worker_init_fn = None
    if args.deterministic:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        worker_init_fn = seed_worker

    train_set = SliceDataset(
        'train',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
        context_slices=args.context_slices)
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=5,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              generator=generator)

    val_set = SliceDataset(
        'val',
        root_dir,
        img_transform=img_transform,
        gt_transform=partial(gt_transform, K),
        debug=args.debug,
        context_slices=args.context_slices)
    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=5,
                            shuffle=False,
                            worker_init_fn=worker_init_fn,
                            generator=generator)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    save_config(args)
    scheduler = make_scheduler(args, optimizer)
    epochs_without_improvement = 0

    if args.mode == "full":
        idk = list(range(K))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        idk = [0, 1, 3, 4]  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    ce_weights: list[float] | None = None
    if args.ce_weights == "invfreq":
        ce_weights = compute_invfreq_weights(train_loader, K, idk, alpha=args.ce_weights_alpha)
        print(f">> Computed invfreq CE weights: {ce_weights}")
    elif args.ce_weights is not None:
        ce_weights = [float(w) for w in args.ce_weights.split(",")]

    if ce_weights is not None and args.loss_fn not in ("ce", "dicece"):
        print(f">> Warning: --ce_weights has no effect for --loss_fn {args.loss_fn}")

    if args.loss_fn == "ce":
        loss_fn = CrossEntropy(idk=idk, weight=ce_weights)
    elif args.loss_fn == "dice":
        loss_fn = Dice(idk=idk)
    elif args.loss_fn == "dicece":
        loss_fn = DiceCE(idk=idk, lambda_=args.dicece_lambda, weight=ce_weights)
    elif args.loss_fn == "balance":
        loss_fn = Balance(idk=idk, alpha=args.balance_alpha, t=args.balance_t, normalized=args.balance_normalized)
    else:
        raise ValueError(f"Invalid loss function {args.loss_fn}")

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0
    epochs_ran: int = 0

    for e in range(args.epochs):
        epochs_ran = e + 1
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    if isinstance(loss_fn, Balance) and m == 'train':
                        loss_fn.step(pred_probs, gt)

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = log_dice_val[e, :, 1:].mean().item()
        improved = current_dice > best_dice + args.early_stopping_min_delta

        if improved:
            message = f">>> Improved dice at epoch {e}: {best_dice:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            epochs_without_improvement = 0

            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(current_dice)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            print(f">>> Learning rate after epoch {e}: {current_lr:.3e}")

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            print(f">>> Early stopping after {epochs_without_improvement} epochs without improvement")
            break

    if args.count_flops: # pytorch FLOPs are estimates and mainly cover conv/matmul operations
        flops_count = estimate_flops(net, optimizer, loss_fn, train_loader, val_loader, device, epochs_ran)
        save_flops_count(args.dest, flops_count)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mps', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help="Keep only a fraction (10 samples) of the datasets, "
                             "to test the logics around epochs and logging easily.")
    parser.add_argument('--loss_fn', choices=["ce", "dice", "dicece", "balance"], default="ce", help="Loss function used during training.")
    parser.add_argument('--dicece_lambda', default=0.5, type=float,
                        help="Weight of the dice term when --loss_fn is dicece, in [0, 1]: "
                             "L = (1 - dicece_lambda) * L_CE + dicece_lambda * L_DICE.")
    parser.add_argument('--ce_weights', default=None, type=str,
                        help="Per-class weights for the CE term (applies to --loss_fn ce and dicece). Either "
                             "comma-separated floats matching the supervised classes, e.g. '0.5,1,1,2,3', or "
                             "'invfreq' to compute inverse-frequency weights from the training set.")
    parser.add_argument('--ce_weights_alpha', default=1.0, type=float,
                        help="Power parameter for --ce_weights invfreq (Sugino et al. 2021, eq. 3)")
    parser.add_argument('--balance_alpha', default=0.5, type=float,
                        help="Weight of the Intra-CBL term when --loss_fn is balance, in [0, 1]: "
                             "BL = (1 - alpha) * Inter-CBL + alpha * Intra-CBL (Xu et al. 2025, eq. 13).")
    parser.add_argument('--balance_t', default=0.9, type=float,
                        help="Threshold t for --loss_fn balance (Xu et al. 2025): splits easy/hard pixels "
                             "for Intra-CBL and is also used in the Inter-CBL convergence check.")
    parser.add_argument('--balance_normalized', action='store_true',
                        help="Normalize Inter-CBL and Intra-CBL so it is a actual average, so that "
                             "the weights sum to one (not part of Xu et al. 2025, but could help).")
    parser.add_argument('--opt', choices=["adam", "adamw"], default="adam", help="Optimizer used during training.")
    parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate used during training.")
    parser.add_argument( '--context_slices', default=0, type=int, help="Number of neighboring slices before and after the current slice. 0 keeps 2D behavior.")
    parser.add_argument('--scheduler', choices=["none", "step", "plateau", "cosine"], default="none", help="Learning rate scheduler.")
    parser.add_argument('--scheduler_step_size', default=10, type=int, help="Step size for --scheduler step.")
    parser.add_argument('--scheduler_gamma', default=0.1, type=float,help="LR decay factor for step/plateau schedulers.")
    parser.add_argument('--scheduler_patience', default=5, type=int, help="Patience for --scheduler plateau.")
    parser.add_argument('--scheduler_t_max', default=0, type=int, help="T_max for --scheduler cosine. 0 means use --epochs.")
    parser.add_argument('--early_stopping_patience', default=0, type=int, help="Stop after this many epochs without validation Dice improvement. 0 disables early stopping.")
    parser.add_argument( '--early_stopping_min_delta', default=0.0, type=float,help="Minimum validation Dice improvement needed to reset early stopping.")
    parser.add_argument('--deterministic', action='store_true', help="Enable deterministic PyTorch/CUDA behavior and seeded DataLoader shuffling.")
    parser.add_argument('--seed', default=0, type=int, help="Seed used when --deterministic is set.")
    parser.add_argument('--count_flops', action='store_true', help="Estimate train/validation FLOPs over 10 batches.")

    args = parser.parse_args()

    if args.deterministic:
        set_deterministic(args.seed)
    
    pprint(args)

    runTraining(args)


if __name__ == '__main__':
    main()
