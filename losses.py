#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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


import torch
from torch import einsum, Tensor

from utils import simplex, sset


class CrossEntropy():
    """The standard cross-entropy, with optional class weighting."""
    def __init__(self, idk: list[int], weight: list[float] | None = None):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = idk
        # Class weight for the CE term, in the same order as idk. Defaults to ones.
        weight = weight if weight is not None else [1.] * len(idk)
        assert len(weight) == len(idk)
        self.weight = Tensor(weight)
        print(f"Initialized {self.__class__.__name__} with {idk=}, {weight=}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()
        weight = self.weight.to(mask.device)

        loss = - einsum("bkwh,bkwh,k->", mask, log_p, weight)
        loss /= einsum("bkwh,k->", mask, weight) + 1e-10

        return loss

class Dice():
    """Standard DICE loss."""
    def __init__(self, idk: list[int], smooth: float = 1e-8):
        self.idk = idk
        self.smooth = smooth
        print(f"Initialized {self.__class__.__name__} with {idk=}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred_softmax = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        intersection = einsum("bkwh,bkwh->bk", pred_softmax, mask)
        union = einsum("bkwh->bk", pred_softmax) + einsum("bkwh->bk", mask)
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_score.mean()


class DiceCE():
    """The 'standard' loss function given during the lecture,
    and the default in nnU-Net, combining cross-entropy and
    dice loss, so that L = (1 - lambda) * L_CE + lambda * L_DICE."""

    def __init__(self, idk: list[int], lambda_: float, smooth: float = 1e-8, weight: list[float] | None = None):
        self.idk = idk
        self.lambda_ = lambda_
        self.ce = CrossEntropy(idk=idk, weight=weight)
        self.dice = Dice(idk=idk, smooth=smooth)
        print(f"Initialized {self.__class__.__name__} with {idk=}, lambda = {lambda_}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        ce_loss = self.ce(pred_softmax, weak_target)
        dice_loss = self.dice(pred_softmax, weak_target)

        return (1 - self.lambda_) * ce_loss + self.lambda_ * dice_loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)


class InterCBL():
    """Inter-class Balance loss (Xu et al. 2025, eq. 11): solves the
    class imbalance between background and foreground only. So unlike 
    weighted CrossEntropy, it does not fix imbalances between individual
    foreground classes (organs).

    We have made one (optional) change to the proposed loss function,
    which is an added `normalizer` factor, which makes sure all weights
    sum to one for a batch, making this an actual weighted average over
    all pixels in the image. 

    L_InterCBL = sum_b_in_batch [(
        - sum_i_in_fg_b      log(p_i^{y_i}) / |fg_b|
        - sum_i_in_bg_hard_b log(p_i^{y_i}) / |fg_b|
        - sum_i_in_bg_easy_b log(p_i^{y_i}) / N) 
    ) / normalizer] / |batch|

    where, for image b in the batch
        y_i = the true class of pixel i
        fg_b = pixels of image b whose true class isn't background
        bg_hard_b = the |fg_b| hardest (lowest p_i^{y_i}) background pixels of image b
        bg_easy_b = the remaining background pixels of image b, bg_b excluding bg_hard_b
        N = |fg| + |bg| so the total number of pixels in the image.
        normalizer = 1 if normalized=False, else 2 + |bg_easy_b| / N

    "we identify the top f pixels with the highest losses in the
    majority class and assign them weights equal to those of the
    minority class pixels" 
    
    Easy background (majority) -> normal 1/N weight
    Hard background (majority) -> larger 1/|fg| weight
    All foreground  (minority) -> larger 1/|fg| weight

    Xu et al. (2025): ./papers/09125-XuF.pdf, ./papers/losses.md"""

    def __init__(self, idk: list[int], normalized: bool = False):
        assert idk[0] == 0, "InterCBL assumes idk[0] is the background class"
        self.idk = idk
        self.normalized = normalized
        print(f"Initialized {self.__class__.__name__} with {idk=}, normalized={normalized}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...] # (B, |idk|, W, H)
        mask = weak_target[:, self.idk, ...].float() # (B, |idk|, W, H)
        p_corr = (pred * mask).sum(dim=1) # prob of the true class, per pixel: (B, W, H)
        bg = mask[:, 0, ...].bool() # (B, W, H), true where pixel's class is background
        fg = ~bg # (B, W, H)

        p_flat = p_corr.flatten(1) # (B, W*H)
        bg_flat = bg.flatten(1) # (B, W*H)
        fg_flat = fg.flatten(1) # (B, W*H)
        log_p_flat = p_flat.clamp_min(1e-10).log() # (B, W*H)

        f = fg_flat.sum(dim=1) # (B,) number of foreground pixels per image, i.e., |fg|
        fg_loss = -(log_p_flat * fg_flat).sum(dim=1) # (B,) loss summed over all foreground pixels
        bg_loss = -(log_p_flat * bg_flat).sum(dim=1) # (B,) loss summed over all background pixels

        # Here we pick the top-f hardest background pixels, i.e., 
        # the background pixels with the bottom-f probability. But 
        # since f = |fg| is different for every image in the batch,
        # we do a trick where we rank all pixels and discard options
        # that are not background. The alternative would be an un-
        # vectorized loop. 
        max_f = int(f.max().item()) 
        hard_loss = torch.zeros_like(fg_loss)
        if max_f > 0:
            # We set all foreground pixel probabilities to 1.0, so even
            # it they end up in the bottom-f_max, they are the highest.
            bg_for_topk = p_flat.detach().masked_fill(fg_flat, 1.0) # (B, P)

            # We take the bottom-f_max, so for some images in the batch 
            # that's also some foreground pixels (with probability 1.0)
            _, hardest_idx = torch.topk(bg_for_topk, max_f, largest=False, dim=1) # (B, max_f)

            # Just a list with all ranks: 0, 1, 2 ... f_max.
            rank = torch.arange(max_f, device=p_corr.device).unsqueeze(0) # (1, max_f)

            # A mask with true for all indices that are of background pixels.
            is_bg = bg_flat.gather(1, hardest_idx) # (B, max_f)

            # We need both checks in the rare case that |fg| > |bg|
            valid = (rank < f.unsqueeze(1)) & is_bg # (B, max_f)

            # We know have a mask for all values that we should count
            # in the hard loss, so gather, mask them and sum them
            gathered = log_p_flat.gather(1, hardest_idx) # (B, max_f)
            hard_loss_masked = -torch.where(valid, gathered, torch.zeros_like(gathered)) # (B, max_f)
            hard_loss = hard_loss_masked.sum(dim=1) # (B,)

        # Since we already have the loss over the whole background, 
        # and we also have the loss over the hard packground pixels,
        # finding the easy ones is trivial.
        easy_loss = bg_loss - hard_loss # (B,)

        N = p_flat.shape[1] # pixel count per image
        loss_per_image = fg_loss / (f + 1e-10) + hard_loss / (f + 1e-10) + easy_loss / N # (B,)

        if self.normalized:
            bg_count = bg_flat.sum(dim=1) # (B,), |bg_b|
            bg_hard_count = torch.minimum(f, bg_count) # (B,), |bg_hard_b|
            bg_easy_count = bg_count - bg_hard_count # (B,), |bg_easy_b|
            normalizer = 2 + bg_easy_count / N # (B,)
            loss_per_image = loss_per_image / normalizer

        return loss_per_image.mean()


class IntraCBL():
    """Intra-class Balance loss as in Xu et al. 2025, eq. 12: an intra-class
    imbalance fix. Makes sure harder pixels of the same class are given 
    more weight.

    We have made one (optional) change to the proposed loss function,
    which is an added `normalizer` factor, which makes sure all weights
    sum to one for a batch, making this an actual weighted average over
    all pixels.

    L_IntraCBL = sum_b_in_batch sum_c_in_classes [
        - sum_i_in_hard_cb log(p_i^c) / |hard_cb|
        - sum_i_in_easy_cb log(p_i^c) / |easy_cb|
    ] / (normalizer * |batch|)

    where, for image b in the batch
        classes = all classes, i.e. idk
        hard_cb = pixels of image b whose true class is c, with p_i^c <= t
        easy_cb = pixels of image b whose true class is c, with p_i^c > t
        normalizer = 1 if normalized=False, else 2 * |classes|

    Xu et al. (2025): ./papers/09125-XuF.pdf, ./papers/losses.md"""

    def __init__(self, idk: list[int], t: float = 0.9, normalized: bool = False):
        self.idk = idk
        self.t = t
        self.normalized = normalized
        print(f"Initialized {self.__class__.__name__} with {idk=}, t={t}, normalized={normalized}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...] # (B, |idk|, W, H)
        mask = weak_target[:, self.idk, ...].float() # (B, |idk|, W, H) 
        log_p = (pred + 1e-10).log() # (B, |idk|, W, H)

        easy = mask * (pred > self.t).float() # (B, |idk|, W, H)
        hard = mask * (pred <= self.t).float() # (B, |idk|, W, H)

        easy_count = easy.sum(dim=(2, 3)) # (B, |idk|)
        hard_count = hard.sum(dim=(2, 3)) # (B, |idk|)

        easy_loss = (easy * log_p).sum(dim=(2, 3)) / (easy_count + 1e-10) # (B, |idk|)
        hard_loss = (hard * log_p).sum(dim=(2, 3)) / (hard_count + 1e-10) # (B, |idk|)

        per_image = -(easy_loss + hard_loss).sum(dim=1) # (B,)

        if self.normalized:
            per_image = per_image / (2 * len(self.idk))

        return per_image.mean() # average over the batch


class Balance():
    """Balance loss as in Xu et al. 2025, eq. 13.
    
    L_Balance = (1 - alpha) * L_InterCBL + alpha * L_IntraCBL. 

    where alpha = hyperparam in [0, 1] to specify intra/inter dominance.
        
    In the first few epochs, L_InterCBL is left out, as the background
    doesn't yet dominate the loss. I.e., the probabilites for a pixel
    being some class will all be rather similar. Therefore, ranking
    pixels by probability to find the "hardest" ones would really just
    be noise. Whereas for L_IntraCBL, we don't do any ranking or top-k,
    but we just do a threshold. A threshold also makes sense whenever
    the model already (by chance) seems to find some pixels easier.

    L_Balance_at_start = L_IntraCBL
    
    Call .step() on every training batch (so not on validation batches) 
    to check the paper's convergence criterion. Once it's converged, it
    will start using the full L_Balance.

    In .step(), it has converged once for more than half the images in 
    a batch the hardest foreground pixel's p_corr > t, AND the best of 
    the f hardest background pixels selected by InterCBL has p_corr > t. 

    Xu et al. (2025): ./papers/09125-XuF.pdf, ./papers/losses.md"""

    def __init__(self, idk: list[int], alpha: float = 0.5, t: float = 0.9, normalized: bool = False):
        self.idk = idk
        self.alpha = alpha
        self.t = t
        self.normalized = normalized
        self.inter = InterCBL(idk, normalized=normalized)
        self.intra = IntraCBL(idk, t=t, normalized=normalized)
        self._converged = False
        print(f"Initialized {self.__class__.__name__} with {idk=}, alpha={alpha}, t={t}, normalized={normalized}")

    @torch.no_grad()
    def step(self, pred_softmax: Tensor, weak_target: Tensor) -> None:
        """From Xu et al (2025):

        "Assume the number of foreground pixels is f ..."

        "Specifically, we consider the network to have converged
        to an appropriate level when more than half of the images
        within the same batch have their f-th best probability value
        in the foreground and background greater than t"

        This quote feels a bit ambiguous. They say the f-th best 
        probability in the background must be greater than t, but 
        I assume by this they mean: the probability of the f-th 
        hardest background pixel must be geater than t."""

        if self._converged:
            return

        pred = pred_softmax[:, self.idk, ...] # (B, |idk|, W, H)
        mask = weak_target[:, self.idk, ...].float() # (B, |idk|, W, H)
        p_corr = (pred * mask).sum(dim=1) # (B, W, H) prob of true correct class
        bg = mask[:, 0, ...].bool() # (B, W, H), true where pixel's class is background
        fg = ~bg # (B, W, H), true where pixel's class is foreground

        ready = 0
        B = p_corr.shape[0] # batch size
        for b in range(B):
            fg_p = p_corr[b][fg[b]] # (|fg|,), foreground pixels of image b
            bg_p = p_corr[b][bg[b]] # (|bg|,), background pixels of image b
            f = fg_p.numel() # |fg|, foreground pixel count for image b

            fg_ok = f == 0 or bool(fg_p.max() > self.t)

            k = min(f, bg_p.numel())
            if k == 0:
                bg_ok = True
            else:
                hardest, _ = torch.topk(bg_p, k, largest=False) # (k,), the k hardest background pixels
                bg_ok = bool(hardest.max() > self.t)

            ready += int(fg_ok and bg_ok) 

        if ready > B / 2:
            self._converged = True
            print(">> Balance first phase converged, from now on also doing InterCBL")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        """Just a weighted average of the inter- and intra-losses, by alpha"""
        if not self._converged:
            return self.intra(pred_softmax, weak_target)

        inter_loss = self.inter(pred_softmax, weak_target)
        intra_loss = self.intra(pred_softmax, weak_target)
        return (1 - self.alpha) * inter_loss + self.alpha * intra_loss

