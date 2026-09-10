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


from torch import einsum, Tensor

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        loss = - einsum("bkwh,bkwh->", mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss

class DiceCE():
    """The 'standard' loss function given during the lecture,
    and the default in nnU-Net, combining cross-entropy
    and dice loss, so that L = L_CE + lambda * L_DICE"""

    def __init__(self, idk: list[int], lambda_: float, smooth: float = 1e-8):
        self.idk = idk
        self.lambda_ = lambda_
        self.smooth = smooth
        print(f"Initialized {self.__class__.__name__} with {idk=}, lambda = {lambda_}")

    def __call__(self, pred_softmax: Tensor, weak_target: Tensor) -> Tensor:
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred_softmax = pred_softmax[:, self.idk, ...]
        mask = weak_target[:, self.idk, ...].float()

        # Cross-entropy term
        log_p = (pred_softmax + 1e-10).log()
        ce_loss = - einsum("bkwh,bkwh->", mask, log_p)
        ce_loss /= mask.sum() + 1e-10

        # Soft dice term, averaged over classes and batch
        intersection = einsum("bkwh,bkwh->bk", pred_softmax, mask)
        union = einsum("bkwh->bk", pred_softmax) + einsum("bkwh->bk", mask)
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return ce_loss + self.lambda_ * dice_loss


class PartialCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        super().__init__(idk=[1], **kwargs)
