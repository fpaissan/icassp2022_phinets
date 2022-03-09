from _modules.phinet_pl.phinet import PhiNet

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import F1 as F1_score, Accuracy
from torch.optim import Optimizer, Adam
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from typing import Tuple, Union
from torch import Tensor
import torch.nn as nn
import torch

import torch
import torch.nn as nn
from torch import Tensor
from _modules.dnet import DNet, FilterBankConv1d


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    Code from https://github.com/ankandrew/online-label-smoothing-pt
    """

    def __init__(self, alpha: float, n_classes: int, smoothing: float = 0.1):
        """
        :param alpha: Term for balancing soft_loss and hard_loss
        :param n_classes: Number of classes of the classification problem
        :param smoothing: Smoothing factor to be used during first epoch in soft_loss
        """
        super(OnlineLabelSmoothing, self).__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        # Initialize soft labels with normal LS for first epoch
        self.register_buffer('supervise', torch.zeros(n_classes, n_classes))
        self.supervise.fill_(smoothing / (n_classes - 1))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Update matrix is used to supervise next epoch
        self.register_buffer('update', torch.zeros_like(self.supervise))
        # For normalizing we need a count for each class
        self.register_buffer('idx_count', torch.zeros(n_classes))
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor):
        # Calculate the final loss
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss

    def soft_loss(self, y_h: Tensor, y: Tensor):
        """
        Calculates the soft loss and calls step
        to update `update`.
        :param y_h: Predicted logits.
        :param y: Ground truth labels.
        :return: Calculates the soft loss based on current supervise matrix.
        """
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.
        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).
        Note on (5): This is done outside this function since we only need to
                     normalize at the end of the epoch.
        """
        # 1. Calculate predicted classes
        y_h_idx = y_h.argmax(dim=-1)
        # 2. Filter only correct
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        # 3. Add y_h probabilities rows as columns to `memory`
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        # 4. Update `idx_count`
        self.idx_count.index_add_(0, y_h_idx_c, torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        This function should be called at the end of the epoch.
        It basically sets the `supervise` matrix to be the `update`
        and re-initializes to zero this last matrix and `idx_count`.
        """
        # 5. Divide memory by `idx_count` to obtain average (column-wise)
        self.idx_count[torch.eq(self.idx_count, 0)] = 1  # Avoid 0 denominator
        # Normalize by taking the average
        self.update /= self.idx_count
        self.idx_count.zero_()
        self.supervise = self.update
        self.update = self.update.clone().zero_()

class GlobPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.f = nn.functional.avg_pool2d
        
    def forward(self,
                x: Tensor) \
                    -> Tensor:
        return self.f(x, x.size()[2:])

class Unsqueeze(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        return torch.unsqueeze(x, 1)

class SEClassifier(LightningModule):
    def __init__(self,
                 cfgs: dict,
                 waveform=False) \
                     -> None:
        super().__init__()
        self.accuracy = Accuracy()
        self.F1 = F1_score()
        self.criterion = OnlineLabelSmoothing(alpha=1, n_classes=cfgs["num_classes"])
        phinet: PhiNet = PhiNet(**cfgs)
        dnet: DNet = DNet(stride_res=300, depth_multiplier=4.5, num_blocks=4, downsampling_layers=[4], squeeze=True, n_filters=3)
        # import nessi
        # input(phinet.block_filters_n)
        # input(nessi.get_torch_size(phinet, None))
        out_dense = 64
        if not waveform:
            self.cnn: nn.Sequential = nn.Sequential(
                phinet,
                nn.Conv2d(
                    phinet.block_filters_n,
                    out_dense,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding="same",
                    bias=False
                ),
                nn.ReLU(),
                nn.BatchNorm2d(out_dense),
                GlobPool(),
                nn.Flatten(),
                nn.Linear(out_dense, cfgs["num_classes"])
            )
        else:
            self.cnn: nn.Sequential = nn.Sequential(
                dnet
            )
    
    def forward(self,
                x: Tensor) \
                    -> Tensor:
        x = torch.unsqueeze(x, 1)
        x = self.cnn(x)
        
        return x
    
    def loss_fn(self, 
                y_hat: Tensor,
                y_target: Tensor) -> Tensor:
        # return nn.CrossEntropyLoss(reduction="mean")(y_hat,
        #                                              y_target)
        return self.criterion(y_hat, y_target)
    
    def on_train_epoch_end(self, **kwargs) -> None:
        self.criterion.next_epoch()
    
    def configure_optimizers(self) \
        -> Tuple[Optimizer, ReduceLROnPlateau]:
        optimizer = Adam(
            self.cnn.parameters(),
            lr=1e-3,
            weight_decay=1e-2
        )
        
        reduce_lr_on_plateau = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=15,
            verbose=True,
            cooldown=5,
            min_lr=1e-8,
        )
        
        return {
            "optimizer": optimizer,
            # "lr_scheduler": reduce_lr_on_plateau,
            # "monitor": "val/loss"
        }
    
    def _step(self,
              x: Tensor,
              label: Tensor) \
                  -> Union[Tensor, Tensor]:
        output = self(x.float())
        #print(x.shape)
        #print(output.shape)
        loss = self.loss_fn(output, label)
        #print(loss)
        pred = torch.argmax(output, dim=1)
        #print(pred.shape)
        acc = self.accuracy(torch.flatten(pred), torch.flatten(label))
        F1 = self.F1(torch.flatten(pred), torch.flatten(label))
        return loss, acc, F1

    def training_step(self, 
                      batch: Tensor,
                      batch_idx: int) -> Tensor:
        x, label = batch
        loss, acc, F1 = self._step(x, label)

        self.log('train/acc', acc, prog_bar=True, on_step=True)
        self.log('train/loss', loss)
        self.log('train/F1', F1)

        return loss

    def validation_step(self,
                        batch: Tensor,
                        batch_idx: int) -> Tensor:
        x, label = batch
        loss, acc, F1 = self._step(x, label)

        self.log('val/acc', acc)
        self.log('val/loss', loss)
        self.log('val/F1', F1)

        return loss

    def test_step(self,
                  batch: Tensor,
                  batch_idx: int) -> Tensor:
        x, label = batch
        loss, acc, F1 = self._step(x, label)

        self.log('test/acc', acc)
        self.log('test/loss', loss)
        self.log('test/F1', F1)

        return loss
