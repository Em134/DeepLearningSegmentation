# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type

class DomainCommon(nn.Module):
    def __init__(self, D_features, mlp_ratio=2.0):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        x = self.D_fc1(x)
        x = self.act(x)
        x = self.D_fc2(x)
        return x


class DomainSpecific(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25):
        super().__init__()
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = nn.GELU()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        x = self.D_fc1(x)
        x = self.act(x)
        x = self.D_fc2(x)
        return x