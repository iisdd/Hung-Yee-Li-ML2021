# 建立神经网络,改网络参数就在这改
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class NET(torch.nn.Module):                     # 定义神经网络
    def __init__(self, state_dim, action_dim):
        # 输出每个动作的q(s, a)
        super(NET, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, state):                   # [m, state_dim] -> [m, action_dim](q值表)
        action_values = self.FC(state)
        return action_values                    # 返回所有a的q