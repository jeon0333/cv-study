"""학습용 CIFAR ResNet (post-activation BasicBlock).

원 논문 ImageNet 설정의 성능을 재현하는 모델이 아닙니다.
입력: (N, 3, 32, 32)
출력: (N, num_classes)
"""
from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """F(x) + shortcut(x) 형태의 두 개 3x3 convolution 블록."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError('채널 수는 양의 정수여야 합니다.')
        if stride not in (1, 2):
            raise ValueError('이 교육용 블록은 stride=1 또는 2를 사용합니다.')

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class TinyResNet(nn.Module):
    """CIFAR-10용 4-stage 소형 ResNet.

    기본 블록 수: [2, 2, 2, 2]
    스템: 3x3 stride 1 (ImageNet 논문의 7x7 stride 2와 다름)
    """

    def __init__(self, num_classes: int = 10, blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)) -> None:
        super().__init__()
        if len(blocks_per_stage) != 4 or any(b <= 0 for b in blocks_per_stage):
            raise ValueError('blocks_per_stage는 양의 정수 4개여야 합니다.')
        if num_classes <= 1:
            raise ValueError('num_classes는 2 이상이어야 합니다.')

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        in_channels = 16
        stages: list[nn.Module] = []
        for stage_idx, (out_channels, count) in enumerate(zip((16, 32, 64, 128), blocks_per_stage)):
            stage_blocks = []
            for block_idx in range(count):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                stage_blocks.append(BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*stage_blocks))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
