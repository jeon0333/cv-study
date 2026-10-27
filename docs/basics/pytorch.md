# PyTorch 코드 읽기: 레이어와 텐서의 흐름

이 페이지는 ResNet 구현을 이해하기 위한 최소한의 문법만 다룹니다.

## 학습 목표

- `nn.Module`이 무엇인지 이해한다.
- `forward()`에서 입력·출력 shape를 추적한다.
- 역전파에서 `loss.backward()`의 역할을 설명한다.

## 1. 모델의 기본 구조

```python
import torch
from torch import nn

class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        x = self.features(x)     # (N, 8, 1, 1)
        x = torch.flatten(x, 1) # (N, 8)
        return self.fc(x)       # (N, 10)

model = TinyClassifier()
images = torch.randn(4, 3, 32, 32)
print(model(images).shape)  # (4, 10)
```

**`nn.Module`**은 학습할 가중치와 계산 과정을 하나로 묶는 기본 단위입니다. `forward()`는 입력이 들어왔을 때 어떤 계산을 수행하는지 기술합니다.

## 2. Gradient가 흐르는지 확인

```python
logits = model(images)
labels = torch.tensor([1, 3, 2, 0])
loss = nn.CrossEntropyLoss()(logits, labels)
loss.backward()

print(model.features[0].weight.grad is not None)  # True
```

`CrossEntropyLoss`는 클래스 점수(softmax 적용 전 로짓)와 정답 클래스 번호를 비교합니다. `loss.backward()`는 최적화 스텝을 수행하는 것이 아니라, 각 매개변수의 기울기를 **계산하여 누적**합니다. 보통 학습 루프에서 `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()` 순서로 사용합니다.

## 3. `x + identity`를 읽을 준비

```python
x = torch.randn(2, 16, 32, 32)
identity = x
residual = torch.randn_like(x)
y = residual + identity
assert y.shape == x.shape
```

이 덧셈이 ResNet의 핵심입니다. 텐서의 shape가 서로 다르면 의도하지 않은 broadcasting이 발생하거나 에러가 날 수 있습니다. ResNet에서는 의도적으로 두 경로의 shape를 맞춥니다.

!!! warning "주의: 실제 ResNet 구현과 다름"
    위 `residual`은 무작위 숫자입니다. 실제로는 Conv·BatchNorm·ReLU 등으로 이루어진 **학습 가능한 분기**가 `F(x)`를 계산합니다.

## 4. 미니 확인 문제

`Conv2d(3, 8, 3, padding=1)` 뒤의 텐서가 `(4, 8, 32, 32)`이고, `AdaptiveAvgPool2d(1)` 뒤는 `(4, 8, 1, 1)`입니다. `flatten(x, 1)`의 출력은 왜 `(4, 8)`이 될까요?

??? success "정답"
    첫 축인 배치 4는 유지하고, 이후 모든 축 `(8, 1, 1)`을 하나로 합치기 때문입니다.

**다음:** [ResNet 구조 분석](../papers/resnet/03-architecture.md)
