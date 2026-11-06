# 02 · DenseNet — 앞에서 만든 특징을 함께 사용하기

ResNet을 마쳤다면 이번에는 **연결된 특징을 어떻게 합칠 것인가**를 살펴봅니다. 학습 목표는 Dense Block의 입력 채널 수를 계산하고, 작은 블록을 구현해 이전 특징이 보존되는지 확인하는 것입니다.

논문은 Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger의 *Densely Connected Convolutional Networks*입니다. 2016년 공개되었고 CVPR 2017에서 발표되었습니다. 아래 원문 읽기는 arXiv v5 기준입니다. [논문과 원문](https://arxiv.org/abs/1608.06993)

| 학습 순서 | 끝나면 답할 질문 |
|---|---|
| 1 · 연결 | 덧셈과 이어 붙이기는 무엇이 다른가? |
| 2 · 채널 | growth rate로 출력 크기를 어떻게 구하는가? |
| 3 · 구조 | Dense Block 사이에서 크기를 어떻게 줄이는가? |
| 4 · 실험 | 파라미터 효율을 어떤 근거로 평가하는가? |
| 5 · 실습 | 코드가 기존 특징과 기울기를 보존하는가? |
| 6 · 확인 | 새 입력 크기에서도 스스로 계산할 수 있는가? |

## 1. ResNet에서 이어지는 질문

ResNet의 항등 shortcut은 앞의 값을 뒤로 전달하는 짧은 경로를 제공합니다. DenseNet은 같은 Dense Block 안에서 앞선 층들의 특징을 후속 층에 전달하며, 채널 방향으로 이어 붙입니다. 이를 **concatenation**이라고 합니다. 각 층은 모인 특징으로부터 새로운 특징을 만듭니다. [원문 §3](https://arxiv.org/html/1608.06993v5#S3)

직접 만든 작은 숫자 예제로 비교해 봅시다. 한 위치의 기존 특징이 `[2, 5]`, 새 특징이 `[3, 7]`이라면 다음과 같습니다.

| 연산 | 결과 | 출력 채널 수 |
|---|---|---|
| 원소별 덧셈 | `[5, 12]` | 2 |
| 채널 이어 붙이기 | `[2, 5, 3, 7]` | 4 |

이어 붙인 결과에서 처음 두 채널은 기존 값을 그대로 가집니다. 다음 합성곱은 네 채널에 가중치를 적용하면서 필요한 조합을 배울 수 있습니다. 채널이 많아지는 비용도 함께 생깁니다.

!!! tip "종이 위에서 해 보기"
    기존 특징 `[2, 5]`에 새 특징 `[9]`를 이어 붙여 보세요. 결과는 `[2, 5, 9]`입니다. 원소별 덧셈에서는 같은 위치끼리 대응시킬 shape를 맞춰야 하지만, 채널 이어 붙이기는 두 입력의 채널 수가 달라도 됩니다. 배치 크기와 공간 크기는 같아야 합니다.

## 2. 수식과 growth rate를 읽기

\[
x_l = H_l([x_0,x_1,\ldots,x_{l-1}])
\]

| 기호 | 여기서의 뜻 |
|---|---|
| \(x_0\) | 현재 Dense Block에 처음 들어온 특징 |
| \(x_l\) | l번째 층이 새로 만든 특징 |
| \(H_l\) | 정규화·활성화·합성곱으로 구성한 변환 |
| \([\cdot]\) | 채널 방향 이어 붙이기 |
| \(k\) | 한 층이 추가하는 채널 수, growth rate |

입력 채널이 \(C_0\)이고 L개 층이 각각 k개를 추가하면, l번째 층의 입력은 \(C_0+(l-1)k\), 블록 전체 출력은 \(C_0+Lk\)개 채널입니다. [원문 식 (2)와 growth rate 설명](https://arxiv.org/html/1608.06993v5#S3)

### 16채널에서 시작하는 계산 연습

입력 `(N, 16, 8, 8)`, `k=8`, `L=3`으로 직접 계산해 봅시다.

| 단계 | 변환이 받는 채널 | 새로 만드는 채널 | 모아 둔 전체 채널 |
|---|---:|---:|---:|
| 시작 | — | — | 16 |
| 첫째 층 | 16 | 8 | 24 |
| 둘째 층 | 24 | 8 | 32 |
| 셋째 층 | 32 | 8 | 40 |

따라서 마지막 출력은 `(N, 40, 8, 8)`입니다. **새 층의 출력 8채널과 블록 전체 출력 40채널을 구분**해야 합니다. k를 12로 바꾸면 `16 + 3×12 = 52`가 됩니다. 같은 블록 안의 3×3 합성곱은 `padding=1, stride=1`로 공간 크기를 유지할 수 있습니다.

## 3. Dense Block과 Transition

DenseNet-B의 각 층은 `BN → ReLU → 1×1 Conv → BN → ReLU → 3×3 Conv` 순서이며, 내부 1×1 합성곱은 4k채널을 만듭니다. Compression은 블록 사이 채널을 \(\lfloor\theta C\rfloor\)로 줄입니다. 두 방법을 함께 쓴 형태가 DenseNet-BC입니다. [원문 §3](https://arxiv.org/html/1608.06993v5#S3)

예를 들어 앞의 40채널 출력에 `θ=0.5`를 적용하면 20채널입니다. 이어서 2×2 평균 풀링을 하면 공간 크기는 `8×8 → 4×4`가 됩니다. 다음 블록은 `(N, 20, 4, 4)`에서 새롭게 시작합니다.

```text
입력 특징 16×8×8
  ↓ Dense Block 3개 층, 각 8채널 추가
모인 특징 40×8×8
  ↓ Transition의 1×1 합성곱
압축 특징 20×8×8
  ↓ 2×2 평균 풀링
다음 블록 입력 20×4×4
```

위 도식의 숫자는 학습용 예제입니다. Transition을 지나며 해상도가 달라지므로, 이전 블록의 모든 특징을 새 블록에 그대로 이어 붙이는 것으로 해석하면 안 됩니다.

### DenseNet-121을 채널 계산으로 따라가기

Torchvision의 DenseNet-121은 시작 채널 64, `k=32`, 블록별 층 수 `(6, 12, 24, 16)`을 사용합니다. 아래는 이 설정에 따라 직접 계산한 채널 수입니다. [Torchvision 구현](https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py)

| 구간 | 계산 | 출력 채널 |
|---|---|---:|
| 블록 1 | 64 + 6×32 | 256 |
| Transition 1 | 256÷2 | 128 |
| 블록 2 | 128 + 12×32 | 512 |
| Transition 2 | 512÷2 | 256 |
| 블록 3 | 256 + 24×32 | 1024 |
| Transition 3 | 1024÷2 | 512 |
| 블록 4 | 512 + 16×32 | 1024 |

121이라는 이름을 셀 때는 각 Dense Layer의 두 합성곱을 포함합니다. 초기 합성곱 1개, 블록 내부 합성곱 `2×(6+12+24+16)=116`개, Transition 합성곱 3개, 마지막 분류층 1개를 더하면 121입니다.

## 4. 논문 실험을 읽는 순서

원문 Table 2의 **C10+는 데이터 증강을 적용한 CIFAR-10 시험 오류율**입니다. 비교용으로 두 행을 읽어 봅시다. [원문 Table 2](https://arxiv.org/html/1608.06993v5#S4.T2)

| 모델 | 파라미터 | C10+ 오류율 |
|---|---:|---:|
| pre-activation ResNet-1001 | 10.2M | 4.62% |
| DenseNet-BC, L=100, k=12 | 0.8M | 4.51% |

오류율 차이는 `4.62−4.51 = 0.11%p`, 파라미터 비율은 `0.8÷10.2 ≈ 7.8%`입니다. 이 두 행은 **비슷한 시험 오류율에 필요한 파라미터 수**를 비교하는 연습에 적합합니다. 연결 방식만의 효과를 분리하려면 훈련 설정과 모델 구성이 통제된 추가 비교가 필요합니다.

### 숫자를 읽고 적을 세 가지

1. 측정값이 오류율인지 정확도인지 적습니다. 위 표는 낮을수록 좋은 오류율입니다.
2. 데이터 증강, 깊이, k, 학습 조건을 함께 기록합니다. 이름이 같아도 설정이 달라지면 비교가 달라집니다.
3. 파라미터 수, 실제 학습 메모리, 실행 시간을 각각 측정합니다. 이어 붙인 중간 텐서를 많이 저장하면 메모리 사용량이 커질 수 있습니다. 저자도 메모리 효율 구현을 따로 제공합니다. [저자 코드와 메모리 효율 구현 안내](https://github.com/liuzhuang13/DenseNet)

## 5. PyTorch로 작은 블록 검증하기

아래 예제를 `dense_block_demo.py`로 저장하고 PyTorch가 설치된 환경에서 `python dense_block_demo.py`로 실행합니다. CPU와 임의 텐서만으로 연결 구조를 확인합니다. 실습의 검증 대상은 **출력 크기, 원래 특징의 보존, 역전파**입니다.

```python
import torch
from torch import nn


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        if min(in_channels, growth_rate, num_layers) < 1:
            raise ValueError("채널 수와 층 수는 양수여야 합니다.")
        self.layers = nn.ModuleList()
        for index in range(num_layers):
            channels = in_channels + index * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, 4 * growth_rate, 1, bias=False),
                nn.BatchNorm2d(4 * growth_rate),
                nn.ReLU(),
                nn.Conv2d(4 * growth_rate, growth_rate, 3,
                          padding=1, bias=False),
            ))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


if __name__ == "__main__":
    torch.manual_seed(0)
    block = DenseBlock(in_channels=16, growth_rate=8, num_layers=3)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    y = block(x)
    assert y.shape == (2, 40, 8, 8)
    torch.testing.assert_close(y[:, :16], x)

    # 새 특징만으로 손실을 만들어 학습 분기의 역전파를 검사합니다.
    y[:, 16:].square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for layer in block.layers:
        grad = layer[-1].weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    print("shape, feature preservation, backward: PASS")
```

### 코드를 수식과 연결하기

`features`에는 입력과 각 층의 새 출력이 쌓입니다. `torch.cat(..., dim=1)`은 `(N,C,H,W)`의 C축을 이어 붙입니다. 매번 쌓인 특징 전체를 `layer`에 전달하는 부분이 식의 대괄호에 해당합니다.

`ModuleList`는 각 층을 모델의 학습 파라미터로 등록합니다. 일반 Python 리스트에만 층을 넣으면 `model.parameters()`로 관리되지 않을 수 있습니다. 마지막 `return`에서는 최초 입력도 포함해 블록 출력을 만듭니다.

### 직접 바꿔 보는 실험

- `growth_rate=12`로 바꾸고 출력 검사를 `(2,52,8,8)`로 고칩니다.
- `num_layers=4`와 `growth_rate=8`을 사용하면 출력은 `(2,48,8,8)`입니다.
- `dim=1`을 `dim=0`으로 잘못 바꾸면 어느 층에서 shape가 맞지 않는지 설명해 봅니다.

분류 성능을 평가하려면 분류기, 데이터셋 분할, 학습 반복과 평가 절차를 추가해야 합니다. 위 PASS는 블록 구현의 검사 결과이며 논문 정확도를 재현한 결과로 기록하면 안 됩니다.

## 6. 이해도 확인

??? question "1. 입력이 24채널이고 k=12인 층이 4개면 블록 출력은?"
    `24 + 4×12 = 72`채널입니다. 네 번째 층이 받는 입력은 `24 + 3×12 = 60`채널입니다.

??? question "2. 72채널에 θ=0.5를 적용하고 16×16을 풀링하면?"
    1×1 합성곱으로 36채널, 2×2 stride 2 평균 풀링으로 8×8이 됩니다. 출력은 `(N,36,8,8)`입니다.

??? question "3. 입력 특징이 뒤에서도 남아 있으면 앞선 층은 학습할 필요가 없나요?"
    후속 층은 이전 특징을 이용해 손실을 계산합니다. 그 계산 경로를 통해 앞선 층의 파라미터에도 기울기가 전달됩니다. 보존된 특징과 학습 가능한 가중치는 구분해야 합니다.

??? question "4. 파라미터가 절반이면 GPU 메모리도 절반인가요?"
    메모리에는 중간 활성값, 기울기, 옵티마이저 상태 등이 들어갑니다. 배치 크기와 구현 방식도 영향을 주므로 실제 최대 메모리를 측정해야 합니다.

??? question "5. DenseNet의 dense는 모든 픽셀을 완전연결한다는 뜻인가요?"
    여기서는 같은 블록 안의 층들이 앞선 특징을 후속 층에 연결하는 패턴을 말합니다. 특징을 만드는 연산에는 합성곱을 사용합니다.

## 7. 마무리와 다음 실험

빈 종이에 세 층의 연결을 그리고 각 화살표의 채널 수를 적어 보세요. 이어서 자신의 말로 '어떤 특징을 재사용하고, 새 층은 무엇을 추가하며, 어디에서 크기를 줄이는가'를 설명하면 됩니다.

[학습 진행표](../../progress.md)에 체크한 뒤, [실험 기록법](../../experiments/index.md)에 k와 층 수를 바꾼 결과를 남깁니다. 다음 로드맵 후보인 U-Net에서는 해상도가 다른 특징을 다시 합칠 때 필요한 연산을 살펴볼 예정입니다.
