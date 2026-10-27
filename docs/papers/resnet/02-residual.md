# 2. Residual Connection: 전체를 다시 만드는 대신, 차이를 더한다

이번 단원은 ResNet을 이해하는 핵심입니다. 하지만 이름을 외우기보다 **왜 이런 계산 형태를 택했는지**를 먼저 이해해 보겠습니다.

## 1. 이름부터 이해하기

**Residual(잔차)**은 '현재 값과 목표 값 사이에 남아 있는 차이'를 뜻합니다. 예측과 정답 사이의 오차만을 뜻하는 용어로 오해하기 쉽지만, 여기서는 **입력 `x`와 블록이 만들고 싶은 변환 `H(x)`의 차이**를 가리킵니다.

| 기호 | 쉬운 뜻 |
|---|---|
| \(x\) | 블록의 입력 |
| \(H(x)\) | 최종적으로 만들고 싶은 출력 |
| \(F(x)\) | 입력에 더할 변화량, 즉 \(H(x)-x\) |

ResNet은 **\(H(x)\) 전체를 바로 학습하게 하는 대신 \(F(x)\)를 학습하도록 구조를 바꿉니다.**

\[
\boxed{H(x)=F(x)+x}
\]

![일반 블록과 잔차 블록의 차이](../../assets/residual-block.svg){ .study-diagram }

이 그림에서 오른쪽으로 우회하는 길이 **shortcut connection(지름길 연결)**입니다. 학습 블록을 통과하지 않은 `x`를 나중에 더합니다.

## 2. 숫자로 먼저 이해하기

현재 값이 `5`이고 목표 출력이 `7`이라면 두 가지 방식으로 생각할 수 있습니다.

- **일반 방식:** 주어진 입력 `5`를 받아서 출력 `7` 전체를 만드는 변환 \(H\)를 학습합니다.
- **잔차 방식:** `5`는 그대로 옆으로 보내고, 학습 블록은 **`+2`라는 변화량**만 만들도록 합니다. 결과는 `5+2=7`입니다.

아래 입력값을 직접 바꿔 보세요. 학습을 실행하는 시뮬레이터는 아니며, **잔차의 정의를 산술적으로 체험하는 도구**입니다.

<div class="example-widget" data-residual-demo>
  <strong>직접 계산해 보기</strong>
  <div class="widget-fields">
    <label>현재 입력 x<input data-residual-x type="number" value="5" step="1" aria-label="현재 입력 x"></label>
    <label>목표 출력 H(x)<input data-residual-target type="number" value="7" step="1" aria-label="목표 출력 H(x)"></label>
  </div>
  <div class="widget-result" aria-live="polite" data-residual-result>그대로 보내는 값 x = 5; 추가로 배울 변화량 F(x) = 2; 결과 F(x) + x = 7</div>
</div>

## 3. 왜 이 구조가 학습을 돕는다고 생각했을까?

추가된 몇 개의 층이 아무것도 변경하지 않아야 하는 상황을 가정해 보겠습니다. 이상적인 함수가 \(H(x)=x\)라면 잔차는 다음과 같습니다.

\[
F(x)=H(x)-x=x-x=0
\]

즉 학습할 분기가 **0에 가까운 출력을 내면** 블록 전체가 입력을 거의 그대로 내보낼 수 있습니다.

반대로 shortcut이 없는 일반 블록은 여러 개의 가중치·활성화를 거쳐 **입력을 그대로 출력하는 변환 자체를 구현**해야 합니다. 논문 저자들은 차이를 학습하는 재구성이 더 최적화하기 쉬울 수 있다고 가설을 세웠습니다.

!!! important "핵심은 가능성이 아니라 학습 난이도"
    \(H(x)\)와 \(F(x)+x\)는 표현 관점에서 같은 목표 함수를 나타낼 수 있습니다. ResNet의 핵심은 **표현 가능한 함수 집합 자체를 바꾸는 것보다는 좋은 해를 학습하기 쉬운 형태로 만드는 것**에 있습니다. 실제 표현 능력은 구조와 제약에 따라 달라질 수 있습니다.

## 4. 역전파에서도 어떤 일이 일어날까?

단순화를 위해 덧셈 직후의 출력을 \(y=x+F(x)\)라고 하겠습니다. 미분하면 다음과 같습니다.

\[
\frac{\partial y}{\partial x}=I+\frac{\partial F(x)}{\partial x}
\]

`I`는 입력을 그대로 통과시키는 항등 경로의 미분입니다. 스칼라(숫자 하나)라면 `1`이고, 벡터나 텐서에서는 항등 행렬(연산자)에 해당합니다.

출력 쪽의 기울기가 입력 쪽으로 전달될 때 **잔차 분기 이외에 항등 경로가 하나 더 존재한다**는 뜻입니다. 깊은 네트워크의 최적화를 도울 수 있는 중요한 이유 중 하나입니다.

!!! warning "단순화한 수식임을 기억하기"
    ResNet 원 논문의 기본 블록(흔히 ResNet v1)은 덧셈 **뒤에 ReLU**가 있습니다. 실제 전체 블록의 미분에는 이 활성화의 미분과 그 뒤의 연산이 함께 작용합니다. 위 수식만으로 '기울기가 절대 사라지지 않는다'는 결론을 낼 수는 없습니다.

## 5. 구현은 놀랄 만큼 간단합니다

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv(x)   # 학습한 F(x)
        return self.relu(residual + x)  # F(x) + x
```

입력과 출력의 **채널 수와 공간 크기가 같은 경우**에만 이처럼 바로 더할 수 있습니다. 구조가 달라져 shape가 달라지면 [다음 단원의 projection shortcut](03-architecture.md)을 사용합니다.

## 6. 흔한 오해

- **'잔차는 정답과 예측 사이의 loss다'** → 이 맥락에서 잔차는 블록의 **목표 변환과 입력의 차이**입니다.
- **'Shortcut은 모든 층을 생략한다'** → 학습 분기와 shortcut 분기는 **함께 계산된 뒤 더해집니다.**
- **'Shortcut에는 반드시 학습할 가중치가 필요하다'** → shape가 같다면 항등 shortcut은 추가 가중치 없이 구현할 수 있습니다.
- **'Residual connection만 넣으면 성능이 무조건 올라간다'** → 논문은 특정 구조와 학습 조건에서 효과를 실험으로 보여 줍니다. 보편적 보장은 아닙니다.

## 확인 문제

입력이 `(2, 64, 32, 32)`이고 `F(x)`가 `(2, 128, 16, 16)`이라면 `F(x)+x`를 그대로 실행할 수 있을까요?

??? success "답"
    두 텐서의 채널 수와 공간 크기가 다릅니다. ResNet에서는 shortcut 쪽에도 공간 다운샘플링과 채널 변환을 적용해 `(2, 128, 16, 16)`로 맞춥니다. **1×1 convolution**은 이러한 projection에 자주 사용됩니다.

**다음:** [3. ResNet 18/34/50의 구조와 projection shortcut](03-architecture.md)
