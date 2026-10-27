# 3. ResNet 구조: BasicBlock, Bottleneck, Projection

이전 페이지에서는 \(H(x)=F(x)+x\)를 이해했습니다. 이제 실제 네트워크에 여러 블록을 연결할 때 **채널 수와 공간 크기**가 어떻게 달라지는지 보겠습니다.

## 1. 전체 구조를 먼저 보기

ResNet은 대략 다음 순서로 동작합니다.

```text
이미지 입력
    ↓
초기 합성곱(Stem)
    ↓
Residual Block 여러 개 (Stage 1)
    ↓  ← Stage 전환 시 공간 크기 감소·채널 증가
Residual Block 여러 개 (Stage 2)
    ↓
Residual Block 여러 개 (Stage 3)
    ↓
Residual Block 여러 개 (Stage 4)
    ↓
Global Average Pooling
    ↓
Fully Connected (분류 점수)
```

**Stage**는 비슷한 공간 해상도와 채널 규모에서 연속적으로 처리하는 블록 묶음입니다. Stage가 바뀔 때 보통 공간 크기를 줄이고 채널 수를 늘립니다.

## 2. BasicBlock: 3×3 합성곱 두 개

ResNet-18/34에서 주로 쓰는 **BasicBlock**은 다음 구성을 갖습니다.

```text
 x ────────────────────────┐
 │                          │
 ↓                          │
 3×3 Conv → BN → ReLU       │
 ↓                          │
 3×3 Conv → BN              │
 │                          │
 └──────── 더하기(+) ←──────┘
              ↓
             ReLU
```

- 3×3 합성곱: 주변의 지역 정보를 처리합니다.
- BN(Batch Normalization): 배치 통계를 사용하여 중간 활성값을 정규화하는 계층입니다.
- ReLU: 음수를 0으로 바꾸는 비선형 활성화 함수입니다.
- Shortcut: 조건이 맞으면 `identity = x`처럼 입력을 통과시킵니다.

### 예: shape가 같을 때

입력이 `(8, 64, 32, 32)`이고 블록 내 두 합성곱이 `64`채널을 유지하며 `stride=1`, `padding=1`이라면, `F(x)`도 `(8, 64, 32, 32)`입니다. 그러므로 원래 입력 `x`를 그대로 더할 수 있습니다.

## 3. Projection shortcut: shape를 맞춰야 한다

다음 단계에서 공간 크기를 절반으로 줄이면서 채널 수를 2배로 늘린다고 합시다.

```text
입력 x                     (8, 64, 32, 32)
 │
 ├─ 학습 경로 F(x) ───────> (8, 128, 16, 16)
 │
 └─ shortcut 1×1 conv ───> (8, 128, 16, 16)
                              ↓
                       두 출력을 더함
```

**Projection(사영/변환) shortcut**은 shortcut에서도 변환을 수행해 shape를 맞추는 방법입니다. 가장 흔한 구현은 `kernel_size=1`의 합성곱과 적절한 stride를 사용하는 것입니다.

\[
y=F(x,\{W_i\})+W_sx
\]

여기서 \(W_s\)는 shortcut에 적용하는 선형 변환입니다. 원 논문에서는 차원이 변할 때 zero-padding을 활용한 옵션(A)과 projection을 사용하는 옵션(B) 등을 비교했습니다. **항등 shortcut은 추가 학습 파라미터가 없지만, projection shortcut에는 파라미터가 생길 수 있습니다.**

## 4. Bottleneck: 1×1 → 3×3 → 1×1

ResNet-50/101/152에서는 더 깊은 모델의 계산 비용을 줄이기 위해 **Bottleneck** 블록을 사용합니다.

```text
x → 1×1 Conv(채널 축소) → 3×3 Conv(공간 특징)
  → 1×1 Conv(채널 확대) → shortcut 더하기 → ReLU
```

1×1 합성곱은 공간상의 이웃을 함께 보지 않고 **같은 위치의 채널 정보**를 섞습니다. 먼저 채널 수를 줄여 3×3 합성곱의 비용을 줄이고, 마지막에 채널 수를 늘립니다.

??? info "ResNet-50의 '50층'은 블록 50개일까요?"
    아닙니다. 관례적으로 합성곱과 마지막 분류 레이어 등을 세는 이름입니다. ResNet-50은 네 개 stage에 `[3,4,6,3]`개의 **Bottleneck block**을 배치합니다. 이것을 그대로 50개의 블록이라고 부르면 틀립니다.

## 5. ResNet-18, 34, 50 비교

| 모델 | 4개 stage의 블록 수 | 블록 종류 | 대략적인 구조적 특징 |
|---|---|---|---|
| ResNet-18 | `[2,2,2,2]` | BasicBlock | 작은 기본 구조 |
| ResNet-34 | `[3,4,6,3]` | BasicBlock | 같은 블록을 더 많이 쌓음 |
| ResNet-50 | `[3,4,6,3]` | Bottleneck | 블록 하나가 3개 Conv를 포함 |

같은 `[3,4,6,3]`이라도 **블록 내부의 합성곱 개수**가 다르므로 34층과 50층이 됩니다.

## 6. 실제 코드에서 확인할 부분

[이 사이트의 `cv_models/resnet.py` 원본](https://github.com/jeon0333/cv-study/blob/main/cv_models/resnet.py)에는 CIFAR-10 학습을 위한 `BasicBlock`과 `TinyResNet` 구현을 넣었습니다. 주요 부분은 다음과 같습니다.

```python
if stride != 1 or in_channels != out_channels:
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1,
                  stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )
else:
    self.shortcut = nn.Identity()
```

이 코드는 **shape가 다르면 shortcut 경로도 변환**하고, 같으면 그대로 전달합니다.

!!! warning "논문 그대로의 ImageNet ResNet-18인가요?"
    아니요. 제공된 교육용 코드는 **CIFAR-10의 작은 32×32 이미지**를 위한 모델입니다. 원 논문의 ImageNet 모델은 입력 크기와 초기 7×7 conv, max pooling 등을 포함한 설정이 다릅니다. 블록 원리는 같지만 재현 성능을 비교해서는 안 됩니다.

## 확인 문제

**Q1.** `(4, 64, 32, 32)`를 `(4, 128, 16, 16)`로 바꾸는 블록에서 shortcut을 그냥 `nn.Identity()`로 쓰면 되나요?

??? success "답"
    안 됩니다. 채널 수와 공간 크기가 달라 덧셈할 수 없습니다. `1×1, stride=2` convolution 등으로 shape를 맞춰야 합니다.

**Q2.** ResNet-34와 ResNet-50의 stage별 블록 수가 같은데 왜 이름이 다를까요?

??? success "답"
    34는 2개의 Conv를 가진 BasicBlock, 50은 3개의 Conv를 가진 Bottleneck을 사용하기 때문입니다.

**다음:** [4. 논문 Table 2와 Figure 4를 읽는 방법](04-results.md)
