# 5. PyTorch 실습: 내가 만든 ResNet이 실제로 동작할까?

여기서부터는 수식을 코드에 연결합니다. 처음부터 ImageNet 전체를 학습할 필요는 없습니다. **정확한 shape, shortcut 동작, gradient 전달**을 확인하고 작은 데이터셋에서 모델을 훈련하는 것이 먼저입니다.

## 실습 목표와 범위

| 요소 | 이번 실습 |
|---|---|
| 문제 | CIFAR-10의 10개 범주 이미지 분류 |
| 입력 | `(N, 3, 32, 32)` RGB 이미지 |
| 출력 | `(N, 10)` 클래스 로짓(logits) |
| 모델 | 교육용 소형 ResNet (`BasicBlock` 사용) |
| 학습 목표 | `CrossEntropyLoss`를 최소화 |
| 평가 | 별도의 CIFAR-10 test subset에서 정확도 기록 |
| 한계 | ImageNet의 원본 ResNet-34 실험을 재현하는 코드는 아님 |

## 1. 실습 환경 설치

Python 3.10+와 PyTorch가 필요합니다. 하드웨어(CPU, NVIDIA GPU, Apple Silicon)에 따라 PyTorch 설치 명령이 다를 수 있으므로 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 본인 환경에 맞는 명령을 확인하세요.

```bash
# 이 저장소를 내려받은 뒤 프로젝트 루트에서:
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell은 .venv\Scripts\Activate.ps1

# CPU/GPU 환경에 맞게 torch, torchvision 설치
# https://pytorch.org/get-started/locally/
```

GitHub에 게시된 코드는 [cv_models/resnet.py](https://github.com/jeon0333/cv-study/blob/main/cv_models/resnet.py), [cv_models/train_cifar10.py](https://github.com/jeon0333/cv-study/blob/main/cv_models/train_cifar10.py)에서 볼 수 있습니다.

## 2. 첫 번째 목표: shape를 확인하세요

`cv_models/resnet.py`에는 `BasicBlock` 클래스가 있습니다. 다음 코드를 직접 실행해 보세요.

```python
import torch
from cv_models.resnet import BasicBlock

x = torch.randn(2, 16, 32, 32)
block = BasicBlock(16, 16)
y = block(x)
print(y.shape)  # torch.Size([2, 16, 32, 32])
```

이유: `stride=1`이고 입력·출력 채널이 16으로 같으므로 항등 shortcut과 합칠 수 있습니다.

## 3. 두 번째 목표: 공간 크기가 바뀌는 블록

```python
block = BasicBlock(16, 32, stride=2)
y = block(x)
print(y.shape)  # torch.Size([2, 32, 16, 16])
```

왜 `stride=2`인데 오류가 나지 않을까요? 학습 경로뿐 아니라 **shortcut 경로에도 1×1 convolution을 넣어 shape를 맞췄기 때문**입니다. [구조 분석](03-architecture.md)을 다시 확인해 보세요.

## 4. 세 번째 목표: 기울기 계산

```python
block = BasicBlock(16, 16)
x = torch.randn(2, 16, 32, 32, requires_grad=True)
y = block(x)
loss = y.square().mean()  # 오직 gradient 확인을 위한 예시 손실
loss.backward()

print(x.grad is not None)          # True
print(torch.isfinite(x.grad).all())  # tensor(True)
```

이 코드는 **역전파가 계산 가능**한지만 확인합니다. ResNet의 정확도나 모든 깊이에서의 gradient 안정성을 증명하는 실험은 아닙니다.

## 5. 자동 테스트 실행

프로젝트 루트에서 다음을 실행합니다.

```bash
python -m unittest discover -s tests -v
```

테스트는 다음을 확인합니다.

- 항등 shortcut에서 입력·출력 shape 일치
- projection shortcut에서 채널 수와 공간 크기 변환
- 블록에서 gradient 계산 가능 여부
- 분류 모델의 출력 `(batch_size, num_classes)` 일치

테스트가 성공했다면 '최소 구현이 동작한다'는 의미입니다. 높은 정확도까지 보장하는 것은 아닙니다.

## 6. 소규모 CIFAR-10 학습 실험

```bash
python cv_models/train_cifar10.py --epochs 2 --subset-size 5000 --eval-size 2000 --batch-size 128
```

최초 실행 시 `torchvision`이 CIFAR-10 데이터셋을 다운로드해야 합니다. 데이터셋 크기와 인터넷 연결을 확인하세요. GPU가 없다면 CPU에서도 실행되지만 실행 시간은 하드웨어에 따라 달라집니다.

실습 코드는 매 epoch의 **훈련 손실·훈련 정확도·평가 정확도**를 출력합니다. 여기에 나오는 숫자는 환경과 무작위성, 데이터 subset 등에 따라 달라지므로 교재에 미리 적어 놓지 않았습니다.

!!! warning "데이터셋 누수 방지"
    훈련 데이터에서 일부만 가져오더라도 **평가에는 CIFAR-10의 공식 test split**을 사용합니다. 다만 여러 실험을 통해 하이퍼파라미터를 선택하려면 test split 대신 training split에서 따로 떼어낸 validation split이 필요합니다. 반복적인 모델 선택에 test set을 쓰면 테스트 성능이 과대평가될 수 있습니다.

## 7. 실험 기록: 재현 가능한 메모 남기기

실험 후 [실험 기록 템플릿](../../experiments/index.md)을 채우세요.

```text
실험 이름: tiny-resnet-baseline
날짜/환경: [작성]
데이터: CIFAR-10 / train subset 5,000 / test subset 2,000
모델: TinyResNet, BasicBlock
학습 설정: epochs 2, batch 128, optimizer SGD, seed 42
훈련 손실: [실행 후 기록]
평가 정확도: [실행 후 기록]
발견한 문제 / 다음 실험: [작성]
```

**확장 과제:** 동일한 채널 수·깊이 범위에서 **shortcut을 제거한 plain 모델**을 만들어 같은 조건으로 비교해 보세요. 단, 더 깊은 네트워크의 성능 저하가 짧은 실습에서 반드시 재현되지는 않습니다.

**다음:** [6. 진짜 이해했는지 확인하는 질문](06-quiz.md)
