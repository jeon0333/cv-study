# 첫 실험 계획: TinyResNet으로 CIFAR-10 분류하기

<small>상태: **실행 전 계획서** · 아래에는 측정하지 않은 정확도 수치를 넣지 않았습니다.</small>

## 무엇을 알고 싶나요?

Residual Block이 **의도한 shape로 작동하고 역전파가 가능한지** 확인한 뒤, 작은 데이터셋을 이용하여 모델이 무작위 예측보다 의미 있는 학습을 수행할 수 있는지 살펴봅니다.

## 실행 조건

| 항목 | 계획 |
|---|---|
| 데이터셋 | CIFAR-10 |
| 훈련 데이터 | 공식 train split에서 5,000개 샘플 사용 |
| 평가 데이터 | 공식 test split에서 2,000개 샘플 사용 |
| 모델 | TinyResNet, BasicBlock |
| Epoch | 2 (동작 확인용, 충분한 수렴 보장 X) |
| Batch size | 128 |
| Seed | 42 |
| 학습 손실 | CrossEntropyLoss |
| 평가 기준 | 분류 정확도(%) |

```bash
python -m unittest discover -s tests -v
python -m cv_models.train_cifar10 --epochs 2 --subset-size 5000 --eval-size 2000
```

## 실행 후 직접 작성할 기록

| 항목 | 관찰값 |
|---|---|
| Python / PyTorch 버전 | 미기록 |
| CPU / GPU 종류 | 미기록 |
| 마지막 epoch 훈련 손실 | **미실행** |
| 마지막 epoch 훈련 정확도 | **미실행** |
| 마지막 epoch test subset 정확도 | **미실행** |
| 실험 소요 환경/비고 | 미기록 |

## 결과를 해석할 때 주의할 점

- 이 실습에서 정확도가 높아졌다고 해서 **원 논문의 ResNet-34 ImageNet 성능을 재현**한 것은 아닙니다.
- 2 epoch는 기능 점검용입니다. 수렴이나 모델 간 우위를 판단하기에 충분하지 않을 수 있습니다.
- 모델 비교를 여러 번 반복할 때는 공식 test split으로 하이퍼파라미터를 고르지 말고 validation split을 따로 구성합니다.
- 정확도가 낮더라도 먼저 **데이터, shape, 손실, gradient, 학습률**을 확인합니다.

## 다음 실험 아이디어

동일 데이터와 학습 조건에서 shortcut의 유무를 바꿔 최적화 곡선을 비교합니다. 가능하면 네트워크 깊이도 바꾸되 파라미터 수 및 계산량 차이를 통제하려는 설계를 추가합니다.

[실습으로 돌아가기](../papers/resnet/05-lab.md)
