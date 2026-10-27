# 7. ResNet 정리: 논문 한 편을 내 지식으로 바꾸기

## 다섯 문장으로 요약하기

1. 이미지 인식용 깊은 CNN은 표현 능력이 크지만, 단순히 더 깊게 쌓으면 **훈련 오류가 증가하는 성능 저하 문제**가 나타날 수 있습니다.
2. ResNet은 입력 \(x\)를 블록 옆으로 통과시키는 **shortcut connection**을 도입하고, 학습 분기가 \(F(x)\)라는 변화를 만들게 합니다.
3. 이때 블록의 핵심 계산은 \(H(x)=F(x)+x\)이며, 항등 변환이 필요하면 잔차 분기가 0에 가까워지는 방향으로 학습할 수 있습니다.
4. 입력·출력 shape가 다르면 shortcut에도 projection 등 적절한 변환을 적용해야 합니다.
5. 원 논문은 ImageNet과 CIFAR-10 등의 실험을 통해 깊은 잔차 신경망이 깊은 plain network보다 더 잘 최적화될 수 있음을 보였습니다.

## 한 눈에 남길 공식

\[
\boxed{F(x)=H(x)-x \qquad \Longleftrightarrow \qquad H(x)=F(x)+x}
\]

이 수식을 볼 때는 **`F(x)`가 전체 정답 자체가 아니라 입력에 더할 변화를 학습한다**고 읽으면 됩니다.

## 논문을 떠올리게 만드는 숫자

원문 Table 2, ImageNet validation **10-crop Top-1 error**:

| | Plain-34 | ResNet-34 |
|---|---:|---:|
| 오류율 | 28.54% | 25.03% |

차이는 **3.51%p**입니다. 최적화 문제의 근거를 온전히 파악하려면 **Figure 4의 훈련 곡선**도 함께 봐야 합니다.

## 스스로 써 볼 1분 설명

> 더 깊은 네트워크가 항상 더 잘 학습되는 것은 아닙니다. 일반적인 네트워크에서는 깊이가 늘면서 훈련 오류조차 커질 수 있습니다. ResNet은 입력을 바로 전달하는 shortcut을 만들고, 중간 블록이 기존 입력을 얼마만큼 바꾸어야 하는지 학습하게 합니다. 즉 \(H(x)=F(x)+x\) 구조입니다. 차원이 다를 때는 shortcut도 변환해 크기를 맞춥니다. 논문에서는 이런 구조가 특히 더 깊은 네트워크의 최적화를 돕는 결과를 보였습니다.

이 문단을 외우기보다는 **자신만의 말**로 다시 적어 보세요.

## 다음으로 할 일

- [학습 진행표](../../progress.md)에서 완료한 항목을 체크합니다.
- [실험 기록](../../experiments/resnet-first-run.md) 양식을 채워 직접 확인한 결과를 남깁니다.
- [전체 로드맵](../../roadmap.md)에서 다음 논문을 선택합니다.

### 연결해서 생각해 볼 주제

- 왜 Batch Normalization과 Residual Connection을 함께 사용할까?
- ResNet v1의 덧셈 후 ReLU와, 이후 연구에서 등장한 **pre-activation** 구조는 어떤 차이가 있을까?
- 분류뿐 아니라 U-Net 계열의 segmentation 구조에서도 skip connection은 어떤 역할을 할까? **모든 skip connection이 덧셈인 것은 아닙니다.**

## 출처

- He, Kaiming et al. (2016), [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385), CVPR 2016.
- [원 논문 코드 및 모델 저장소](https://github.com/KaimingHe/deep-residual-networks)

이 사이트의 설명과 도식은 학습을 위해 새로 작성한 내용입니다. 원 논문 그림과 실험 결과의 저작권과 출처는 저자에게 있습니다.
