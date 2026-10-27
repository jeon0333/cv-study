# 01 · ResNet — 깊은 네트워크를 학습시키는 방법

<div class="reading-meta">
  <span class="reading-pill">원문: He et al., 2015 / CVPR 2016</span>
  <span class="reading-pill">주제: 이미지 분류</span>
  <span class="reading-pill">핵심: Residual Learning</span>
  <span class="reading-pill">실습: PyTorch</span>
</div>

> **한 줄 소개:** 네트워크에 층을 더 쌓았는데 훈련 성능이 떨어지는 이유를 살펴보고, 입력을 건너뛰어 더하는 *Residual Connection*이 학습을 어떻게 돕는지 이해합니다.

- 논문: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- 저자: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- 발표: arXiv 2015, CVPR 2016 ([공식 공개본](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html))

## 이 논문을 왜 읽나요?

이미지 인식 모델의 성능을 높이려면 더 깊은 신경망이 유리할 것 같습니다. 그러나 **더 많은 층을 가진 모델이 실제로 더 잘 학습되리라는 보장은 없습니다.** ResNet은 이 어려움을 설명하고 해결 방향을 제시한 중요한 연구입니다.

## 연구를 구성하는 여섯 요소

| 항목 | 이 논문에서의 의미 |
|---|---|
| **문제 (Problem)** | 층 수를 늘렸을 때 최적화가 어려워져 훈련 오류도 악화되는 현상 |
| **입력 (Input)** | 이미지 텐서. 예를 들어 ImageNet의 RGB 이미지 |
| **출력 (Output)** | 이미지의 클래스에 대한 예측 점수 |
| **모델의 역할** | 합성곱 신경망을 통해 특징을 추출하고 클래스 점수 계산 |
| **학습 목표 (Loss)** | 이미지 분류의 손실을 최소화하도록 가중치를 학습. 논문 ImageNet 학습은 분류 문제의 SGD 최적화를 사용 |
| **주의·한계** | Shortcut만 추가한다고 모든 문제에서 성능이 보장되지는 않음. 정확한 효과는 구조·훈련 설정에 의존 |

## 학습할 순서

| 순서 | 질문 | 페이지 |
|---|---|---|
| 1 | 네트워크는 깊을수록 왜 더 어려워질까? | [문제 정의](01-problem.md) |
| 2 | 왜 `H(x)` 대신 `F(x)`를 학습할까? | [Residual Connection](02-residual.md) |
| 3 | 18층·34층·50층 ResNet은 어떻게 다른가? | [구조 분석](03-architecture.md) |
| 4 | 논문 속 실험은 무엇을 증명하는가? | [실험 결과 읽기](04-results.md) |
| 5 | 블록을 직접 구현하면 무엇을 검사해야 하나? | [PyTorch 실습](05-lab.md) |
| 6 | 정말 이해했는가? | [확인 문제](06-quiz.md) |
| 7 | 어떻게 기억하고 다음 연구로 연결할까? | [정리와 확장](07-review.md) |

## 시작하기 전 선행 지식

**필수:** [CNN 기초](../../basics/cnn.md)에서 `channel`, `Conv2d`, `(N,C,H,W)`를 이해하고 [최적화와 역전파](../../basics/optimization.md)에서 `훈련 오류`와 `검증 오류`를 구분할 수 있으면 됩니다. PyTorch를 처음 접한다면 [코드 읽기 안내](../../basics/pytorch.md)를 먼저 읽으세요.

!!! tip "단원 완료 기준"
    - `degradation problem`을 과적합과 구분할 수 있다.
    - `H(x) = F(x) + x`를 구체적인 숫자 예제로 설명할 수 있다.
    - `out += identity`에 들어오는 두 텐서의 shape를 맞출 수 있다.
    - 원문 Table 2에서 주장과 근거를 찾아 제시할 수 있다.
    - ResNet 블록을 구현하고 기울기가 계산되는지 확인할 수 있다.

**첫 페이지로 이동:** [1. 왜 깊기만 해서는 안 될까?](01-problem.md)
