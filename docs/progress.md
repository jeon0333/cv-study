# ResNet 학습 진행표

브라우저에서 공부한 부분을 체크할 수 있습니다. 체크 상태는 **현재 브라우저의 로컬 저장 공간**에 보관되며, GitHub 계정이나 다른 기기와 자동으로 동기화되지는 않습니다.

<div class="study-progress">
  <strong data-progress-count>0 / 8 완료</strong> <span style="float:right" data-progress-percent>0%</span>
  <div class="progress-rail"><span class="progress-fill" role="progressbar" aria-label="학습 진도" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0" data-progress-bar></span></div>
  <ul class="study-checklist">
    <li><label><input type="checkbox" data-study-check="cnn"> <span><a href="../basics/cnn/">CNN의 입력·출력과 합성곱</a>을 설명할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="optimization"> <span><a href="../basics/optimization/">훈련 오류와 성능 저하 문제</a>를 구분할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="problem"> <span><a href="../papers/resnet/01-problem/">ResNet이 해결하는 문제</a>를 한 문장으로 말할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="residual"> <span><a href="../papers/resnet/02-residual/">H(x)=F(x)+x</a>를 숫자와 수식으로 설명할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="architecture"> <span><a href="../papers/resnet/03-architecture/">BasicBlock과 Bottleneck</a>의 차이를 말할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="results"> <span><a href="../papers/resnet/04-results/">논문 Table 2</a>를 정확하게 해석할 수 있다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="lab"> <span><a href="../papers/resnet/05-lab/">PyTorch 블록 테스트</a>를 실행하고 결과를 확인했다.</span></label></li>
    <li><label><input type="checkbox" data-study-check="quiz"> <span><a href="../papers/resnet/06-quiz/">이해도 확인 문제</a>를 답을 보지 않고 풀었다.</span></label></li>
  </ul>
</div>

!!! info "학습 기록을 오래 보존하려면"
    체크 상태는 브라우저 데이터 삭제 시 사라질 수 있습니다. 중요한 질문과 실험 결과는 [실험 노트](experiments/index.md) 또는 별도의 Markdown 파일에 직접 적어 GitHub에 커밋하세요.

## 스스로 확인할 마지막 질문

- 다른 사람에게 ResNet을 1분 동안 설명한다면 어떤 순서로 설명할까요?
- 왜 더 깊은 plain network의 **훈련 오류**가 오히려 높아진 것일까요?
- `out += identity`를 실행하기 위한 shape 조건은 무엇일까요?

[단원 전체 정리 보러 가기](papers/resnet/07-review.md)
