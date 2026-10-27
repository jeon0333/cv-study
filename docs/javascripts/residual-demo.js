(function () {
  'use strict';
  function init() {
    const widget = document.querySelector('[data-residual-demo]');
    if (!widget) return;
    const xField = widget.querySelector('[data-residual-x]');
    const targetField = widget.querySelector('[data-residual-target]');
    const result = widget.querySelector('[data-residual-result]');
    const update = () => {
      const x = Number(xField.value);
      const target = Number(targetField.value);
      if (!Number.isFinite(x) || !Number.isFinite(target) || !xField.value || !targetField.value) {
        result.textContent = '유효한 숫자를 입력해 주세요.';
        return;
      }
      const delta = target - x;
      result.textContent = `그대로 보내는 값 x = ${x}; 추가로 배울 변화량 F(x) = ${delta}; 결과 F(x) + x = ${target}`;
    };
    xField.oninput = update;
    targetField.oninput = update;
    update();
  }
  if (typeof document$ !== 'undefined') document$.subscribe(init);
  else document.addEventListener('DOMContentLoaded', init);
})();
