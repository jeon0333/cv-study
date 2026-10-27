(function () {
  'use strict';
  const STORAGE_KEY = 'cv-deep-study-progress-v1';
  function getProgress() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); }
    catch (error) { return {}; }
  }
  function saveProgress(data) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(data)); }
    catch (error) { /* 저장이 차단된 브라우저에서는 현재 페이지의 체크만 사용 */ }
  }
  function init() {
    const checks = [...document.querySelectorAll('input[data-study-check]')];
    if (!checks.length) return;
    const saved = getProgress();
    checks.forEach(el => {
      el.checked = Boolean(saved[el.dataset.studyCheck]);
      el.onchange = () => {
        saved[el.dataset.studyCheck] = el.checked;
        saveProgress(saved);
        update();
      };
    });
    function update() {
      const count = checks.filter(el => el.checked).length;
      const percent = checks.length ? Math.round(count / checks.length * 100) : 0;
      document.querySelectorAll('[data-progress-count]').forEach(el => { el.textContent = `${count} / ${checks.length} 완료`; });
      document.querySelectorAll('[data-progress-percent]').forEach(el => { el.textContent = `${percent}%`; });
      document.querySelectorAll('[data-progress-bar]').forEach(el => { el.style.width = `${percent}%`; el.setAttribute('aria-valuenow', String(percent)); });
    }
    update();
  }
  if (typeof document$ !== 'undefined') document$.subscribe(init);
  else document.addEventListener('DOMContentLoaded', init);
})();
