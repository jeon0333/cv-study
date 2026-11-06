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
    const sidebar = document.querySelector('.md-sidebar--primary');
    const header = document.querySelector('.md-header__inner');
    if (sidebar && header) {
      sidebar.id = 'study-navigation';
      sidebar.querySelectorAll('label[for^="__nav_"]').forEach(label => {
        const input = document.getElementById(label.htmlFor);
        label.setAttribute('role', 'button');
        label.tabIndex = 0;
        label.setAttribute('aria-label', label.parentElement.textContent.trim() + ' 하위 목차');
        const updateSection = () => label.setAttribute('aria-expanded', String(input.checked));
        input.onchange = updateSection;
        label.onkeydown = event => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            input.click();
          }
        };
        updateSection();
      });
      let button = document.querySelector('.study-nav-toggle');
      if (!button) {
        button = document.createElement('button');
        button.type = 'button';
        button.className = 'study-nav-toggle';
        button.setAttribute('aria-controls', sidebar.id);
        header.prepend(button);
      }
      const updateNavigation = () => {
        const collapsed = document.documentElement.classList.contains('study-nav-collapsed');
        button.textContent = collapsed ? '☰ 목차 펼치기' : '☰ 목차 접기';
        button.setAttribute('aria-expanded', String(!collapsed));
      };
      button.onclick = () => {
        document.documentElement.classList.toggle('study-nav-collapsed');
        updateNavigation();
      };
      updateNavigation();
    }
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
