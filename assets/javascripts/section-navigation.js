/* Enhance native anchor navigation without replacing browser history. */
(() => {
  const outline = document.querySelector('.section-navigation');
  if (!outline) return;

  const summary = outline.querySelector('summary');
  const links = [...outline.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => document.getElementById(link.hash.slice(1)));
  const wideScreen = window.matchMedia('(min-width: 1680px)');
  let framePending = false;

  function updateLayout() {
    const focusInside = outline.contains(document.activeElement);
    outline.open = wideScreen.matches;
    if (focusInside) {
      (wideScreen.matches ? links[0] : summary).focus({ preventScroll: true });
    }
  }

  function closeOutline() {
    if (wideScreen.matches || !outline.open) return;
    if (outline.contains(document.activeElement)) {
      summary.focus({ preventScroll: true });
    }
    outline.open = false;
  }

  function updateCurrentSection() {
    framePending = false;
    // The next heading ends a section, including any sibling tables or videos.
    const readingLine = Math.max(64, window.innerHeight * 0.2);
    let current = -1;
    sections.forEach((section, index) => {
      if (section && section.getBoundingClientRect().top <= readingLine) {
        current = index;
      }
    });
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      current = sections.length - 1;
    }
    links.forEach((link, index) => {
      if (index === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }

  function scheduleUpdate() {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(updateCurrentSection);
  }

  links.forEach((link, index) => {
    link.addEventListener('click', event => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      closeOutline();
      const heading = sections[index]?.querySelector('h2');
      if (heading) {
        heading.setAttribute('tabindex', '-1');
        heading.focus({ preventScroll: true });
      }
      // Let the anchor perform the scroll and create its normal history entry.
    });
  });
  document.addEventListener('click', event => {
    if (!outline.contains(event.target)) closeOutline();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !wideScreen.matches && outline.open) {
      closeOutline();
    }
  });
  wideScreen.addEventListener('change', updateLayout);
  window.addEventListener('scroll', scheduleUpdate, { passive: true });
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('hashchange', scheduleUpdate);
  window.addEventListener('pageshow', scheduleUpdate);
  window.addEventListener('load', scheduleUpdate);
  if ('ResizeObserver' in window) {
    new ResizeObserver(scheduleUpdate).observe(document.querySelector('main'));
  }
  updateLayout();
  outline.setAttribute('data-enhanced', '');
  updateCurrentSection();
})();
