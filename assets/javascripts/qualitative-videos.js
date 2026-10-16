/* Load original-quality videos only after an explicit playback request. */
(() => {
  const demo = document.querySelector('[data-video-demo]');
  if (!demo) return;
  const player = demo.querySelector('video');
  const play = demo.querySelector('[data-play-video]');
  const status = demo.querySelector('[data-video-status]');
  const title = demo.querySelector('[data-video-title]');
  const caption = demo.querySelector('[data-video-caption]');
  const original = demo.querySelector('[data-original-video]');
  const choices = [...demo.querySelectorAll('[data-video-src]')];
  let selected;
  let generation = 0;

  function select(choice) {
    generation += 1;
    selected = choice;
    player.pause();
    player.removeAttribute('src');
    player.load();
    player.controls = false;
    player.poster = choice.dataset.poster;
    title.textContent = `${choice.dataset.sample} — ${choice.dataset.title}`;
    caption.textContent = choice.dataset.caption;
    original.href = choice.dataset.videoSrc;
    play.textContent = 'Play video';
    play.setAttribute('aria-label', `Play ${choice.dataset.sample}: ${choice.dataset.title}`);
    play.hidden = false;
    status.textContent = '';
    choices.forEach(button => button.setAttribute('aria-pressed', String(button === choice)));
  }

  function showError(message) {
    status.textContent = message;
    play.textContent = 'Try again';
    play.setAttribute('aria-label', `Retry ${selected.dataset.sample}`);
    play.hidden = false;
  }

  play.addEventListener('click', async () => {
    const request = generation;
    status.textContent = 'Loading video…';
    play.hidden = true;
    player.controls = true;
    player.src = selected.dataset.videoSrc;
    try {
      await player.play();
    } catch (error) {
      if (request !== generation || error.name === 'AbortError') return;
      showError('Playback could not start. Try again or open the original video.');
    }
  });
  player.addEventListener('playing', () => {
    status.textContent = '';
    play.hidden = true;
  });
  player.addEventListener('error', () => {
    if (player.getAttribute('src') === selected.dataset.videoSrc) {
      showError('The video could not be loaded. Try again or open the original video.');
    }
  });
  choices.forEach(choice => choice.addEventListener('click', () => select(choice)));
  select(choices[0]);
  demo.querySelector('[data-player-stage]').hidden = false;
  demo.querySelector('[data-video-selector]').hidden = false;
})();
