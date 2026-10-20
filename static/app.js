/**
 * Simple Fake News Detector — Frontend Logic
 * Direct text submission, fast detection, and safe DOM rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  const newsInput = document.getElementById('news-input');
  const charCount = document.getElementById('char-count');
  const wordCount = document.getElementById('word-count');
  const btnDetect = document.getElementById('btn-detect');
  const btnClear = document.getElementById('btn-clear');
  const btnSampleReal = document.getElementById('btn-sample-real');
  const btnSampleFake = document.getElementById('btn-sample-fake');

  // Views
  const viewIdle = document.getElementById('view-idle');
  const viewLoading = document.getElementById('view-loading');
  const viewError = document.getElementById('view-error');
  const viewVerdict = document.getElementById('view-verdict');
  const errorHeading = document.getElementById('error-heading');
  const errorText = document.getElementById('error-text');

  // Verdict elements
  const verdictBanner = document.getElementById('verdict-banner');
  const verdictBadge = document.getElementById('verdict-badge');
  const confidenceBadge = document.getElementById('confidence-badge');
  const shortWarning = document.getElementById('short-warning');
  const shortWarningText = document.getElementById('short-warning-text');
  const pctReal = document.getElementById('pct-real');
  const pctFake = document.getElementById('pct-fake');
  const meterFillReal = document.getElementById('meter-fill-real');
  const meterFillFake = document.getElementById('meter-fill-fake');
  const signalsGrid = document.getElementById('signals-grid');

  // Sample articles
  const SAMPLES = {
    real: "WASHINGTON — The United States Senate passed a bipartisan infrastructure bill on Tuesday following weeks of negotiations between Democratic and Republican lawmakers. The legislation allocates $550 billion in new federal investments for highways, bridges, public transit, and clean drinking water facilities. Lawmakers voted 69 to 30 to approve the measure.",
    fake: "SHOCKING PROOF: Secret Globalist Cabal Caught Red-Handed Poisoning City Water Supply with Mind Control Chemicals! An explosive new report that mainstream media refuses to cover exposes corrupt politicians paid millions in offshore kickbacks. SHARE THIS EVERYWHERE before Twitter and Facebook delete it forever!"
  };

  // Update text counter
  function updateCounters() {
    const text = newsInput.value;
    const chars = text.length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;

    charCount.textContent = `${chars.toLocaleString()} characters`;
    wordCount.textContent = `${words.toLocaleString()} words`;
  }

  newsInput.addEventListener('input', updateCounters);

  // Quick Samples
  btnSampleReal.addEventListener('click', () => {
    newsInput.value = SAMPLES.real;
    updateCounters();
    newsInput.focus();
  });

  btnSampleFake.addEventListener('click', () => {
    newsInput.value = SAMPLES.fake;
    updateCounters();
    newsInput.focus();
  });

  // Clear Button
  btnClear.addEventListener('click', () => {
    newsInput.value = '';
    updateCounters();
    setViewState('idle');
    newsInput.focus();
  });

  // State switcher
  function setViewState(state) {
    viewIdle.hidden = true;
    viewLoading.hidden = true;
    viewError.hidden = true;
    viewVerdict.hidden = true;

    if (state === 'idle') viewIdle.hidden = false;
    else if (state === 'loading') viewLoading.hidden = false;
    else if (state === 'error') viewError.hidden = false;
    else if (state === 'verdict') viewVerdict.hidden = false;
  }

  function showError(heading, message) {
    errorHeading.textContent = heading;
    errorText.textContent = message;
    setViewState('error');
  }

  // Detect Action
  btnDetect.addEventListener('click', async () => {
    const text = newsInput.value.trim();

    if (!text) {
      showError('Empty Input', 'Please type or paste news text into the box.');
      return;
    }

    const words = text.split(/\s+/).length;
    if (words < 3) {
      showError('Text Too Short', 'Please enter at least 3 words (e.g. a complete headline).');
      return;
    }

    btnDetect.disabled = true;
    btnDetect.classList.add('loading');
    setViewState('loading');

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server returned error (${response.status})`);
      }

      displayResult(data);
    } catch (err) {
      showError('Detection Error', err.message);
    } finally {
      btnDetect.disabled = false;
      btnDetect.classList.remove('loading');
    }
  });

  // Display Result
  function displayResult(data) {
    setViewState('verdict');

    const verdict = data.verdict || 'Unclear';
    const confidence = Math.round((data.confidence || 0.5) * 100);
    const realProb = Math.round((data.real_probability !== undefined ? data.real_probability : 0.5) * 100);
    const fakeProb = Math.round((data.fake_probability !== undefined ? data.fake_probability : 0.5) * 100);

    // Banner & Badge
    verdictBanner.className = 'verdict-banner';
    if (verdict === 'Real') {
      verdictBanner.classList.add('verdict-real');
      verdictBadge.textContent = 'REAL NEWS';
    } else if (verdict === 'Fake') {
      verdictBanner.classList.add('verdict-fake');
      verdictBadge.textContent = 'FAKE NEWS';
    } else {
      verdictBanner.classList.add('verdict-unclear');
      verdictBadge.textContent = 'UNCLEAR';
    }

    confidenceBadge.textContent = `${confidence}% Confidence`;

    // Short warning
    if (data.warning) {
      shortWarningText.textContent = data.warning;
      shortWarning.hidden = false;
    } else {
      shortWarning.hidden = true;
    }

    // Probability Meter
    pctReal.textContent = `${realProb}%`;
    pctFake.textContent = `${fakeProb}%`;
    meterFillReal.style.width = `${realProb}%`;
    meterFillFake.style.width = `${fakeProb}%`;

    // Influential Words
    signalsGrid.replaceChildren();

    const signals = data.signals || [];
    if (signals.length === 0) {
      const p = document.createElement('p');
      p.className = 'signals-sub';
      p.textContent = 'No dominant keywords detected in this text.';
      signalsGrid.appendChild(p);
    } else {
      signals.forEach(item => {
        const row = document.createElement('div');
        row.className = 'signal-item';

        const wordSpan = document.createElement('span');
        wordSpan.className = 'signal-word';
        wordSpan.textContent = item.word;
        row.appendChild(wordSpan);

        const track = document.createElement('div');
        track.className = 'signal-bar-track';
        const fill = document.createElement('div');
        const isFake = item.points_to === 'fake';
        fill.className = `signal-bar-fill ${isFake ? 'fill-fake' : 'fill-real'}`;
        fill.style.width = `${Math.round((item.weight || 0.1) * 100)}%`;
        track.appendChild(fill);
        row.appendChild(track);

        const tag = document.createElement('span');
        tag.className = `signal-tag ${isFake ? 'tag-fake' : 'tag-real'}`;
        tag.textContent = isFake ? 'Fake' : 'Real';
        row.appendChild(tag);

        signalsGrid.appendChild(row);
      });
    }
  }

  // Initial counter call
  updateCounters();
});
