/**
 * Interactive Dashboard JavaScript Engine for Fire & Smoke Detection v2
 */

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initDropzone();
  initLiveFeed();
  initAlertHistory();
  initAudioAlarm();
});

/* -------------------------------------------------------------------------- */
/* Tab Switching System                                                      */
/* -------------------------------------------------------------------------- */
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');

      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const target = document.getElementById(tabId);
      if (target) target.classList.add('active');

      if (tabId === 'history-tab') {
        fetchHistory();
      }
    });
  });
}

/* -------------------------------------------------------------------------- */
/* Image Upload & Grad-CAM Heatmap Viewer                                    */
/* -------------------------------------------------------------------------- */
function initDropzone() {
  const dropzone = document.getElementById('image-dropzone');
  const fileInput = document.getElementById('image-file-input');

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    }, false);
  });

  dropzone.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length > 0) {
      handleImageUpload(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
      handleImageUpload(fileInput.files[0]);
    }
  });
}

async function handleImageUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  const loader = document.getElementById('predict-loader');
  const resultsBox = document.getElementById('predict-results');

  if (loader) loader.style.display = 'block';
  if (resultsBox) resultsBox.style.display = 'none';

  try {
    const response = await fetch('/api/v1/predict/image', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (loader) loader.style.display = 'none';

    if (data.status === 'success') {
      displayResults(data.result);
      if (data.result.is_fire) {
        triggerAudioAlert();
      }
    } else {
      alert(`Error: ${data.message || 'Prediction failed'}`);
    }
  } catch (err) {
    if (loader) loader.style.display = 'none';
    console.error('Upload Error:', err);
    alert('Failed to connect to detection server.');
  }
}

function displayResults(res) {
  const resultsBox = document.getElementById('predict-results');
  if (!resultsBox) return;

  resultsBox.style.display = 'block';

  document.getElementById('res-label').innerText = res.label;
  document.getElementById('res-confidence').innerText = `${res.confidence}%`;
  document.getElementById('res-original-img').src = res.image_url;
  document.getElementById('res-heatmap-img').src = res.gradcam_url;

  const badge = document.getElementById('res-badge');
  if (badge) {
    badge.className = res.is_fire ? 'badge badge-fire' : 'badge badge-safe';
    badge.innerText = res.is_fire ? '🔥 FIRE DETECTED' : '✅ SAFE / NORMAL';
  }

  document.getElementById('res-hsv-score').innerText = `${(res.hsv_fire_ratio * 100).toFixed(1)}%`;
  let guardStatus = 'Standard';
  if (res.face_suppressed) guardStatus = 'Active (Face Suppressed)';
  else if (res.sunset_guard) guardStatus = 'Active (Sunset / Landscape Guard)';
  document.getElementById('res-guard-status').innerText = guardStatus;
}

/* -------------------------------------------------------------------------- */
/* Web Audio API Alarm Synthesizer                                            */
/* -------------------------------------------------------------------------- */
let audioCtx = null;
let soundEnabled = true;

function initAudioAlarm() {
  const toggleBtn = document.getElementById('toggle-alarm-btn');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      soundEnabled = !soundEnabled;
      toggleBtn.innerText = soundEnabled ? '🔊 Alarm: ON' : '🔇 Alarm: OFF';
      toggleBtn.className = soundEnabled ? 'btn btn-secondary' : 'btn btn-secondary muted';
    });
  }
}

function triggerAudioAlert() {
  if (!soundEnabled) return;
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(880, audioCtx.currentTime); // A5 note
    osc.frequency.exponentialRampToValueAtTime(440, audioCtx.currentTime + 0.4);

    gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.4);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start();
    osc.stop(audioCtx.currentTime + 0.4);
  } catch (e) {
    console.log('Audio playback prevented or unsupported');
  }
}

/* -------------------------------------------------------------------------- */
/* Live Webcam Stream & HUD                                                  */
/* -------------------------------------------------------------------------- */
function initLiveFeed() {
  const startBtn = document.getElementById('start-webcam-btn');
  const stopBtn = document.getElementById('stop-webcam-btn');
  const feedImg = document.getElementById('webcam-feed-img');
  const placeholder = document.getElementById('webcam-placeholder');

  if (!startBtn || !feedImg) return;

  startBtn.addEventListener('click', () => {
    feedImg.src = '/video_feed';
    feedImg.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
    startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'inline-flex';
  });

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      feedImg.src = '';
      feedImg.style.display = 'none';
      if (placeholder) placeholder.style.display = 'flex';
      stopBtn.style.display = 'none';
      startBtn.style.display = 'inline-flex';
    });
  }
}

/* -------------------------------------------------------------------------- */
/* Event History Log Table & CSV Export                                       */
/* -------------------------------------------------------------------------- */
async function fetchHistory() {
  const filterSelect = document.getElementById('history-filter');
  const label = filterSelect ? filterSelect.value : 'all';

  try {
    const res = await fetch(`/api/v1/history?label=${label}`);
    const data = await res.json();
    if (data.status === 'success') {
      renderHistoryTable(data.history);
    }
  } catch (err) {
    console.error('Failed to fetch history:', err);
  }
}

function renderHistoryTable(events) {
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;

  if (events.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--text-muted);">No detection events logged yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = events.map(e => `
    <tr>
      <td>#${e.id}</td>
      <td>${e.timestamp}</td>
      <td>
        <span class="${e.is_alert ? 'badge badge-fire' : 'badge badge-safe'}">
          ${e.label}
        </span>
      </td>
      <td>${e.confidence}%</td>
      <td>${e.source}</td>
    </tr>
  `).join('');
}

document.addEventListener('change', (e) => {
  if (e.target && e.target.id === 'history-filter') {
    fetchHistory();
  }
});
