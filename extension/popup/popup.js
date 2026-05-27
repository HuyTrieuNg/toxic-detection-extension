/**
 * Popup Script — VOZ Toxic Detector
 * Điều khiển UI popup: toggle, model selector, stats, rescan.
 */

const $ = (id) => document.getElementById(id);

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let currentState = {
  enabled: true,
  selectedModel: 'bilstm',
  apiUrl: 'http://localhost:8000/api',
};

let isVozThread = false;

// ─────────────────────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  await loadState();
  await checkCurrentTab();
  renderUI();
  attachEventListeners();
  await fetchStats();
  await checkApiHealth();
});

// ─────────────────────────────────────────────────────────────
// Load state từ chrome.storage
// ─────────────────────────────────────────────────────────────
async function loadState() {
  return new Promise((resolve) => {
    chrome.storage.local.get(['enabled', 'selectedModel', 'apiUrl'], (data) => {
      currentState.enabled = data.enabled ?? true;
      currentState.selectedModel = data.selectedModel ?? 'bilstm';
      currentState.apiUrl = data.apiUrl ?? 'http://localhost:8000/api';
      resolve();
    });
  });
}

// ─────────────────────────────────────────────────────────────
// Kiểm tra tab hiện tại có phải VOZ thread không
// ─────────────────────────────────────────────────────────────
async function checkCurrentTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  isVozThread = tab?.url?.match(/https:\/\/voz\.vn\/t\/[^/]+/) != null;
}

// ─────────────────────────────────────────────────────────────
// Render UI theo state
// ─────────────────────────────────────────────────────────────
function renderUI() {
  // Toggle
  $('toggleEnabled').checked = currentState.enabled;

  // Model radio
  const radio = document.querySelector(`input[name="model"][value="${currentState.selectedModel}"]`);
  if (radio) radio.checked = true;

  // Model cards disabled khi extension tắt
  updateModelCardsState();

  // Show/hide warning
  if (!isVozThread) {
    $('notThreadWarning').style.display = 'block';
    $('btnRescan').disabled = true;
  }

  // API URL display
  const urlDisplay = currentState.apiUrl.replace('http://', '').replace('/api', '');
  $('apiUrlDisplay').textContent = urlDisplay;
}

function updateModelCardsState() {
  const disabled = !currentState.enabled;
  document.querySelectorAll('input[name="model"]').forEach((radio) => {
    radio.disabled = disabled;
  });
  document.querySelectorAll('.model-card').forEach((card) => {
    card.style.opacity = disabled ? '0.5' : '1';
    card.style.pointerEvents = disabled ? 'none' : 'auto';
  });
  $('btnRescan').disabled = disabled || !isVozThread;
}

// ─────────────────────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────────────────────
function attachEventListeners() {
  // Toggle enabled
  $('toggleEnabled').addEventListener('change', async (e) => {
    currentState.enabled = e.target.checked;
    await saveState();
    updateModelCardsState();
    notifyContentScript({ type: 'STATE_CHANGED', payload: currentState });
  });

  // Model selection
  document.querySelectorAll('input[name="model"]').forEach((radio) => {
    radio.addEventListener('change', async (e) => {
      const prevModel = currentState.selectedModel;
      currentState.selectedModel = e.target.value;
      await saveState();

      if (prevModel !== currentState.selectedModel && isVozThread) {
        setStatus('loading', `Đang chuyển sang ${getModelDisplayName(currentState.selectedModel)}...`);
        notifyContentScript({
          type: 'STATE_CHANGED',
          payload: currentState,
        });
        // Reset stats while re-scanning
        updateStats({ total: 0, toxic: 0, nontoxic: 0, avg_ms_per_comment: 0 });
      }
    });
  });

  // Rescan button
  $('btnRescan').addEventListener('click', async () => {
    if (!isVozThread) return;
    $('btnRescan').disabled = true;
    $('btnRescan').innerHTML = '<span class="btn-icon">⏳</span> Đang quét...';
    setStatus('loading', 'Đang quét bình luận...');
    updateStats({ total: 0, toxic: 0, nontoxic: 0, avg_ms_per_comment: 0 });
    await notifyContentScript({ type: 'RESCAN' });
    // Button sẽ được reset khi nhận STATS_UPDATE từ content script
  });
}

// ─────────────────────────────────────────────────────────────
// Lấy stats từ content script
// ─────────────────────────────────────────────────────────────
async function fetchStats() {
  if (!isVozThread) return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return;

  chrome.tabs.sendMessage(tab.id, { type: 'GET_STATS' }, (response) => {
    if (chrome.runtime.lastError || !response) return;
    updateStats(response);
  });
}

// ─────────────────────────────────────────────────────────────
// Update stats UI
// ─────────────────────────────────────────────────────────────
function formatMs(ms) {
  if (!ms || ms === 0) return '—';
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  return `${ms.toFixed(1)} ms`;
}

function updateStats({ total, toxic, nontoxic, avg_ms_per_comment }) {
  $('statTotal').textContent = total ?? '—';
  $('statToxic').textContent = toxic ?? '—';
  $('statSafe').textContent = nontoxic ?? '—';
  $('statAvgTime').textContent = formatMs(avg_ms_per_comment ?? 0);

  if (total > 0) {
    const pct = Math.round((toxic / total) * 100);
    $('toxicPercent').textContent = `${pct}%`;
    $('toxicBarFill').style.width = `${pct}%`;
    $('toxicBarWrap').style.display = 'block';
  }
}

// ─────────────────────────────────────────────────────────────
// API health check
// ─────────────────────────────────────────────────────────────
async function checkApiHealth() {
  setStatus('loading', 'Đang kết nối API...');
  try {
    const response = await fetch(`${currentState.apiUrl}/health/`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000),
    });
    if (response.ok) {
      const data = await response.json();
      const loadedModels = Object.entries(data.models || {})
        .filter(([, v]) => v)
        .map(([k]) => k)
        .join(', ');
      setStatus('ok', loadedModels ? `Model đã tải: ${loadedModels}` : 'API sẵn sàng');
    } else {
      setStatus('error', `API lỗi: HTTP ${response.status}`);
    }
  } catch (err) {
    setStatus('error', 'Không kết nối được API. Server đang chạy chưa?');
  }
}

// ─────────────────────────────────────────────────────────────
// Status banner
// ─────────────────────────────────────────────────────────────
function setStatus(type, message) {
  const dot = $('statusDot');
  const text = $('statusText');
  dot.className = `status-dot status--${type}`;
  text.textContent = message;
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────
async function saveState() {
  return new Promise((resolve) => {
    chrome.storage.local.set({
      enabled: currentState.enabled,
      selectedModel: currentState.selectedModel,
      apiUrl: currentState.apiUrl,
    }, resolve);
  });
}

async function notifyContentScript(message, callback) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    if (callback) callback(null);
    return;
  }
  chrome.tabs.sendMessage(tab.id, message, (response) => {
    // Bỏ qua lỗi nếu content script chưa inject (tab mới mở)
    if (chrome.runtime.lastError) {
      console.warn('[Popup] Content script chưa sẵn sàng:', chrome.runtime.lastError.message);
    }
    if (callback) callback(response);
  });
}

function getModelDisplayName(model) {
  return model === 'bilstm' ? 'BiLSTM' : 'PhoBERT';
}

// ─────────────────────────────────────────────────────────────
// Listen for stats updates from content script (via background)
// ─────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'STATS_UPDATE') {
    updateStats(message.payload);
    setStatus('ok', 'Hoàn thành quét trang');
    $('btnRescan').disabled = false;
    $('btnRescan').innerHTML = '<span class="btn-icon">🔍</span> Quét lại trang';
  }
});
