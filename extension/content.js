/**
 * Content Script — VOZ.vn Thread Page
 *
 * Chạy trên trang https://voz.vn/t/*
 * Thu thập bình luận → gọi API TRỰC TIẾP (không qua background) → gán nhãn.
 *
 * HTML Selectors (VOZ.vn XenForo 2):
 *   - Container bình luận : article.message.message--post
 *   - Nội dung text       : div[itemprop="text"] .bbWrapper
 *   - Post ID             : article[data-content]  (e.g. "post-42156356")
 *   - Author              : article[data-author]
 */

const API_URL_DEFAULT = 'http://localhost:8000/api';
const BATCH_SIZE = 20;
const SCAN_DEBOUNCE_MS = 1000;

// ─────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────
let extensionEnabled = true;
let selectedModel = 'bilstm';
let apiUrl = API_URL_DEFAULT;
let scanInProgress = false;
let labeledPostIds = new Set();
let stats = { total: 0, toxic: 0, nontoxic: 0 };

// ─────────────────────────────────────────────────────────────
// Init — đọc state từ storage rồi bắt đầu scan
// ─────────────────────────────────────────────────────────────
async function init() {
  try {
    // Đọc trực tiếp từ chrome.storage thay vì relay qua background worker
    // → tránh lỗi khi service worker đang ngủ
    const stored = await chrome.storage.local.get({
      enabled: true,
      selectedModel: 'bilstm',
      apiUrl: API_URL_DEFAULT,
    });
    extensionEnabled = stored.enabled;
    selectedModel = stored.selectedModel;
    apiUrl = stored.apiUrl;
  } catch (err) {
    console.warn('[VOZ Toxic Detector] Không đọc được storage, dùng mặc định:', err);
  }

  if (!extensionEnabled) {
    console.log('[VOZ Toxic Detector] Extension đang tắt.');
    return;
  }

  console.log(`[VOZ Toxic Detector] Khởi động — model: ${selectedModel}, api: ${apiUrl}`);
  await scanAllComments();
  observeNewComments();
}

// ─────────────────────────────────────────────────────────────
// Message listener — nhận lệnh từ popup
// ─────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'RESCAN') {
    labeledPostIds.clear();
    stats = { total: 0, toxic: 0, nontoxic: 0 };
    removeAllLabels();
    scanAllComments().then(() => sendResponse({ success: true }));
    return true;
  }

  if (message.type === 'STATE_CHANGED') {
    const { enabled, selectedModel: newModel, apiUrl: newApiUrl } = message.payload;
    const modelChanged = newModel && newModel !== selectedModel;

    extensionEnabled = enabled;
    if (newModel) selectedModel = newModel;
    if (newApiUrl) apiUrl = newApiUrl;

    if (!enabled) {
      removeAllLabels();
      sendResponse({ success: true });
      return true;
    }

    if (modelChanged) {
      // Model thay đổi → xóa nhãn cũ và rescan
      labeledPostIds.clear();
      stats = { total: 0, toxic: 0, nontoxic: 0 };
      removeAllLabels();
    }
    scanAllComments();
    sendResponse({ success: true });
    return true;
  }

  if (message.type === 'GET_STATS') {
    sendResponse(stats);
    return true;
  }
});

// ─────────────────────────────────────────────────────────────
// Core: thu thập và phân loại bình luận
// ─────────────────────────────────────────────────────────────
async function scanAllComments() {
  if (scanInProgress) return;
  scanInProgress = true;

  try {
    const posts = getUnlabeledPosts();
    console.log(`[VOZ Toxic Detector] Tìm thấy ${posts.length} bình luận chưa gán nhãn`);

    if (posts.length === 0) return;

    for (let i = 0; i < posts.length; i += BATCH_SIZE) {
      const batch = posts.slice(i, i + BATCH_SIZE);
      await processBatch(batch);
    }
  } catch (err) {
    console.error('[VOZ Toxic Detector] Lỗi scan:', err);
  } finally {
    scanInProgress = false;
    notifyStats();
  }
}

function getUnlabeledPosts() {
  const articles = document.querySelectorAll('article.message.message--post');
  const result = [];
  for (const article of articles) {
    const postId = article.getAttribute('data-content');
    if (!postId || labeledPostIds.has(postId)) continue;

    // Lấy text từ .bbWrapper bên trong div[itemprop="text"]
    const textEl = article.querySelector('div[itemprop="text"] .bbWrapper');
    if (!textEl) continue;

    const text = extractCommentText(textEl);
    if (!text) continue;

    result.push({ postId, text, article });
  }
  return result;
}

function extractCommentText(textEl) {
  const clone = textEl.cloneNode(true);
  clone.querySelectorAll('blockquote').forEach((el) => el.remove());
  clone.querySelectorAll('i').forEach((el) => el.remove());
  clone.querySelectorAll('.bbImageWrapper').forEach((el) => el.remove());
  return clone.innerText.trim();
}

async function processBatch(posts) {
  const texts = posts.map((p) => p.text);

  console.log(`[VOZ Toxic Detector] Gửi batch ${posts.length} bình luận đến API [${selectedModel}]`);

  let data;
  try {
    // Gọi API TRỰC TIẾP từ content script — không relay qua background worker
    // Chrome cho phép vì host_permissions có "http://localhost:8000/*"
    const response = await fetch(`${apiUrl}/predict/batch/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts, model: selectedModel }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error(`[VOZ Toxic Detector] API trả lỗi ${response.status}:`, errText);
      return;
    }

    data = await response.json();
    console.log(`[VOZ Toxic Detector] Nhận kết quả: ${data.toxic_count}/${data.total} toxic`);
  } catch (err) {
    console.error('[VOZ Toxic Detector] Không kết nối được API (server đang chạy chưa?):', err.message);
    return;
  }

  // Gán nhãn cho từng bình luận
  data.results.forEach((result) => {
    const { id: idx, label, confidence } = result;
    const post = posts[idx];
    if (!post) return;

    labeledPostIds.add(post.postId);
    attachLabel(post.article, label, confidence);

    stats.total++;
    if (label === 'toxic') stats.toxic++;
    else stats.nontoxic++;
  });
}

// ─────────────────────────────────────────────────────────────
// DOM: gán nhãn vào bình luận
// ─────────────────────────────────────────────────────────────
function attachLabel(article, label, confidence) {
  if (article.querySelector('.voz-toxic-badge')) return;

  const isToxic = label === 'toxic';
  const percent = Math.round(confidence * 100);

  const badge = document.createElement('div');
  badge.className = `voz-toxic-badge voz-badge--${isToxic ? 'toxic' : 'safe'}`;
  badge.title = `${isToxic ? '⚠ Có thể độc hại' : '✓ Bình thường'} — ${percent}% (${selectedModel.toUpperCase()})`;

  badge.innerHTML = `
    <span class="voz-badge-icon">${isToxic ? '⚠' : '✓'}</span>
    <span class="voz-badge-text">${isToxic ? 'TOXIC' : 'SAFE'}</span>
    <span class="voz-badge-confidence">${percent}%</span>
  `;

  if (isToxic) {
    article.classList.add('voz-message--toxic');
  }

  // Chèn vào danh sách attribution-opposite (cạnh nút #số)
  const header = article.querySelector('header.message-attribution');
  if (header) {
    const oppositeList = header.querySelector('.message-attribution-opposite');
    if (oppositeList) {
      const li = document.createElement('li');
      li.appendChild(badge);
      oppositeList.prepend(li);
      return;
    }
    header.appendChild(badge);
  } else {
    const main = article.querySelector('.message-main');
    if (main) main.prepend(badge);
  }
}

function removeAllLabels() {
  document.querySelectorAll('.voz-toxic-badge').forEach((el) => el.remove());
  document.querySelectorAll('.voz-message--toxic').forEach((el) => {
    el.classList.remove('voz-message--toxic');
  });
}

// ─────────────────────────────────────────────────────────────
// MutationObserver: bình luận mới được load (infinite scroll)
// ─────────────────────────────────────────────────────────────
let debounceTimer = null;

function observeNewComments() {
  const target = document.querySelector('.p-body-content') || document.body;
  const observer = new MutationObserver(() => {
    if (!extensionEnabled) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      scanAllComments();
    }, SCAN_DEBOUNCE_MS);
  });
  observer.observe(target, { childList: true, subtree: true });
}

// ─────────────────────────────────────────────────────────────
// Notify popup về stats mới (nếu popup đang mở)
// ─────────────────────────────────────────────────────────────
function notifyStats() {
  chrome.runtime.sendMessage({ type: 'STATS_UPDATE', payload: stats }).catch(() => {
    // Popup không mở — bỏ qua
  });
}

// ─────────────────────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────────────────────
init();
