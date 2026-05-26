/**
 * Background Service Worker
 * Chỉ quản lý state (storage). API calls được thực hiện trực tiếp
 * từ content script để tránh vấn đề service worker bị kill khi idle.
 */

const DEFAULT_STATE = {
  enabled: true,
  selectedModel: 'bilstm',
  apiUrl: 'http://localhost:8000/api',
};

// Khởi tạo state khi extension được cài lần đầu
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(Object.keys(DEFAULT_STATE), (stored) => {
    const toSet = {};
    for (const [key, defaultVal] of Object.entries(DEFAULT_STATE)) {
      if (stored[key] === undefined) toSet[key] = defaultVal;
    }
    if (Object.keys(toSet).length > 0) {
      chrome.storage.local.set(toSet);
    }
  });
  console.log('[VOZ Toxic Detector] Extension installed/updated.');
});

// Lắng nghe message từ popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'GET_STATE') {
    chrome.storage.local.get(Object.keys(DEFAULT_STATE), sendResponse);
    return true;
  }

  if (message.type === 'SET_STATE') {
    chrome.storage.local.set(message.payload, () => {
      sendResponse({ success: true });
    });
    return true;
  }
});
