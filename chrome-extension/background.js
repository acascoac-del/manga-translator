// Register Context Menu on Installation
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "translate-manga-image",
    title: "Traducir imagen de manga",
    contexts: ["image"]
  });
  console.log("Manga Translator context menu registered.");
});

// Listener for Context Menu Clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "translate-manga-image") {
    console.log("Triggering single image translation for:", info.srcUrl);
    
    // Retrieve configuration preferences
    chrome.storage.local.get({
      sourceLanguageMode: 'auto',
      targetLang: 'es',
      translator: 'llamacpp',
      lmstudioModel: 'auto',
      lmstudioHost: 'http://localhost:1234/v1',
      groqApiKey: '',
      groqModel: 'auto',
      apiUrl: 'http://127.0.0.1:8001'
    }, (settings) => {
      if (['llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'].includes(settings.groqModel)) {
        settings.groqModel = 'auto';
        chrome.storage.local.set({ groqModel: 'auto' });
      }
      // Send translation command to the active tab content script
      chrome.tabs.sendMessage(tab.id, {
        action: 'translate_single_image',
        imageUrl: info.srcUrl,
        ...settings,
        sourceLang: settings.sourceLanguageMode === 'ja' ? 'ja' : 'auto'
      }, (response) => {
        if (chrome.runtime.lastError) {
          console.error("Error sending message to content script:", chrome.runtime.lastError.message);
        } else {
          console.log("Single image translation request received by content script:", response);
        }
      });
    });
  }
});

// Listener for runtime messages
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // Manejar el ping para mantener vivo al Service Worker
  if (request.action === 'ping') {
    sendResponse({ success: true, status: 'alive' });
    return false; // Respondemos síncronamente
  }
  else if (request.action === 'fetch_image_base64') {
    fetchImageBase64(request.url, request.referer)
      .then(base64 => sendResponse({ success: true, base64 }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }
  else if (request.action === 'fetch_lmstudio_models') {
    const safeApiUrl = request.apiUrl.replace('localhost', '127.0.0.1').replace(/\/$/, '');
    fetch(`${safeApiUrl}/lmstudio/models`)
      .then(resp => {
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return resp.json();
      })
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }
  else if (request.action === 'translate_image') {
    // Force 127.0.0.1 to avoid IPv6 timeout issues with Docker Desktop on Windows
    const safeApiUrl = request.apiUrl.replace('localhost', '127.0.0.1').replace(/\/$/, '');
    fetch(`${safeApiUrl}/translate/json`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request.payload),
      // Normal pages now finish in seconds. Waiting three minutes made a
      // stalled page look as if the whole chapter had frozen and delayed the
      // content script retry unnecessarily.
      signal: AbortSignal.timeout(45000)
    })
      .then(resp => {
        if (!resp.ok) throw new Error(`API returned HTTP ${resp.status}`);
        return resp.json();
      })
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }
});

// Hotlink-protected CDNs return 403 unless Referer matches the reader page.
// fetch() in a service worker does not reliably apply a custom referrer, so the
// Referer is forced at the network layer with a per-host session rule.
const _refererRuleIds = new Map();
let _nextRefererRuleId = 7001;

async function ensureRefererRule(url, referer) {
  if (!chrome.declarativeNetRequest || !referer) return;
  let host;
  try {
    host = new URL(url).hostname;
  } catch (e) {
    return;
  }

  let ruleId = _refererRuleIds.get(host);
  if (!ruleId) {
    ruleId = _nextRefererRuleId++;
    _refererRuleIds.set(host, ruleId);
  }

  try {
    await chrome.declarativeNetRequest.updateSessionRules({
      removeRuleIds: [ruleId],
      addRules: [{
        id: ruleId,
        priority: 1,
        action: {
          type: 'modifyHeaders',
          requestHeaders: [
            { operation: 'set', header: 'Referer', value: referer },
            { operation: 'remove', header: 'Origin' },
          ],
        },
        condition: {
          requestDomains: [host],
          resourceTypes: ['xmlhttprequest', 'image', 'media', 'other'],
        },
      }],
    });
  } catch (e) {
    console.warn('Could not install referer rule for', host, e);
  }
}

// Helper to fetch and convert image to base64 bypassing CORS.
async function fetchImageBase64(url, pageReferer) {
  let originReferer = null;
  try {
    originReferer = new URL(url).origin + '/';
  } catch (e) { /* invalid url, ignore */ }

  // Force the reader-page Referer on the wire for hotlink protection.
  await ensureRefererRule(url, pageReferer || originReferer);

  const headers = {
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
  };

  const response = await fetch(url, { headers, credentials: 'include' });
  if (!response.ok) {
    throw new Error(`Failed to fetch image: HTTP ${response.status}`);
  }
  const arrayBuffer = await response.arrayBuffer();
  const bytes = new Uint8Array(arrayBuffer);
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}
