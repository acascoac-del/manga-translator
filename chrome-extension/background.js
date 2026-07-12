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
      targetLang: 'es',
      translator: 'google',
      ollamaModel: 'llama3',
      ollamaHost: 'http://localhost:8080',
      groqApiKey: '',
      groqModel: 'llama-3.1-8b-instant',
      apiUrl: 'http://127.0.0.1:8001'
    }, (settings) => {
      // Send translation command to the active tab content script
      chrome.tabs.sendMessage(tab.id, {
        action: 'translate_single_image',
        imageUrl: info.srcUrl,
        ...settings
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
    fetchImageBase64(request.url)
      .then(base64 => sendResponse({ success: true, base64 }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true; // Keep message channel open for async response
  }
  else if (request.action === 'fetch_ollama_models') {
    const safeApiUrl = request.apiUrl.replace('localhost', '127.0.0.1').replace(/\/$/, '');
    fetch(`${safeApiUrl}/ollama/models`)
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
      body: JSON.stringify(request.payload)
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

// Helper to fetch and convert image to base64 bypassing CORS
async function fetchImageBase64(url) {
  const response = await fetch(url);
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
