// Global states
let activeProgressWidget = null;
let translationRunActive = false;
let activeWidgetData = {
  total: 0,
  completed: 0,
  rollingTimes: [],
  startTime: 0
};

// Listen for messages from popup or background scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'translate_page') {
    if (translationRunActive) {
      sendResponse({ success: false, reason: 'Translation already running' });
      return;
    }
    const images = discoverImages();
    if (images.length === 0) {
      sendResponse({ success: false, reason: 'No images found' });
      return;
    }

    translationRunActive = true;
    sendResponse({ success: true, count: images.length });
    startProgressiveTranslation(images, request).finally(() => {
      translationRunActive = false;
    });
  } 
  else if (request.action === 'translate_single_image') {
    if (translationRunActive) {
      sendResponse({ success: false, reason: 'Translation already running' });
      return;
    }
    const imgEl = findImageElementByUrl(request.imageUrl);
    if (imgEl) {
      translationRunActive = true;
      sendResponse({ success: true });
      translateSingleImage(imgEl, request).finally(() => {
        translationRunActive = false;
      });
    } else {
      sendResponse({ success: false, reason: 'Image element not found in DOM' });
    }
  }
  // Se eliminó el "return true;" que estaba aquí para evitar conflictos asíncronos
});

// Discover all large images on the webpage (ignores icons, logos, ads)
function discoverImages() {
  // Prefer a dedicated chapter reader when the site exposes one. Looking at
  // every <img> also sends logos, loading SVGs, ad pixels and credit banners
  // to the backend; on mgeko that inflated a 22-page chapter to 26 jobs and
  // produced invalid-image errors before the real pages started.
  const readerSelectors = [
    '#chapter-reader',
    '.chapter-reader',
    '.reading-content',
    '.reader-content',
    '#readerarea',
    '#reader-area'
  ];
  let images = [];
  for (const selector of readerSelectors) {
    const root = document.querySelector(selector);
    if (!root) continue;
    const scoped = Array.from(root.querySelectorAll('img'));
    if (scoped.length >= 2) {
      images = scoped;
      break;
    }
  }
  if (images.length === 0) {
    images = Array.from(document.querySelectorAll('img'));
  }

  return images.filter(img => {
    // Ignore already translated
    if (img.dataset.mangaTranslated === 'true') return false;
    if (img.dataset.mangaTranslating === 'true') return false;

    const hasLazySrc = img.hasAttribute('data-src') || 
                       img.hasAttribute('data-original') || 
                       img.hasAttribute('data-lazy-src') ||
                       img.hasAttribute('data-url') ||
                       img.hasAttribute('data-full-url') ||
                       img.hasAttribute('data-cdn-src') ||
                       img.hasAttribute('srcset');

    const rect = img.getBoundingClientRect();
    const w = img.naturalWidth || rect.width || img.width;
    const h = img.naturalHeight || rect.height || img.height;
    const source = (
      img.currentSrc || img.src || img.dataset.src || img.dataset.original || ''
    ).toLowerCase();

    if (/\.svg(?:[?#]|$)/.test(source)) return false;
    if (/(?:^|[\/_-])(loading|loader|logo|credits?)(?:[\/_\-.]|$)/.test(source)) {
      return false;
    }
    
    // If it has lazy src attributes or reasonable dimensions, consider it a manga page
    if (!hasLazySrc && (w < 400 || h < 500)) return false;
    
    return true;
  });
}

// Find image element by source URL in DOM
function findImageElementByUrl(url) {
  const images = document.querySelectorAll('img');
  for (const img of images) {
    if (img.src === url || img.dataset.originalSrc === url) {
      return img;
    }
  }
  return null;
}

// Create a glassmorphic floating progress dashboard widget
function createProgressWidget(totalCount) {
  if (activeProgressWidget) {
    activeProgressWidget.remove();
  }

  activeWidgetData = {
    total: totalCount,
    completed: 0,
    rollingTimes: [],
    startTime: Date.now()
  };

  const widget = document.createElement('div');
  widget.className = 'manga-progress-widget';
  widget.id = 'manga-progress-widget-root';
  
  widget.innerHTML = `
    <!-- Minimized view icon badge -->
    <div class="manga-progress-minimized-badge" id="manga-widget-minimized">
      <span style="font-size: 14px;">文</span>
      <span id="manga-widget-min-text">0/${totalCount}</span>
    </div>

    <!-- Full expanded content -->
    <div class="manga-progress-full-content" id="manga-widget-expanded">
      <div class="manga-progress-widget-header">
        <span class="manga-progress-widget-title">Traduciendo Manga</span>
        <div class="manga-progress-widget-controls">
          <button class="manga-progress-btn" id="manga-btn-minimize" title="Minimizar">_</button>
          <button class="manga-progress-btn" id="manga-btn-close" title="Cerrar">×</button>
        </div>
      </div>
      
      <div class="manga-progress-stats">
        <span>Progreso:</span>
        <span id="manga-txt-progress">0 de ${totalCount} páginas</span>
      </div>
      
      <div class="manga-progress-bar-bg">
        <div class="manga-progress-bar-fill" id="manga-bar-fill"></div>
      </div>
      
      <div class="manga-progress-eta">
        <span>Tiempo Estimado:</span>
        <span class="manga-progress-eta-badge" id="manga-txt-eta">Calculando...</span>
      </div>
    </div>
  `;

  document.body.appendChild(widget);
  activeProgressWidget = widget;

  // Add event listeners
  const btnMin = widget.querySelector('#manga-btn-minimize');
  const btnClose = widget.querySelector('#manga-btn-close');

  btnMin.addEventListener('click', (e) => {
    e.stopPropagation();
    widget.classList.add('minimized');
  });

  btnClose.addEventListener('click', (e) => {
    e.stopPropagation();
    widget.remove();
    activeProgressWidget = null;
  });

  // Restore when clicking minimized bubble
  widget.addEventListener('click', () => {
    if (widget.classList.contains('minimized')) {
      widget.classList.remove('minimized');
    }
  });
}

// Update progress stats in floating widget
function updateProgressWidget(completedIncrement, elapsedMs) {
  if (!activeProgressWidget) return;

  activeWidgetData.completed += completedIncrement;
  if (elapsedMs > 0) {
    activeWidgetData.rollingTimes.push(elapsedMs);
  }

  const { total, completed, rollingTimes } = activeWidgetData;
  const pct = Math.min(100, Math.round((completed / total) * 100));

  // Update text values
  document.getElementById('manga-txt-progress').textContent = `${completed} de ${total} páginas`;
  document.getElementById('manga-bar-fill').style.width = `${pct}%`;
  document.getElementById('manga-widget-min-text').textContent = `${completed}/${total}`;

  // Calculate ETA
  const txtEta = document.getElementById('manga-txt-eta');
  if (completed === total) {
    txtEta.textContent = '¡Completado!';
    txtEta.style.background = 'rgba(46, 213, 115, 0.2)';
    txtEta.style.color = '#2ed573';
  } else if (rollingTimes.length > 0) {
    const sum = rollingTimes.reduce((a, b) => a + b, 0);
    const avg = sum / rollingTimes.length;
    const remaining = total - completed;
    const etaMs = avg * remaining;
    
    txtEta.textContent = formatTime(etaMs);
  } else {
    txtEta.textContent = 'Calculando...';
  }
}

function formatTime(ms) {
  const seconds = Math.ceil(ms / 1000);
  if (seconds < 60) {
    return `~${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const remSecs = seconds % 60;
  return `~${minutes}m ${remSecs}s`;
}

// Overlay a loading spinner over image
function addLoaderOverlay(imgEl) {
  // If already wrapped, return existing overlay
  if (imgEl.parentElement && imgEl.parentElement.classList.contains('manga-translate-wrapper')) {
    return imgEl.parentElement.querySelector('.manga-translate-spinner-overlay');
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'manga-translate-wrapper';
  
  // Set wrapper styles to mirror image dimensions and layout
  const rect = imgEl.getBoundingClientRect();
  wrapper.style.width = (rect.width || imgEl.clientWidth || imgEl.naturalWidth || 200) + 'px';
  wrapper.style.height = (rect.height || imgEl.clientHeight || imgEl.naturalHeight || 300) + 'px';
  wrapper.style.display = window.getComputedStyle(imgEl).display;
  
  imgEl.replaceWith(wrapper);
  wrapper.appendChild(imgEl);

  const overlay = document.createElement('div');
  overlay.className = 'manga-translate-spinner-overlay';
  overlay.innerHTML = `
    <div class="manga-translate-spinner"></div>
    <span style="font-weight: 500;">Traduciendo...</span>
  `;
  wrapper.appendChild(overlay);
  return overlay;
}

// Remove loading spinner wrapper
function removeLoaderOverlay(imgEl) {
  const wrapper = imgEl.parentElement;
  if (wrapper && wrapper.classList.contains('manga-translate-wrapper')) {
    const overlay = wrapper.querySelector('.manga-translate-spinner-overlay');
    if (overlay) overlay.remove();
    wrapper.replaceWith(imgEl);
  }
}

// Extract base64 representation of image source
async function getBase64FromImage(imgEl) {
  // Extract candidate URLs from common lazy-loading attributes
  const candidates = [
    // Always prefer the immutable source saved before the first translation.
    // This prevents translating an already rendered image again, which
    // otherwise accumulates multiple Spanish texts on the same bubble.
    imgEl.dataset.originalSrc,
    imgEl.getAttribute('data-src'),
    imgEl.getAttribute('data-original'),
    imgEl.getAttribute('data-lazy-src'),
    imgEl.getAttribute('data-url'),
    imgEl.getAttribute('data-full-url'),
    imgEl.getAttribute('data-cdn-src'),
    imgEl.src
  ].filter(Boolean);

  // Find the first valid non-placeholder URL
  let realSrc = candidates.find(url => {
    if (!url) return false;
    if (url.startsWith('data:')) {
      // Ignore tiny placeholder data URIs (e.g. 1x1 transparent gifs < 500 chars)
      return url.length > 500;
    }
    const lower = url.toLowerCase();
    return !lower.includes('placeholder') && !lower.includes('loading') && !lower.includes('blank.gif') && !lower.includes('transparent.png');
  }) || imgEl.src;

  // Convert relative URL to absolute URL to prevent service worker fetch failure
  if (realSrc && !realSrc.startsWith('data:') && !realSrc.startsWith('blob:')) {
    try {
      realSrc = new URL(realSrc, window.location.href).href;
    } catch (err) {
      console.warn("Could not resolve absolute URL:", realSrc, err);
    }
  }

  // Ensure image is loaded in memory if it hasn't loaded yet
  if (!imgEl.complete || imgEl.naturalWidth === 0) {
    await new Promise((resolve) => {
      const tempImg = new Image();
      tempImg.crossOrigin = 'anonymous';
      tempImg.onload = () => resolve();
      tempImg.onerror = () => resolve();
      tempImg.src = realSrc;
      // Timeout after 5 seconds if image fails to load
      setTimeout(resolve, 5000);
    });
  }

  // 1. Handle non-placeholder data URLs directly
  if (realSrc.startsWith('data:')) {
    return realSrc.split(',')[1];
  }

  // 2. Handle blob URLs (background scripts cannot fetch blob URLs from pages)
  if (realSrc.startsWith('blob:')) {
    try {
      const response = await fetch(realSrc);
      const blob = await response.blob();
      return await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(',')[1]);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (e) {
      console.warn("Failed to read blob URL, falling back to canvas", e);
    }
  }

  // 3. Ask background service worker to fetch and convert to base64 bypassing CORS
  try {
    return await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({
        action: 'fetch_image_base64',
        url: realSrc,
        referer: window.location.href
      }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (response && response.success && response.base64) {
          resolve(response.base64);
        } else {
          reject(new Error(response ? response.error : 'Failed to fetch image'));
        }
      });
    });
  } catch (e) {
    console.warn("Background fetch failed for image, falling back to canvas", realSrc, e);
  }

  // 4. Fallback: Draw directly to canvas
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = imgEl.naturalWidth || imgEl.width || 800;
      canvas.height = imgEl.naturalHeight || imgEl.height || 1200;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgEl, 0, 0, canvas.width, canvas.height);
      const dataURL = canvas.toDataURL('image/jpeg', 0.9);
      resolve(dataURL.split(',')[1]);
    } catch (canvasErr) {
      reject(new Error(`Canvas fallback failed: ${canvasErr.message}`));
    }
  });
}

// Send one page to the background worker. A page must not be silently skipped
// because the service worker was briefly suspended or the local API restarted
// while a long request was in flight.
async function requestImageTranslation(config, payload, maxAttempts = 2) {
  let lastError;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({
          action: 'translate_image',
          apiUrl: config.apiUrl,
          payload
        }, (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else if (response && response.success) {
            resolve(response.data);
          } else {
            reject(new Error(response ? response.error : 'Failed to translate image via background worker'));
          }
        });
      });
    } catch (error) {
      lastError = error;
      if (attempt >= maxAttempts) break;

      console.warn(
        `Translation request failed (attempt ${attempt}/${maxAttempts}); retrying page`,
        error
      );
      // Give a restarted API/service worker time to become ready without
      // blocking the page or advancing the chapter progress counter.
      await new Promise(resolve => setTimeout(resolve, 1500));
    }
  }

  throw lastError || new Error('Failed to translate image');
}

// Progressive image translation orchestrator
async function startProgressiveTranslation(images, config) {
  createProgressWidget(images.length);

  // The API serializes the shared detection/OCR/inpainting pipeline for every
  // engine. A second browser worker can only queue another full base64 page;
  // it does not improve throughput and needlessly raises browser/API memory.
  const concurrency = Math.min(images.length, 1);
  let index = 0;
  
  // Iniciar un ping de latido cada 20 segundos para mantener vivo el background.js
  const keepAlive = setInterval(() => {
    chrome.runtime.sendMessage({ action: 'ping' });
  }, 20000); 
  
  const worker = async () => {
    while (index < images.length) {
      const currentIndex = index++;
      const imgEl = images[currentIndex];
      const imgStartTime = Date.now();
      imgEl.dataset.mangaTranslating = 'true';
      
      addLoaderOverlay(imgEl);
      
      try {
        const base64Str = await getBase64FromImage(imgEl);
        
        // Prepare API request payload
        const payload = {
          image: base64Str,
          // Detect every page independently. A stale manual setting or one
          // atypical page must never route later English pages through the
          // Japanese OCR model.
          source_lang: config.sourceLang === 'ja' ? 'ja' : 'auto',
          target_lang: config.targetLang,
          translator: config.translator,
          lmstudio_model: config.lmstudioModel,
          lmstudio_host: config.lmstudioHost,
          nllb_host: config.nllbHost,
          nllb_engine: config.nllbEngine,
          groq_api_key: config.groqApiKey,
          groq_model: config.groqModel,
          detector: 'ctd',
          ocr: 'auto',
          inpainter: 'lama',
          renderer: 'manga2eng'
        };

        // Request translation via background worker to bypass page CSP restrictions
        const result = await requestImageTranslation(config, payload);
        
        if (result.success && result.translated_image) {
          // Replace source immediately
          if (!imgEl.dataset.originalSrc) {
            imgEl.dataset.originalSrc = imgEl.src;
          }
          imgEl.src = `data:image/jpeg;base64,${result.translated_image}`;
          imgEl.dataset.mangaTranslated = 'true';
          imgEl.classList.add('manga-translated-fade-in');
        } else {
          throw new Error(result.error || 'API translation failed');
        }

      } catch (e) {
        console.error(`Failed to translate image ${currentIndex}:`, e);
      } finally {
        delete imgEl.dataset.mangaTranslating;
        removeLoaderOverlay(imgEl);
        const elapsed = Date.now() - imgStartTime;
        updateProgressWidget(1, elapsed);
      }
    }
  };

  // Launch workers in parallel
  const workers = Array.from({ length: concurrency }, () => worker());
  await Promise.all(workers);
  
  // Detener el ping cuando todas las imágenes terminen
  clearInterval(keepAlive);
}

// Translate a single image triggered from the context menu
async function translateSingleImage(imgEl, config) {
  const imgStartTime = Date.now();
  imgEl.dataset.mangaTranslating = 'true';
  addLoaderOverlay(imgEl);

  // Latido para imágenes individuales
  const keepAlive = setInterval(() => chrome.runtime.sendMessage({ action: 'ping' }), 20000);

  try {
    const base64Str = await getBase64FromImage(imgEl);
    const payload = {
      image: base64Str,
      source_lang: config.sourceLang === 'ja' ? 'ja' : 'auto',
      target_lang: config.targetLang,
      translator: config.translator,
      lmstudio_model: config.lmstudioModel,
      lmstudio_host: config.lmstudioHost,
      nllb_host: config.nllbHost,
      nllb_engine: config.nllbEngine,
      groq_api_key: config.groqApiKey,
      groq_model: config.groqModel,
      detector: 'ctd',
      ocr: 'auto',
      inpainter: 'lama',
      renderer: 'manga2eng'
    };

    // Request translation via background worker to bypass page CSP restrictions
    const result = await requestImageTranslation(config, payload);
    
    if (result.success && result.translated_image) {
      if (!imgEl.dataset.originalSrc) {
        imgEl.dataset.originalSrc = imgEl.src;
      }
      imgEl.src = `data:image/jpeg;base64,${result.translated_image}`;
      imgEl.dataset.mangaTranslated = 'true';
      imgEl.classList.add('manga-translated-fade-in');
    } else {
      throw new Error(result.error || 'API translation failed');
    }

  } catch (e) {
    console.error('Single image translation failed:', e);
    alert('Fallo la traducción de la imagen. Verifica que la API local esté corriendo.');
  } finally {
    delete imgEl.dataset.mangaTranslating;
    // Limpiar intervalo
    clearInterval(keepAlive);
    removeLoaderOverlay(imgEl);
  }
}
