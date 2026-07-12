// Global states
let activeProgressWidget = null;
let activeWidgetData = {
  total: 0,
  completed: 0,
  rollingTimes: [],
  startTime: 0
};

// Listen for messages from popup or background scripts
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'translate_page') {
    const images = discoverImages();
    if (images.length === 0) {
      sendResponse({ success: false, reason: 'No images found' });
      return;
    }

    sendResponse({ success: true, count: images.length });
    startProgressiveTranslation(images, request);
  } 
  else if (request.action === 'translate_single_image') {
    const imgEl = findImageElementByUrl(request.imageUrl);
    if (imgEl) {
      sendResponse({ success: true });
      translateSingleImage(imgEl, request);
    } else {
      sendResponse({ success: false, reason: 'Image element not found in DOM' });
    }
  }
  // Se eliminó el "return true;" que estaba aquí para evitar conflictos asíncronos
});

// Discover all large images on the webpage (ignores icons, logos, ads)
function discoverImages() {
  const images = Array.from(document.querySelectorAll('img'));
  return images.filter(img => {
    // Basic heuristics for comic/manga pages
    const rect = img.getBoundingClientRect();
    const w = img.naturalWidth || rect.width || img.width;
    const h = img.naturalHeight || rect.height || img.height;
    
    // Ignore small icons or thin borders (unless they are placeholders hiding a real image)
    const hasLazySrc = img.hasAttribute('data-src') || img.hasAttribute('data-original') || img.hasAttribute('data-lazy-src');
    if (!hasLazySrc && (w < 200 || h < 200)) return false;
    
    // Ignore already translated
    if (img.dataset.mangaTranslated === 'true') return false;
    
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
  // Buscar la URL real oculta por el Lazy Loading
  const src = imgEl.getAttribute('data-src') || 
              imgEl.getAttribute('data-original') || 
              imgEl.getAttribute('data-lazy-src') || 
              imgEl.src;
  
  // 1. Handle data URLs directly
  if (src.startsWith('data:')) {
    return src.split(',')[1];
  }
  
  // 2. Handle blob URLs (background scripts cannot fetch blob URLs from pages)
  if (src.startsWith('blob:')) {
    try {
      const response = await fetch(src);
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
        url: src
      }, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else if (response && response.success) {
          resolve(response.base64);
        } else {
          reject(new Error(response ? response.error : 'Failed to fetch image'));
        }
      });
    });
  } catch (e) {
    console.warn("Background fetch failed for image, falling back to canvas", src, e);
    // 4. Fallback: Draw to canvas
    return new Promise((resolve, reject) => {
      try {
        const canvas = document.createElement('canvas');
        // Si es un placeholder, el naturalWidth será 0, forzamos un mínimo para evitar errores
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
}

// Progressive image translation orchestrator
async function startProgressiveTranslation(images, config) {
  createProgressWidget(images.length);

  const concurrency = 2; // Run 2 parallel requests to avoid overloading local CPU/GPU/Ollama
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
      
      addLoaderOverlay(imgEl);
      
      try {
        const base64Str = await getBase64FromImage(imgEl);
        
        // Prepare API request payload
        const payload = {
          image: base64Str,
          target_lang: config.targetLang,
          translator: config.translator,
          ollama_model: config.ollamaModel,
          ollama_host: config.ollamaHost,
          libretranslate_host: config.libretranslateHost,
          groq_api_key: config.groqApiKey,
          groq_model: config.groqModel,
          detector: 'ctd',
          ocr: 'manga_ocr',
          inpainter: 'lama',
          renderer: 'manga2eng'
        };

        // Request translation via background worker to bypass page CSP restrictions
        const result = await new Promise((resolve, reject) => {
          chrome.runtime.sendMessage({
            action: 'translate_image',
            apiUrl: config.apiUrl,
            payload: payload
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
        
        if (result.success && result.translated_image) {
          // Replace source immediately
          imgEl.dataset.originalSrc = imgEl.src;
          imgEl.src = `data:image/jpeg;base64,${result.translated_image}`;
          imgEl.dataset.mangaTranslated = 'true';
          imgEl.classList.add('manga-translated-fade-in');
        } else {
          throw new Error(result.error || 'API translation failed');
        }

      } catch (e) {
        console.error(`Failed to translate image ${currentIndex}:`, e);
      } finally {
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
  addLoaderOverlay(imgEl);

  // Latido para imágenes individuales
  const keepAlive = setInterval(() => chrome.runtime.sendMessage({ action: 'ping' }), 20000);

  try {
    const base64Str = await getBase64FromImage(imgEl);
    const payload = {
      image: base64Str,
      target_lang: config.targetLang,
      translator: config.translator,
      ollama_model: config.ollamaModel,
      ollama_host: config.ollamaHost,
      libretranslate_host: config.libretranslateHost,
      groq_api_key: config.groqApiKey,
      groq_model: config.groqModel,
      detector: 'ctd',
      ocr: 'manga_ocr',
      inpainter: 'lama',
      renderer: 'manga2eng'
    };

    // Request translation via background worker to bypass page CSP restrictions
    const result = await new Promise((resolve, reject) => {
      chrome.runtime.sendMessage({
        action: 'translate_image',
        apiUrl: config.apiUrl,
        payload: payload
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
    
    if (result.success && result.translated_image) {
      imgEl.dataset.originalSrc = imgEl.src;
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
    // Limpiar intervalo
    clearInterval(keepAlive);
    removeLoaderOverlay(imgEl);
  }
}