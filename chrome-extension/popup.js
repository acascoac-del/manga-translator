const DEFAULT_SETTINGS = {
  sourceLanguageMode: 'auto',
  targetLang: 'es',
  translator: 'llamacpp',
  lmstudioModel: 'auto',
  lmstudioHost: 'http://localhost:1234/v1',
  ollamaModel: null,
  groqApiKey: '',
  groqModel: 'auto',
  apiPreset: 'http://localhost:8001',
  apiUrl: 'http://127.0.0.1:8001'
};

document.addEventListener('DOMContentLoaded', () => {
  const sourceLanguageSelect = document.getElementById('sourceLanguageMode');
  const targetLangSelect = document.getElementById('targetLang');
  const lmstudioModelSelect = document.getElementById('lmstudioModel');
  const groqApiKeyInput = document.getElementById('groqApiKey');
  const groqModelSelect = document.getElementById('groqModel');
  const apiPresetSelect = document.getElementById('apiPreset');
  const apiCustomGroup = document.getElementById('apiCustomGroup');
  const apiUrlInput = document.getElementById('apiUrl');
  const btnTranslate = document.getElementById('btnTranslate');
  const serverStatus = document.getElementById('serverStatus');

  const engineCards = document.querySelectorAll('.engine-card');
  let currentTranslator = 'llamacpp';

  // Load saved settings
  chrome.storage.local.get(DEFAULT_SETTINGS, (settings) => {
    const retiredGroqModels = new Set([
      'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'
    ]);
    if (retiredGroqModels.has(settings.groqModel)) {
      settings.groqModel = 'auto';
      chrome.storage.local.set({ groqModel: 'auto' });
    }
    currentTranslator = settings.translator === 'ollama'
      ? 'lmstudio'
      : (settings.translator || 'llamacpp');
    if (settings.translator === 'ollama') {
      settings.lmstudioModel = 'auto';
      chrome.storage.local.set({ translator: 'lmstudio', lmstudioModel: 'auto' });
    }
    sourceLanguageSelect.value = settings.sourceLanguageMode === 'ja' ? 'ja' : 'auto';
    targetLangSelect.value = settings.targetLang;
    apiPresetSelect.value = settings.apiPreset || 'http://localhost:8001';
    apiUrlInput.value = settings.apiUrl;
    groqApiKeyInput.value = settings.groqApiKey;
    groqModelSelect.value = settings.groqModel;
    if (!groqModelSelect.value) {
      settings.groqModel = 'auto';
      groqModelSelect.value = 'auto';
      chrome.storage.local.set({ groqModel: 'auto' });
    }

    setActiveEngine(currentTranslator);
    toggleApiCustomGroup(apiPresetSelect.value);
    checkServerStatus(settings.apiUrl);

    if (currentTranslator === 'lmstudio') {
      loadLmStudioModels(settings.apiUrl, settings.lmstudioModel);
    } else {
      populateLmStudioFallback(settings.lmstudioModel);
    }
  });

  // Engine card selection
  engineCards.forEach(card => {
    card.addEventListener('click', () => {
      const engine = card.dataset.engine;
      currentTranslator = engine;
      setActiveEngine(engine);
      saveSettings();

      if (engine === 'lmstudio') {
        loadLmStudioModels(apiUrlInput.value, lmstudioModelSelect.value);
      }
    });
  });

  function setActiveEngine(engine) {
    engineCards.forEach(card => {
      card.classList.toggle('active', card.dataset.engine === engine);
    });

    // Toggle settings panels
    const llamacppPanel = document.getElementById('llamacppSettings');
    if (llamacppPanel) llamacppPanel.classList.toggle('visible', engine === 'llamacpp');
    document.getElementById('groqSettings').classList.toggle('visible', engine === 'groq');
    document.getElementById('lmstudioSettings').classList.toggle('visible', engine === 'lmstudio');
  }

  function populateLmStudioFallback(currentModel) {
    lmstudioModelSelect.innerHTML = `
      <option value="auto">Automático (inicia el servidor de LM Studio)</option>
    `;
    for (let opt of lmstudioModelSelect.options) {
      if (opt.value === currentModel) opt.selected = true;
    }
  }

  async function loadLmStudioModels(apiUrl, currentModel) {
    chrome.runtime.sendMessage({
      action: 'fetch_lmstudio_models',
      apiUrl: apiUrl
    }, (response) => {
      if (chrome.runtime.lastError) {
        console.error('No se pudieron consultar los modelos de LM Studio:', chrome.runtime.lastError.message);
        populateLmStudioFallback(currentModel);
        return;
      }

      if (response?.success && response?.data?.models?.length > 0) {
        const models = response.data.models;
        lmstudioModelSelect.innerHTML = '';
        const autoOption = document.createElement('option');
        autoOption.value = 'auto';
        autoOption.textContent = 'Automático (recomendado)';
        autoOption.selected = !currentModel || currentModel === 'auto' || !models.includes(currentModel);
        lmstudioModelSelect.appendChild(autoOption);
        models.forEach(model => {
          const opt = document.createElement('option');
          opt.value = model;
          opt.textContent = model;
          if (model === currentModel) opt.selected = true;
          lmstudioModelSelect.appendChild(opt);
        });
        saveSettings();
        return;
      }
      populateLmStudioFallback(currentModel);
    });
  }

  async function checkServerStatus(apiUrl) {
    try {
      const safeUrl = apiUrl.replace('localhost', '127.0.0.1').replace(/\/$/, '');
      const resp = await fetch(`${safeUrl}/health`, { method: 'GET', signal: AbortSignal.timeout(3000) });
      if (resp.ok) {
        serverStatus.className = 'server-status connected';
        serverStatus.title = 'Servidor conectado';
      } else {
        serverStatus.className = 'server-status error';
        serverStatus.title = 'Servidor con errores';
      }
    } catch {
      serverStatus.className = 'server-status error';
      serverStatus.title = 'Servidor no reachable';
    }
  }

  // Save settings
  const saveSettings = () => {
    chrome.storage.local.set({
      sourceLanguageMode: sourceLanguageSelect.value,
      targetLang: targetLangSelect.value,
      translator: currentTranslator,
      lmstudioModel: lmstudioModelSelect.value,
      lmstudioHost: 'http://localhost:1234/v1',
      groqApiKey: groqApiKeyInput.value,
      groqModel: groqModelSelect.value,
      apiPreset: apiPresetSelect.value,
      apiUrl: apiUrlInput.value
    });
  };

  // Event listeners
  sourceLanguageSelect.addEventListener('change', saveSettings);
  targetLangSelect.addEventListener('change', saveSettings);
  lmstudioModelSelect.addEventListener('change', saveSettings);
  groqApiKeyInput.addEventListener('input', saveSettings);
  groqModelSelect.addEventListener('change', saveSettings);

  apiPresetSelect.addEventListener('change', () => {
    const preset = apiPresetSelect.value;
    toggleApiCustomGroup(preset);
    if (preset !== 'custom') apiUrlInput.value = preset;
    saveSettings();
    checkServerStatus(apiUrlInput.value);
    if (currentTranslator === 'lmstudio') {
      loadLmStudioModels(apiUrlInput.value, lmstudioModelSelect.value);
    }
  });

  apiUrlInput.addEventListener('input', () => {
    saveSettings();
    checkServerStatus(apiUrlInput.value);
    if (currentTranslator === 'lmstudio') {
      loadLmStudioModels(apiUrlInput.value, lmstudioModelSelect.value);
    }
  });

  function toggleApiCustomGroup(preset) {
    apiCustomGroup.style.display = preset === 'custom' ? 'block' : 'none';
  }

  // Translate button
  btnTranslate.addEventListener('click', async () => {
    btnTranslate.disabled = true;
    btnTranslate.innerHTML = '<span class="btn-icon">&#8987;</span> Enviando...';

    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (tab) {
      const config = {
        action: 'translate_page',
        sourceLang: sourceLanguageSelect.value,
        targetLang: targetLangSelect.value,
        translator: currentTranslator,
        lmstudioModel: lmstudioModelSelect.value,
        groqApiKey: groqApiKeyInput.value,
        groqModel: groqModelSelect.value,
        lmstudioHost: 'http://localhost:1234/v1',
        apiUrl: apiUrlInput.value
      };

      chrome.tabs.sendMessage(tab.id, config, (response) => {
        btnTranslate.disabled = false;
        btnTranslate.innerHTML = '<span class="btn-icon">&#9654;</span> Traducir Página';

        if (chrome.runtime.lastError) {
          alert('Error: Recarga la página de manga antes de traducir.');
        } else if (response?.success) {
          window.close();
        } else {
          alert('No se encontraron imágenes para traducir.');
        }
      });
    } else {
      btnTranslate.disabled = false;
      btnTranslate.innerHTML = '<span class="btn-icon">&#9654;</span> Traducir Página';
    }
  });
});
