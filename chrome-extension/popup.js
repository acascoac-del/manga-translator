// Default config settings
const DEFAULT_SETTINGS = {
  targetLang: 'es',
  translator: 'google',
  ollamaModel: 'qwen2.5:3b',
  groqApiKey: '',
  groqModel: 'llama-3.1-8b-instant',
  libretranslateHost: 'http://localhost:5000',
  apiPreset: 'http://localhost:8001',
  apiUrl: 'http://127.0.0.1:8001'
};

document.addEventListener('DOMContentLoaded', () => {
  const targetLangSelect = document.getElementById('targetLang');
  const translatorSelect = document.getElementById('translator');
  const ollamaSettingsDiv = document.getElementById('ollamaSettings');
  const ollamaModelSelect = document.getElementById('ollamaModel');
  const groqSettingsDiv = document.getElementById('groqSettings');
  const groqApiKeyInput = document.getElementById('groqApiKey');
  const groqModelSelect = document.getElementById('groqModel');
  const libreTranslateSettingsDiv = document.getElementById('libreTranslateSettings');
  const libreTranslateHostInput = document.getElementById('libretranslateHost');
  const apiPresetSelect = document.getElementById('apiPreset');
  const apiCustomGroup = document.getElementById('apiCustomGroup');
  const apiUrlInput = document.getElementById('apiUrl');
  const btnTranslate = document.getElementById('btnTranslate');

  // Load saved settings
  chrome.storage.local.get(DEFAULT_SETTINGS, (settings) => {
    targetLangSelect.value = settings.targetLang;
    translatorSelect.value = settings.translator;
    apiPresetSelect.value = settings.apiPreset || 'http://localhost:8001';
    apiUrlInput.value = settings.apiUrl;
    libreTranslateHostInput.value = settings.libretranslateHost;
    groqApiKeyInput.value = settings.groqApiKey;
    groqModelSelect.value = settings.groqModel;
    
    toggleApiCustomGroup(apiPresetSelect.value);
    toggleSettings(settings.translator);

    // Load Ollama models and select active, only if Ollama is the selected translator
    if (settings.translator === 'ollama') {
      loadOllamaModels(settings.apiUrl, settings.ollamaModel);
    } else {
      // Just populate the fallback list so the dropdown isn't empty if they switch later
      populateOllamaFallback(settings.ollamaModel);
    }
  });

  function populateOllamaFallback(currentModel) {
    ollamaModelSelect.innerHTML = `
      <option value="qwen2.5:3b">qwen2.5:3b</option>
      <option value="gemma4:e4b">gemma4:e4b</option>
      <option value="gemma3:4b">gemma3:4b</option>
      <option value="llama3">llama3</option>
    `;
    for (let opt of ollamaModelSelect.options) {
      if (opt.value === currentModel) {
        opt.selected = true;
      }
    }
  }

  // Load models from API via background worker to bypass CORS
  async function loadOllamaModels(apiUrl, currentModel) {
    chrome.runtime.sendMessage({
      action: 'fetch_ollama_models',
      apiUrl: apiUrl
    }, (response) => {
      if (response && response.success && response.data && response.data.models) {
        const models = response.data.models;
        if (models.length > 0) {
          ollamaModelSelect.innerHTML = '';
          models.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            if (model === currentModel) {
              opt.selected = true;
            }
            ollamaModelSelect.appendChild(opt);
          });
          saveSettings();
          return;
        }
      }
      
      // Fallback list
      console.warn("Could not query Ollama models from background worker, using fallback list.");
      populateOllamaFallback(currentModel);
    });
  }

  // Save settings on changes
  const saveSettings = () => {
    chrome.storage.local.set({
      targetLang: targetLangSelect.value,
      translator: translatorSelect.value,
      ollamaModel: ollamaModelSelect.value,
      groqApiKey: groqApiKeyInput.value,
      groqModel: groqModelSelect.value,
      libretranslateHost: libreTranslateHostInput.value,
      apiPreset: apiPresetSelect.value,
      apiUrl: apiUrlInput.value
    });
  };

  targetLangSelect.addEventListener('change', saveSettings);
  ollamaModelSelect.addEventListener('change', saveSettings);
  groqApiKeyInput.addEventListener('input', saveSettings);
  groqModelSelect.addEventListener('change', saveSettings);
  libreTranslateHostInput.addEventListener('input', saveSettings);
  
  apiPresetSelect.addEventListener('change', () => {
    const preset = apiPresetSelect.value;
    toggleApiCustomGroup(preset);
    if (preset !== 'custom') {
      apiUrlInput.value = preset;
    }
    saveSettings();
    if (translatorSelect.value === 'ollama') {
      loadOllamaModels(apiUrlInput.value, ollamaModelSelect.value);
    }
  });
  
  apiUrlInput.addEventListener('input', () => {
    saveSettings();
    if (translatorSelect.value === 'ollama') {
      loadOllamaModels(apiUrlInput.value, ollamaModelSelect.value);
    }
  });

  translatorSelect.addEventListener('change', () => {
    toggleSettings(translatorSelect.value);
    saveSettings();
    if (translatorSelect.value === 'ollama') {
      loadOllamaModels(apiUrlInput.value, ollamaModelSelect.value);
    }
  });

  function toggleSettings(value) {
    ollamaSettingsDiv.style.display = value === 'ollama' ? 'block' : 'none';
    groqSettingsDiv.style.display = value === 'groq' ? 'block' : 'none';
    libreTranslateSettingsDiv.style.display = value === 'libretranslate' ? 'block' : 'none';
  }
  
  function toggleApiCustomGroup(preset) {
    if (preset === 'custom') {
      apiCustomGroup.style.display = 'block';
    } else {
      apiCustomGroup.style.display = 'none';
    }
  }

  // Trigger translation in content script of current page
  btnTranslate.addEventListener('click', async () => {
    btnTranslate.disabled = true;
    btnTranslate.textContent = 'Enviando...';

    // Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (tab) {
      const config = {
        action: 'translate_page',
        targetLang: targetLangSelect.value,
        translator: translatorSelect.value,
        ollamaModel: ollamaModelSelect.value,
        groqApiKey: groqApiKeyInput.value,
        groqModel: groqModelSelect.value,
        libretranslateHost: libreTranslateHostInput.value,
        ollamaHost: 'http://localhost:8080', // preconfigured port, resolved internally in API to host.docker.internal
        apiUrl: apiUrlInput.value
      };
      
      // Send message to content script
      chrome.tabs.sendMessage(tab.id, config, (response) => {
        btnTranslate.disabled = false;
        btnTranslate.textContent = 'Traducir Página Completa';
        
        if (chrome.runtime.lastError) {
          alert('Error: Por favor recarga la página de manga antes de intentar traducir.');
        } else if (response && response.success) {
          window.close(); // Close popup if successfully sent
        } else {
          alert('No se encontraron imágenes aptas para traducir en esta página.');
        }
      });
    } else {
      btnTranslate.disabled = false;
      btnTranslate.textContent = 'Traducir Página Completa';
    }
  });
});
