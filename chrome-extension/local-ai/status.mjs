import { browserTranslator } from './pipeline.mjs';
document.querySelector('#check').addEventListener('click', async () => {
  const status = document.querySelector('#status');
  try {
    const translator = await browserTranslator();
    try {
      const output = await translator.translate('こんにちは');
      status.textContent = `Modelo local disponible. Prueba: ${output}`;
    } finally { translator.destroy?.(); }
  } catch (error) { status.textContent = `Traducción local no disponible: ${error.message}. No se utilizará un servicio remoto.`; }
});
