export function localTranslator(report) {
  let model;
  let language;

  return async function prepare(sourceLanguage) {
    const api = globalThis.Translator;
    if (!api?.availability || !api?.create) throw Error('Chrome Translator no está disponible.');
    if (model && sourceLanguage === language) return model;
    const options = { sourceLanguage, targetLanguage: 'es' };
    let timer;
    let availability;
    try {
      availability = await Promise.race([
        api.availability(options),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(Error('Chrome no respondió al consultar el modelo local.')), 8000);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
    if (!['available', 'downloadable', 'downloading'].includes(availability)) {
      throw Error('Chrome no ofrece el modelo ' + sourceLanguage + ' → es.');
    }

    if (availability !== 'available') report('Preparando la descarga del modelo local…');
    const controller = new AbortController();
    const downloadTimeout = setTimeout(() => controller.abort(), 60000);
    let next;
    try {
      next = await api.create({
        ...options,
        signal: controller.signal,
        monitor(download) {
          download.addEventListener('downloadprogress', event => {
            report('Descargando modelo local: ' + Math.round(event.loaded * 100) + '%');
          });
        },
      });
    } catch (error) {
      if (controller.signal.aborted) throw Error('Chrome agotó el tiempo para preparar el modelo local.');
      throw error;
    } finally {
      clearTimeout(downloadTimeout);
    }
    model?.destroy?.();
    model = next;
    language = sourceLanguage;
    return model;
  };
}

export function reviewableSpanish(source, output) {
  const text = String(output ?? '').trim();
  return text.length <= 450 && text !== source.trim()
    && /[\p{Script=Latin}]/u.test(text)
    && !/[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]/u.test(text);
}
