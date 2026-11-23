// Experimental 0.4 pipeline. Confidence is adapter evidence, never inferred from text alone.
export const PLAN = Object.freeze({ free: true, premiumUSDMonthly: 2.99, billingActive: false });
const finite = n => typeof n === 'number' && Number.isFinite(n);
export function validateBubble(bubble, page) {
  const box = bubble?.bbox;
  if (!Array.isArray(box) || box.length !== 4 || !box.every(finite) ||
      !finite(page?.width) || !finite(page?.height) || page.width <= 0 || page.height <= 0) return 'invalid_geometry';
  const [x, y, w, h] = box;
  if (x < 0 || y < 0 || w < 16 || h < 16 || x + w > page.width || y + h > page.height) return 'outside_page';
  if (w * h / (page.width * page.height) > 0.15) return 'region_too_large';
  if (bubble.kind !== 'speech_bubble' || !finite(bubble.confidence) || bubble.confidence < 0.9 || bubble.confidence > 1) return 'unconfirmed_bubble';
  return null;
}
export function validateJapanese(ocr) {
  if (!finite(ocr?.confidence) || ocr.confidence < 0.9 || ocr.confidence > 1) return 'low_ocr_confidence';
  if (typeof ocr.text !== 'string') return 'invalid_ocr';
  const text = ocr.text.replace(/\s/gu, '');
  if (text.length < 2 || text.length > 500) return 'invalid_text_length';
  const japanese = (text.match(/[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}ー々〆]/gu) || []).length;
  if (japanese / text.length < 0.7 || /[\p{Script=Latin}\p{Script=Cyrillic}]/u.test(text)) return 'mixed_or_non_japanese';
  return null;
}
// Use only within an already segmented bubble; columns read right to left, then top to bottom.
export function orderVerticalColumns(columns) {
  return [...columns].sort((a, b) => b.bbox[0] - a.bbox[0] || a.bbox[1] - b.bbox[1]);
}
export async function browserTranslator(api = globalThis.Translator, timeoutMs = 8000) {
  const options = { sourceLanguage: 'ja', targetLanguage: 'es' };
  if (!api || typeof api.availability !== 'function' || typeof api.create !== 'function') throw new Error('local_translation_unavailable');
  let timer;
  let status;
  try {
    status = await Promise.race([
      api.availability(options),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('local_translation_timeout')), timeoutMs); }),
    ]);
  } finally { clearTimeout(timer); }
  // Never trigger a model download without a separate user action.
  if (status !== 'available') throw new Error(`local_translation_${status}`);
  return api.create(options);
}
export async function reviewPage({ page, detector, ocr, translator, now = () => performance.now() }) {
  if (!detector?.detect || !ocr?.recognize || !translator?.translate) throw new Error('local_adapter_unavailable');
  const bubbles = await detector.detect(page);
  const records = [];
  for (const bubble of bubbles) {
    const start = now();
    const record = { bbox: bubble.bbox, detectionConfidence: bubble.confidence, sourceText: '', translation: '', ocrConfidence: null, elapsedMs: 0, reason: null, status: 'rejected' };
    try {
      record.reason = validateBubble(bubble, page);
      if (!record.reason) {
        const result = await ocr.recognize(page, bubble, { language: 'jpn_vert', scale: 3, psm: [5, 6], columnOrder: 'right-to-left' });
        record.sourceText = result.text;
        record.ocrConfidence = result.confidence;
        record.reason = validateJapanese(result);
        if (!record.reason) {
          const translated = await translator.translate(result.text);
          if (typeof translated !== 'string' || !translated.trim() || translated.trim() === result.text.trim() || /[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]/u.test(translated)) record.reason = 'invalid_translation';
          else { record.translation = translated.trim(); record.status = 'pending_review'; }
        }
      }
    } catch (error) { record.reason = `adapter_error:${error.message}`; }
    record.elapsedMs = now() - start;
    records.push(record);
  }
  // No accepted/translated count until the user reviews each result.
  return records;
}
