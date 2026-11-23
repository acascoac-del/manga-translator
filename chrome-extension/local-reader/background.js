'use strict';
chrome.runtime.onInstalled.addListener(async () => {
  const { plan } = await chrome.storage.local.get('plan');
  if (!plan) await chrome.storage.local.set({plan: 'free'});
});
const MAX_IMAGE_BYTES = 12 * 1024 * 1024;
function base64FromBytes(bytes) {
  let result = '';
  for (let i = 0; i < bytes.length; i += 0x6000) {
    result += String.fromCharCode(...bytes.subarray(i, i + 0x6000));
  }
  return btoa(result);
}
chrome.runtime.onMessage.addListener((msg, sender, respond) => {
  if (msg?.type !== 'IMPORT_IMAGE') return false;
  (async () => {
    const url = new URL(msg.url);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Solo se admiten imágenes HTTP/HTTPS.');
    const response = await fetch(url.href, {credentials:'omit', redirect:'follow'});
    if (!response.ok) throw new Error(`No se pudo descargar la imagen (HTTP ${response.status}).`);
    const type = response.headers.get('content-type')?.split(';')[0] || '';
    if (!type.startsWith('image/')) throw new Error('La dirección no devuelve una imagen.');
    const size = Number(response.headers.get('content-length') || 0);
    if (size > MAX_IMAGE_BYTES) throw new Error('Imagen demasiado grande (máximo 12 MB).');
    const buffer = await response.arrayBuffer();
    if (buffer.byteLength > MAX_IMAGE_BYTES) throw new Error('Imagen demasiado grande (máximo 12 MB).');
    const dataUrl = `data:${type};base64,${base64FromBytes(new Uint8Array(buffer))}`;
    await chrome.storage.local.set({latestCapture: dataUrl, capturedUrl: url.href, capturedAt: Date.now()});
    respond({ok:true});
  })().catch(err => respond({ok:false,error: String(err?.message || err)}));
  return true;
});
