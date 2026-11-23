import { detectBubbles } from './detector.mjs';
import { recognizeCrop } from './ocr.mjs';
import { validateJapanese } from './pipeline.mjs';
import { addCaption } from './renderer.mjs';
import { localTranslator, reviewableSpanish } from './translator.mjs';

const $ = id => document.getElementById(id);
const canvas = $('manga');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const guides = $('guides');
const guideCtx = guides.getContext('2d');
let imageReady = false;
let selection, start, worker, workerLanguage;
let busy = false;
let proposals = [];
let unverifiedText;
const captions = [];
const status = message => { $('status').textContent = message; };
const prepareTranslator = localTranslator(status);

function clearFields() {
  unverifiedText = undefined;
  $('original').value = '';
  $('translated').value = '';
}

function redraw() {
  guideCtx.clearRect(0, 0, guides.width, guides.height);
  guideCtx.lineWidth = Math.max(2, guides.width / 850);
  guideCtx.setLineDash([12, 9]);
  guideCtx.strokeStyle = '#2fc5a6';
  for (const { box } of proposals) guideCtx.strokeRect(box.x, box.y, box.width, box.height);
  if (selection) {
    guideCtx.strokeStyle = '#7654e8';
    guideCtx.strokeRect(selection.x, selection.y, selection.width, selection.height);
  }
}

function xy(event) {
  const box=canvas.getBoundingClientRect();
  return {x: Math.max(0,Math.min(canvas.width,Math.round((event.clientX-box.left)*canvas.width/box.width))),
    y:Math.max(0,Math.min(canvas.height,Math.round((event.clientY-box.top)*canvas.height/box.height)))};
}
function boxBetween(a,b){return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),width:Math.abs(a.x-b.x),height:Math.abs(a.y-b.y)};}
async function run(button, task) {
  if(busy)return;busy=true;button.disabled=true;
  try{await task();}catch(e){status('Error: '+(e?.message||e));}
  finally{busy=false;button.disabled=false;}
}

canvas.addEventListener('pointerdown', event => {
  if (!imageReady || busy) return;
  start = xy(event);
  selection = null;
  clearFields();
  canvas.setPointerCapture(event.pointerId);
});
function updateSelection(event) {
  if (!start) return;
  selection = boxBetween(start, xy(event));
  $('coordinates').textContent = selection.width + ' × ' + selection.height + ' px';
  redraw();
}
canvas.addEventListener('pointermove', updateSelection);
canvas.addEventListener('pointerup', event => {
  updateSelection(event);
  start = null;
});
canvas.addEventListener('pointercancel', () => {
  start = selection = null;
  redraw();
});

async function loadImage(url) {
  const image = new Image();
  image.src = url;
  await image.decode();
  if (image.naturalWidth * image.naturalHeight > 26000000) throw Error('Máximo 26 millones de píxeles.');
  canvas.width = guides.width = image.naturalWidth;
  canvas.height = guides.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);
  imageReady = true;
  start = selection = null;
  proposals = [];
  captions.length = 0;
  $('captions').replaceChildren();
  $('reviewList').replaceChildren();
  $('coordinates').textContent = 'Sin selección';
  clearFields();
  redraw();
  $('pageInfo').textContent = 'Imagen ' + canvas.width + ' × ' + canvas.height;
  status('Imagen lista. Detectá globos o seleccioná uno.');
}

$('fileInput').addEventListener('change', event => {
  const file = event.target.files?.[0];
  event.target.value = '';
  if (!file) return;
  void run($('fileInput'), async () => {
    const url = URL.createObjectURL(file);
    try { await loadImage(url); }
    finally { URL.revokeObjectURL(url); }
  });
});
void run($('fileInput'), async () => {
  const { latestCapture } = await chrome.storage.local.get('latestCapture');
  if (latestCapture) await loadImage(latestCapture);
  else status('Importá una imagen o usá Buscar imágenes desde la extensión.');
});

function checkBox(box) {
  if (!imageReady || !box || box.width < 9 || box.height < 9) throw Error('Seleccioná un globo.');
  if (box.width * box.height > canvas.width * canvas.height * 0.12) throw Error('Seleccioná una región más pequeña.');
}
function crop(box) {
  checkBox(box);
  const off = document.createElement('canvas');
  off.width = box.width;
  off.height = box.height;
  off.getContext('2d').drawImage(canvas, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  return off;
}
function pageBubbles() {
  const preview = document.createElement('canvas');
  const factor = Math.min(1, 1400 / Math.max(canvas.width, canvas.height));
  preview.width = Math.max(1, Math.round(canvas.width * factor));
  preview.height = Math.max(1, Math.round(canvas.height * factor));
  const previewCtx = preview.getContext('2d');
  previewCtx.drawImage(canvas, 0, 0, preview.width, preview.height);
  const sx = canvas.width / preview.width;
  const sy = canvas.height / preview.height;
  return detectBubbles(previewCtx.getImageData(0, 0, preview.width, preview.height)).map(([x, y, w, h]) => {
    const left = Math.round(x * sx);
    const top = Math.round(y * sy);
    return { x: left, y: top, width: Math.round((x + w) * sx) - left, height: Math.round((y + h) * sy) - top };
  });
}
async function getWorker(language) {
  if (worker && workerLanguage === language) return worker;
  if (worker) await worker.terminate();
  worker = null;
  if (!globalThis.Tesseract?.createWorker) throw Error('Prepará la extensión con npm run build.');
  status('Inicializando Tesseract local…');
  worker = await Tesseract.createWorker(language, 1, {
    workerPath: chrome.runtime.getURL('vendor/worker.min.js'),
    corePath: chrome.runtime.getURL('vendor/core/'),
    langPath: chrome.runtime.getURL('lang/'),
    workerBlobURL: false,
    cacheMethod: 'none',
    gzip: true,
  });
  workerLanguage = language;
  return worker;
}
async function recognize(box, language, maskOutline = false) {
  const image = crop(box);
  const engine = await getWorker(language);
  return recognizeCrop(engine, image, language === 'jpn_vert', validateJapanese, maskOutline);
}

function showRecords(records) {
  const list = $('reviewList');
  list.replaceChildren();
  for (const [index, record] of records.entries()) {
    const details = document.createElement('details');
    details.className = 'proposal';
    const summary = document.createElement('summary');
    summary.textContent = 'Región ' + (index + 1) + (record.reason ? ' · ' + record.reason : ' · revisar');
    const log = document.createElement('pre');
    log.textContent = JSON.stringify(record, null, 2);
    details.append(summary);
    if (!validateJapanese(record)) {
      const button = document.createElement('button');
      button.textContent = 'Seleccionar para revisar';
      button.addEventListener('click', () => {
        if (busy) return;
        selection = { ...record.box };
        $('coordinates').textContent = selection.width + ' × ' + selection.height + ' px';
        unverifiedText = undefined;
        $('original').value = record.text;
        $('translated').value = record.translation ?? '';
        redraw();
        status('Revisá la transcripción y el español antes de confirmar la anotación.');
      });
      details.append(button);
    }
    details.append(log);
    list.append(details);
  }
}

function scan(button, translate) {
  return run(button, async () => {
    if (!imageReady) throw Error('Importá una imagen.');
    const language = $('ocrMode').value;
    const boxes = pageBubbles();
    proposals = [];
    $('reviewList').replaceChildren();
    redraw();
    const records = [];
    let model;
    let modelError;
    // Begin model setup while the click still grants user activation.
    if (translate && boxes.length) {
      try { model = await prepareTranslator('ja'); }
      catch (error) { modelError = error.message; }
    }
    for (const [index, box] of boxes.entries()) {
      status('Reconociendo globo ' + (index + 1) + ' de ' + boxes.length + '…');
      const started = performance.now();
      const record = { box, text: '', confidence: null, translation: '', reason: null };
      try {
        Object.assign(record, await recognize(box, language, true));
        record.reason = validateJapanese(record);
        if (!record.reason && modelError) record.reason = modelError;
        if (!record.reason && model) {
          const output = await model.translate(record.text);
          if (reviewableSpanish(record.text, output)) record.translation = output.trim();
          else record.reason = 'invalid_translation';
        }
      } catch (error) {
        record.reason = error.message;
      }
      record.elapsedMs = Math.round(performance.now() - started);
      records.push(record);
    }
    proposals = records.filter(record => !validateJapanese(record));
    showRecords(records);
    redraw();
    status(boxes.length
      ? proposals.length + ' transcripciones para revisar; ' + records.filter(r => r.translation).length + ' traducciones propuestas.'
        + (modelError ? ' ' + modelError : '')
      : 'No se detectaron globos candidatos. Probá con una selección manual.');
  });
}
$('scanOnly').addEventListener('click', () => scan($('scanOnly'), false));
$('auto').addEventListener('click', () => scan($('auto'), true));

$('ocr').addEventListener('click', () => run($('ocr'), async () => {
  $('translated').value = '';
  const result = await recognize(selection, $('ocrMode').value);
  unverifiedText = validateJapanese(result) ? result.text : undefined;
  $('original').value = result.text;
  status(validateJapanese(result)
    ? 'OCR dudoso. Corregí la transcripción antes de traducir.'
    : 'OCR listo para revisión.');
}));
$('translate').addEventListener('click', () => run($('translate'), async () => {
  const source = $('original').value.trim();
  $('translated').value = '';
  if (source === unverifiedText) throw Error('Corregí la transcripción dudosa antes de traducir.');
  if (!source) throw Error('Ingresá texto para traducir.');
  if ($('lang').value === 'ja' && !/[\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Han}]/u.test(source)) {
    throw Error('Revisá el texto japonés.');
  }
  const model = await prepareTranslator($('lang').value);
  status('Traduciendo en el dispositivo…');
  const output = await model.translate(source);
  if (!reviewableSpanish(source, output)) throw Error('La salida del modelo requiere corrección manual.');
  $('translated').value = output.trim();
  status('Revisá la traducción antes de confirmar la anotación.');
}));
$('original').addEventListener('input', () => { $('translated').value = ''; });
$('paint').addEventListener('click', () => run($('paint'), async () => {
  checkBox(selection);
  const text = $('translated').value.trim();
  if (!text) throw Error('Ingresá el texto en español.');
  if (!confirm('¿Confirmás el texto de esta anotación?')) return;
  captions.push(addCaption($('captions'), canvas, selection, text, Number($('fontSize').value)));
  selection = null;
  clearFields();
  redraw();
  status('Anotación agregada. Podés quitarla con Deshacer.');
}));
$('undo').addEventListener('click', () => {
  if (busy) return;
  const caption = captions.pop();
  caption?.remove();
  status(caption ? 'Anotación retirada.' : 'No hay anotaciones.');
});
$('save').addEventListener('click', () => {
  if (!imageReady || busy) return;
  canvas.toBlob(blob => {
    if (!blob) return status('No se pudo guardar la imagen.');
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'manga-original.png';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }, 'image/png');
});

$('ocrHealth').textContent = globalThis.Tesseract?.createWorker ? 'OCR Tesseract local preparado' : 'OCR local sin preparar';
$('translatorHealth').textContent = globalThis.Translator ? 'Chrome Translator disponible para consultar' : 'Chrome Translator ausente';
