'use strict';
const $ = id => document.getElementById(id);
const canvas = $('manga');
const ctx = canvas.getContext('2d', {willReadFrequently:true});
let base = null, selection = null, start = null, history = [], translator = null, activeLanguage = '', busy = false, proposals = [];
const status = msg => {$('status').textContent = msg;};
function xy(event) {
  const box=canvas.getBoundingClientRect();
  return {x: Math.max(0,Math.min(canvas.width,Math.round((event.clientX-box.left)*canvas.width/box.width))),
    y:Math.max(0,Math.min(canvas.height,Math.round((event.clientY-box.top)*canvas.height/box.height)))};
}
function boxBetween(a,b){return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),width:Math.abs(a.x-b.x),height:Math.abs(a.y-b.y)};}
function redraw() {
  if (!base) return;
  ctx.putImageData(base,0,0);
  if(proposals.length){ctx.save();ctx.strokeStyle='#2fc5a6';ctx.lineWidth=Math.max(2,canvas.width/850);ctx.setLineDash([12,9]);for(const p of proposals)ctx.strokeRect(p.box.x,p.box.y,p.box.width,p.box.height);ctx.restore();}
  if(selection){ctx.save();ctx.strokeStyle='#7654e8';ctx.lineWidth=Math.max(2,canvas.width/600);ctx.setLineDash([10,5]);ctx.strokeRect(selection.x,selection.y,selection.width,selection.height);ctx.restore();}
}
canvas.addEventListener('pointerdown',e=>{if(!base||busy)return;start=xy(e);selection=null;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!start)return;selection=boxBetween(start,xy(e));$('coordinates').textContent=`${selection.width} × ${selection.height} px`;redraw();});
canvas.addEventListener('pointerup',e=>{if(!start)return;selection=boxBetween(start,xy(e));start=null;redraw();status(selection.width>8&&selection.height>8?'Globo seleccionado. Usá OCR o ingresá su texto.':'Seleccioná un área más grande.');});
async function loadImageFromUrl(url){
  const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(Error('La imagen no se puede decodificar.'));image.src=url;});
  if(image.naturalWidth*image.naturalHeight>26000000)throw Error('Máximo 26 millones de píxeles.');
  canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;ctx.drawImage(image,0,0);base=ctx.getImageData(0,0,canvas.width,canvas.height);
  history=[];selection=null;proposals=[];$('reviewList').replaceChildren();$('pageInfo').textContent=`Imagen ${canvas.width} × ${canvas.height}`;status('Imagen lista. Seleccioná un globo.');
}
$('fileInput').addEventListener('change', async e=>{const file=e.target.files?.[0];if(!file)return;const url=URL.createObjectURL(file);try{await loadImageFromUrl(url);}catch(err){status('Error: '+err.message);}finally{URL.revokeObjectURL(url);}});
(async()=>{try{
  const {latestCapture}=await chrome.storage.local.get('latestCapture');
  if(!latestCapture){status('No hay imagen. Importá un archivo de tu PC o usá Buscar imágenes desde la extensión.');return;}
  await loadImageFromUrl(latestCapture);
  status('Imagen lista. Probá «Detectar y traducir toda la imagen».');
} catch(e){status('Error: '+e.message);}})();
async function run(button, task) {
  if(busy)return;busy=true;button.disabled=true;
  try{await task();}catch(e){status('Error: '+(e?.message||e));}
  finally{busy=false;button.disabled=false;}
}
function imageCrop(box) {
  if(!base)throw Error('No hay imagen cargada.');
  if(!box||box.width<9||box.height<9)throw Error('Seleccioná un globo con el cursor.');
  const off=document.createElement('canvas');off.width=Math.ceil(box.width);off.height=Math.ceil(box.height);
  off.getContext('2d').putImageData(base, -Math.round(box.x), -Math.round(box.y));return off;
}
let ocrWorker = null;
let ocrLanguage = null;
async function getOCRWorker(language) {
  if (!globalThis.Tesseract?.createWorker) {
    throw Error('Motor OCR sin preparar. Cerrá Chrome, ejecutá INSTALAR-WINDOWS.cmd y recargá la extensión.');
  }
  if (ocrWorker && ocrLanguage === language) return ocrWorker;
  if (ocrWorker) { await ocrWorker.terminate(); ocrWorker=null; }
  status(`Inicializando OCR japonés ${language === 'jpn_vert' ? 'vertical' : 'horizontal'} local…`);
  const root = chrome.runtime.getURL('');
  ocrWorker = await Tesseract.createWorker(language, 1, {
    workerPath: chrome.runtime.getURL('vendor/worker.min.js'),
    corePath: chrome.runtime.getURL('vendor/core/'),
    langPath: chrome.runtime.getURL('lang/'),
    workerBlobURL: false,
    gzip: true,
    cacheMethod: 'none',
    logger(message) {
      if(message?.status === 'recognizing text' && Number.isFinite(message.progress))
        status(`OCR local: ${Math.round(message.progress * 100)}%`);
    }
  });
  ocrLanguage = language;
  return ocrWorker;
}
async function detect(image) {
  const w = await getOCRWorker($('ocrMode').value);
  const result = await w.recognize(image);
  // Tesseract.js v5 returns lines with text, confidence and bounding boxes.
  const lines = result.data?.lines || [];
  const filtered = lines.filter(l => String(l.text||'').trim() && l.bbox);
  if(filtered.length) return filtered.map(l => ({
    text: l.text.trim(), confidence: l.confidence,
    boundingBox: {x:l.bbox.x0, y:l.bbox.y0, width:l.bbox.x1-l.bbox.x0, height:l.bbox.y1-l.bbox.y0}
  }));
  // No inventar una región de imagen completa si no existen cajas OCR confiables.
  return [];
}
function region(result, offset={x:0,y:0}) {
  const b=result.boundingBox;
  if(!b||![b.x,b.y,b.width,b.height].every(Number.isFinite)||b.width<5||b.height<5)return null;
  return {x:Math.max(0,b.x+offset.x),y:Math.max(0,b.y+offset.y),width:b.width,height:b.height};
}
async function translatorFor(language) {
  if(typeof globalThis.Translator==='undefined')throw Error('Esta versión de Chrome no ofrece Translator API. Se necesita Chrome de escritorio compatible.');
  if(translator&&activeLanguage===language)return translator;
  const availability=await Translator.availability({sourceLanguage:language,targetLanguage:'es'});
  if(availability==='unavailable')throw Error(`No está disponible el idioma ${language} → es.`);
  status(availability==='downloadable'?'Descargando el paquete de traducción local…':'Preparando traductor local…');
  translator?.destroy?.();translator=null;
  translator=await Translator.create({sourceLanguage:language,targetLanguage:'es',monitor(monitor){monitor.addEventListener('downloadprogress',e=>{
    if(typeof e.loaded==='number')status(`Descargando modelo: ${Math.round(e.loaded*100)}%`);
  });}});activeLanguage=language;return translator;
}
function linesToFit(text,width,fontsize) {
  ctx.font=`700 ${fontsize}px system-ui,sans-serif`;
  const lines=[];for(const paragraph of text.split('\n')){
    let current='';for(const word of paragraph.split(/\s+/).filter(Boolean)){
      const joined=current?`${current} ${word}`:word;
      if(ctx.measureText(joined).width>width&&current){lines.push(current);current=word;}else current=joined;
    }lines.push(current);
  }return lines;
}
function drawText(box,text,fontSize) {
  const x=Math.max(0,Math.round(box.x)),y=Math.max(0,Math.round(box.y));
  const w=Math.min(canvas.width-x,Math.round(box.width)),h=Math.min(canvas.height-y,Math.round(box.height));
  if(w<9||h<9)return;
  // An opaque white rectangle is intentional; this is NOT inpainting.
  ctx.fillStyle='#ffffff';ctx.fillRect(x,y,w,h);
  ctx.fillStyle='#11131d';ctx.textAlign='center';ctx.textBaseline='middle';
  let size=fontSize,lines=[];
  do {lines=linesToFit(text,Math.max(5,w-10),size);if(lines.length*size*1.2<=h-5)break;size--;}while(size>8);
  ctx.font=`700 ${size}px system-ui,sans-serif`;
  const lineHeight=size*1.2, top=y+(h-lines.length*lineHeight)/2+lineHeight/2;
  lines.forEach((line,index)=>ctx.fillText(line,x+w/2,top+index*lineHeight,Math.max(5,w-8)));
}
function saveState(){history.push(ctx.getImageData(0,0,canvas.width,canvas.height));if(history.length>12)history.shift();}
$('ocr').addEventListener('click',()=>run($('ocr'),async()=>{
  const off=imageCrop(selection);status('Reconociendo texto en el globo…');
  const results=await detect(off);const text=results.map(r=>r.text||'').filter(Boolean).join('\n');
  if(!text)throw Error('No se reconoció texto. Escribilo en el campo original.');
  $('original').value=text;status(`OCR completado: ${results.length} regiones. Revisá el texto.`);
}));
$('translate').addEventListener('click',()=>run($('translate'),async()=>{
  const original=$('original').value.trim();if(!original)throw Error('Ingresá texto para traducir.');
  const model=await translatorFor($('lang').value);status('Traduciendo en el dispositivo…');
  $('translated').value=await model.translate(original);status('Traducción lista para insertar.');
}));
$('paint').addEventListener('click',()=>run($('paint'),async()=>{
  if(!selection||selection.width<9||selection.height<9)throw Error('Seleccioná un globo.');
  const text=$('translated').value.trim();if(!text)throw Error('Escribí o traducí el texto en español.');
  if(selection.width*selection.height>canvas.width*canvas.height*0.12)throw Error('Región demasiado grande. Seleccioná únicamente el interior blanco del globo.');
  if(!confirm('Esta acción cubre el área seleccionada de blanco (no es inpainting). ¿Confirmás que la selección está dentro de un globo blanco?'))return;
  redraw();saveState();drawText(selection,text,Number($('fontSize').value));base=ctx.getImageData(0,0,canvas.width,canvas.height);
  selection=null;proposals=[];$('reviewList').replaceChildren();redraw();status('Texto colocado en la copia del manga.');
}));
$('scanOnly').addEventListener('click',()=>run($('scanOnly'),async()=>{
  if(!base)throw Error('No hay imagen cargada.');
  const blocks=await detect(imageCrop({x:0,y:0,width:canvas.width,height:canvas.height}));
  $('original').value=blocks.map(b=>b.text).join('\n');
  status(blocks.length ? `OCR completado: ${blocks.length} líneas. Revisá la transcripción.` : 'El OCR no encontró texto. Probá otro modo o una selección.');
}));
function looksJapanese(value){
  const text=String(value||'').replace(/\s/g,'');
  if(text.length<2)return false;
  const jp=(text.match(/[\u3040-\u30ff\u3400-\u9fff]/g)||[]).length;
  return jp>=2 && jp/Math.max(1,text.length)>=0.35;
}
function safeCandidate(item){
  const b=item.box;
  if(!b || !looksJapanese(item.text))return false;
  if(!Number.isFinite(item.confidence)||item.confidence<45)return false;
  if(b.x<0||b.y<0||b.x+b.width>canvas.width||b.y+b.height>canvas.height)return false;
  if(b.width<12||b.height<15)return false;
  // Regiones masivas suelen ser detecciones erróneas de fondo o paneles completos.
  if(b.width*b.height>canvas.width*canvas.height*0.10)return false;
  if(b.width>canvas.width*0.42||b.height>canvas.height*0.68)return false;
  return true;
}
function showProposals(){
  const list=$('reviewList');list.replaceChildren();
  for(const [i,p] of proposals.entries()){
    const article=document.createElement('div');article.className='proposal';
    const heading=document.createElement('strong');heading.textContent=`Región ${i+1} · confianza ${Math.round(p.confidence)}%`;
    const orig=document.createElement('div');orig.textContent='JP: '+p.text;
    const es=document.createElement('div');es.textContent='ES: '+p.translated;
    const btn=document.createElement('button');btn.textContent='Seleccionar y revisar';btn.className='secondary';
    btn.addEventListener('click',()=>{
      selection={...p.box};$('original').value=p.text;$('translated').value=p.translated;
      $('coordinates').textContent=`${Math.round(selection.width)} × ${Math.round(selection.height)} px`;
      redraw();status('Región seleccionada. Corregí el texto y pulsá «Insertar en la imagen» solo si está dentro de un globo blanco.');
    });
    article.append(heading,orig,es,btn);list.append(article);
  }
}
$('auto').addEventListener('click',()=>run($('auto'),async()=>{
  if(!base)throw Error('No hay imagen cargada.');
  proposals=[];$('reviewList').replaceChildren();redraw();
  status('Buscando texto en la imagen sin modificarla…');
  const results=await detect(imageCrop({x:0,y:0,width:canvas.width,height:canvas.height}));
  const all=results.map(r=>({text:(r.text||'').trim(),box:region(r),confidence:Number(r.confidence)}));
  const filtered=all.filter(safeCandidate).slice(0,35);
  if(!filtered.length){
    $('original').value=all.map(x=>x.text).filter(Boolean).join('\n');
    throw Error(`OCR obtuvo ${all.length} bloques, pero ninguno pasó el filtro de seguridad. No se modificó la imagen. Seleccioná manualmente un globo y probá su OCR.`);
  }
  const model=await translatorFor($('lang').value);
  const ready=[];
  for(const [i,item] of filtered.entries()){
    status(`Traduciendo propuesta ${i+1} de ${filtered.length}…`);
    try{const translated=String(await model.translate(item.text)).trim();
      if(translated && translated.length<=450)ready.push({...item,translated});
    }catch(e){/* descartamos la propuesta sin alterar la imagen */}
  }
  proposals=ready;showProposals();redraw();
  status(`Revisión segura: ${ready.length} regiones propuestas; ${all.length-filtered.length} descartadas por baja confianza o tamaño. Nada fue borrado. Seleccioná una región para corregir y aplicar manualmente.`);
}));
$('undo').addEventListener('click',()=>{
  if(busy)return;const previous=history.pop();if(!previous){status('No hay cambios para deshacer.');return;}
  base=previous;selection=null;redraw();status('Cambio deshecho.');
});
$('save').addEventListener('click',()=>{
  if(!base){status('No hay imagen para guardar.');return;}
  const off=document.createElement('canvas');off.width=canvas.width;off.height=canvas.height;
  off.getContext('2d').putImageData(base,0,0);
  off.toBlob(blob=>{if(!blob){status('No se pudo guardar la imagen.');return;}
    const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;
    link.download='mangatranslate-ai-0.3.png';link.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
  },'image/png');
});

(function diagnostics(){const ocr=!!globalThis.Tesseract?.createWorker;const tr=typeof globalThis.Translator!=='undefined';$('ocrHealth').textContent='OCR Tesseract local: '+(!!globalThis.Tesseract?.createWorker?'preparado':'falta ejecutar instalador');$('translatorHealth').textContent='Traductor local: '+(tr?'API disponible (verificar idiomas)':'NO disponible en este Chrome');$('ocrHealth').style.color=ocr?'#83e6aa':'#ffbe84';$('translatorHealth').style.color=tr?'#83e6aa':'#ffbe84';})();
