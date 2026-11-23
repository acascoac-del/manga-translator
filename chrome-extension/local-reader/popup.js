'use strict';
const $ = id => document.getElementById(id);
const message = text => $('message').textContent = text;
async function currentTab() {
 const [tab]=await chrome.tabs.query({active:true,currentWindow:true});
 if (!tab?.id||!/^https?:\/\//.test(tab.url||'')) throw Error('Abrí una página http/https con el manga (no chrome://).'); return tab;
}
async function openReader(){await chrome.tabs.create({url:chrome.runtime.getURL('reader.html')});window.close();}
async function withBusy(btn,task){btn.disabled=true;try{await task();}catch(e){message('Error: '+(e?.message||e));}finally{btn.disabled=false;}}
function collectPageImages(){
 const set=new Map();const interesting=/\.(avif|jpe?g|png|webp|gif)(\?|$)/i;
 function add(url,w=0,h=0,source='img'){
  if(!url||url.startsWith('data:')||url.startsWith('blob:'))return;
  try {const href=new URL(url,document.baseURI).href;if(!/^https?:/.test(href))return;
   if(!set.has(href))set.set(href,{src:href,w,h,source});}catch{}
 }
 for(const im of document.querySelectorAll('img')){
  const w=im.naturalWidth||im.width||Number(im.getAttribute('width'))||0;
  const h=im.naturalHeight||im.height||Number(im.getAttribute('height'))||0;
  // No hard dimension minimum: lazy-loaded images often report 0x0.
  for(const attr of ['currentSrc','src','data-src','data-original','data-lazy-src','data-url','data-image','data-echo']){
   const value=attr==='currentSrc'?im.currentSrc:im.getAttribute(attr);
   if(value)add(value,w,h,attr);
  }
  const srcset=im.getAttribute('data-srcset')||im.getAttribute('srcset');
  if(srcset){for(const candidate of srcset.split(',')){const u=candidate.trim().split(/\s+/)[0];if(u)add(u,w,h,'srcset');}}
 }
 for(const el of document.querySelectorAll('source[srcset],source[data-srcset]')){
  const srcset=el.getAttribute('srcset')||el.getAttribute('data-srcset');
  for(const candidate of srcset.split(',')){const u=candidate.trim().split(/\s+/)[0];if(u)add(u,0,0,'picture');}
 }
 for(const el of document.querySelectorAll('[style*="background"]')){
  const bg=getComputedStyle(el).backgroundImage;for(const match of bg.matchAll(/url\(["']?([^"')]+)["']?\)/g))add(match[1],el.clientWidth,el.clientHeight,'background');
 }
 const items=[...set.values()].filter(x=>x.w>=180||x.h>=250||interesting.test(x.src));
 return {items:items.slice(0,200),total:items.length,frame:location.href,imagesSeen:document.images.length};
}
$('scan').addEventListener('click',()=>withBusy($('scan'),async()=>{
 message('Buscando imágenes y contenido lazy-load…');const tab=await currentTab();
 const results=await chrome.scripting.executeScript({target:{tabId:tab.id,allFrames:true},func:collectPageImages});
 const images=new Map();let totalDom=0;
 for(const result of results){const data=result.result;if(!data)continue;totalDom+=data.imagesSeen||0;for(const im of data.items||[])if(!images.has(im.src))images.set(im.src,im);}
 const target=$('images');target.replaceChildren();const imgs=[...images.values()];
 if(!imgs.length){message(`No encontré imágenes candidatas (${totalDom} etiquetas IMG). Quizás se usan canvas, blob, o aún no cargaron: desplazate por el capítulo y repetí, o usá Capturar.`);return;}
 message(`${imgs.length} imágenes candidatas (${totalDom} IMG). Elegí una; algunas son portadas o anuncios.`);
 for(const [index,im] of imgs.entries()){
  const button=document.createElement('button');button.className='image-option';
  button.textContent=`Imagen ${index+1} · ${im.w||'?'}×${im.h||'?'} · ${im.source}`;
  button.title=im.src;button.addEventListener('click',()=>withBusy(button,async()=>{
   message('Importando…');const origin=new URL(im.src).origin+'/*';
   if(!await chrome.permissions.contains({origins:[origin]})){
    const granted=await chrome.permissions.request({origins:[origin]});
    if(!granted)throw Error('Permiso denegado para el dominio de la imagen. Usá Capturar o Importar archivo.');
   }
   const result=await chrome.runtime.sendMessage({type:'IMPORT_IMAGE',url:im.src});
   if(!result?.ok)throw Error((result?.error||'No se importó')+'. Probá captura o archivo local.');
   await openReader();
  }));target.append(button);
 }
}));
$('capture').addEventListener('click',()=>withBusy($('capture'),async()=>{
 const tab=await currentTab();message('Capturando pantalla visible…');
 const image=await chrome.tabs.captureVisibleTab(tab.windowId,{format:'jpeg',quality:90});
 await chrome.storage.local.set({latestCapture:image,capturedUrl:tab.url,capturedAt:Date.now()});await openReader();
}));
$('reader').addEventListener('click',openReader);
