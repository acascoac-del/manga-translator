importScripts('config.js');
const MAX_BYTES = 5 * 1024 * 1024;
const MENU = 'translate-with-credits';
chrome.storage.local.setAccessLevel({accessLevel:'TRUSTED_CONTEXTS'});
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({id:MENU,title:'Traducir con mis créditos',contexts:['image']});
  chrome.alarms.create('poll-jobs',{periodInMinutes:0.5});
});
chrome.runtime.onStartup.addListener(() => chrome.alarms.create('poll-jobs',{periodInMinutes:0.5}));
async function api(path, options={}) {
  const {accessToken} = await chrome.storage.local.get('accessToken');
  if (!accessToken) throw new Error('Conectá tu cuenta desde la extensión.');
  const response = await fetch(API_ORIGIN + path, {...options,headers:{'Authorization':`Bearer ${accessToken}`,
    'Content-Type':'application/json'},signal:AbortSignal.timeout(20000)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Error del servidor (${response.status})`);
  return data;
}
async function message(text) {
  await chrome.storage.local.set({lastMessage:text});
  await chrome.action.setBadgeText({text:'!'});
}
chrome.runtime.onMessage.addListener((request,sender,respond) => {
  if (sender.id !== chrome.runtime.id || sender.tab || request.action !== 'balance') return false;
  api('/api/balance/').then(respond).catch(error=>respond({error:error.message}));
  return true;
});
function asBase64(bytes) {
  let text='';
  for(let i=0;i<bytes.length;i+=8192) text+=String.fromCharCode(...bytes.subarray(i,i+8192));
  return btoa(text);
}
async function readImage(url) {
  const response=await fetch(url,{credentials:'omit',signal:AbortSignal.timeout(20000)});
  if(!response.ok) throw new Error('No se pudo descargar la imagen. Algunos sitios protegen sus imágenes.');
  const reader=response.body.getReader(); let size=0; const chunks=[];
  while(true) {
    const {done,value}=await reader.read(); if(done) break;
    size+=value.length;
    if(size>MAX_BYTES){await reader.cancel();throw new Error('Máximo 5 MB por imagen.');}
    chunks.push(value);
  }
  const bytes=new Uint8Array(size);let offset=0;
  for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
  return asBase64(bytes);
}
chrome.contextMenus.onClicked.addListener(async (info,tab) => {
  if(info.menuItemId!==MENU || !tab?.id) return;
  try {
    const url=new URL(info.srcUrl);
    if(!['http:','https:'].includes(url.protocol)) throw new Error('Seleccioná una imagen HTTP o HTTPS.');
    const granted=await chrome.permissions.request({origins:[url.origin+'/*']});
    if(!granted) throw new Error('Se necesita permiso para descargar esa imagen.');
    const {source='auto'}=await chrome.storage.local.get('source');
    // Persist a stable request id before sending; a timeout/retry must never
    // create a second charge for the same uncertain request.
    const key='pending:'+tab.id+':'+info.srcUrl+':'+source;
    const old=(await chrome.storage.local.get(key))[key];
    const pending=old || {request_key:crypto.randomUUID(),tabId:tab.id,pageUrl:tab.url,imageUrl:info.srcUrl,source,created:Date.now()};
    if(pending.jobId){await message('Esa imagen ya está en cola.');return;}
    await chrome.storage.local.set({[key]:pending});
    const image=await readImage(info.srcUrl);
    const job=await api('/api/jobs/',{method:'POST',body:JSON.stringify({image,source,target:'es',request_key:pending.request_key})});
    pending.jobId=job.id;
    await chrome.storage.local.set({[key]:pending});
    await message('Imagen en cola. Podés seguir leyendo; el saldo está reservado.');
    await poll();
  } catch(error){await message(error.message);}
});
let polling=false;
async function poll() {
  if(polling) return;
  polling=true;
  try {
    const values=await chrome.storage.local.get(null);
    for(const [key,pending] of Object.entries(values)) {
      if(!key.startsWith('pending:') || !pending.jobId) continue;
      try {
        const result=await api(`/api/jobs/${pending.jobId}/`);
        if(result.state==='completed') {
          if(result.translated_image) {
            const tab=await chrome.tabs.get(pending.tabId).catch(()=>null);
            if(tab?.url===pending.pageUrl) await chrome.scripting.executeScript({target:{tabId:pending.tabId},
              func:(url,b64)=>{for(const img of document.images){if(img.src===url||img.currentSrc===url){img.dataset.mangaOriginal=img.src;img.removeAttribute('srcset');img.src='data:image/png;base64,'+b64;}}},
              args:[pending.imageUrl,result.translated_image]}).catch(()=>{});
          }
          await chrome.storage.local.remove(key);
          await message(result.error || 'Traducción lista. También podés descargarla desde tu cuenta.');
        } else if(result.state==='failed') {
          await chrome.storage.local.remove(key); await message(result.error || 'Falló la traducción; crédito devuelto.');
        } else if(result.state==='dispatch_unknown') {
          await message('Trabajo en revisión; no lo reenvíes. El crédito sigue reservado.');
        }
      }catch(error){await message(error.message);}
    }
  }finally{polling=false;}
}
chrome.alarms.onAlarm.addListener(alarm=>{if(alarm.name==='poll-jobs') poll();});
