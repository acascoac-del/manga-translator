import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless=new', '--remote-debugging-pipe', '--enable-unsafe-extension-debugging',
  '--user-data-dir=' + mkdtempSync(resolve(tmpdir(), 'mt-chrome04-')), 'about:blank',
], { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
let id = 0, buffer = '';
const pending = new Map();
chrome.stdio[4].setEncoding('utf8');
chrome.stdio[4].on('data', chunk => {
  buffer += chunk;
  let index;
  while ((index = buffer.indexOf('\0')) >= 0) {
    const message = JSON.parse(buffer.slice(0, index));
    buffer = buffer.slice(index + 1);
    const waiter = pending.get(message.id);
    if (waiter) {
      pending.delete(message.id);
      if (message.error) waiter.reject(Error(JSON.stringify(message.error)));
      else waiter.resolve(message.result);
    }
    if (message.method === 'Page.javascriptDialogOpening') {
      void send('Page.handleJavaScriptDialog', { accept: true }, message.sessionId);
    }
  }
});
function send(method, params = {}, sessionId) {
  const key = ++id;
  return new Promise((resolve, reject) => {
    pending.set(key, { resolve, reject });
    chrome.stdio[3].write(JSON.stringify({ id: key, method, params, sessionId }) + '\0');
  });
}
const limit = setTimeout(() => { chrome.kill(); process.exit(1); }, 130000);
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
try {
  console.log(await send('Browser.getVersion'));
  const loaded = await send('Extensions.loadUnpacked', { path: resolve('dist') });
  console.log(loaded);
  const {targetId} = await send('Target.createTarget', {url: 'chrome-extension://' + loaded.id + '/reader.html'});
  const {sessionId} = await send('Target.attachToTarget', {targetId, flatten: true});
  await send('Page.enable', {}, sessionId);
  const evaluate = async (expression, userGesture = false) => {
    const r = await send('Runtime.evaluate', {expression, awaitPromise: true, returnByValue: true, userGesture}, sessionId);
    if (r.exceptionDetails) throw Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  await pause(650);
  console.log(await evaluate("document.getElementById('ocrHealth').textContent"));
  for (const [index, columns] of [['今日は','良い天気'],['明日は','学校です'],['電車が','来ました']].entries()) {
    await evaluate(`(async () => {
      const sample = document.createElement('canvas');
      sample.width = 740; sample.height = 1000;
      const g = sample.getContext('2d');
      g.fillStyle = 'white'; g.fillRect(0,0,740,1000);
      g.fillStyle = '#333'; g.fillRect(40,40,660,920);
      g.fillStyle = 'white'; g.beginPath(); g.ellipse(525,220,68,140,0,0,Math.PI*2); g.fill();
      g.fillStyle = '#111'; g.font = '30px "Yu Gothic", "Meiryo", sans-serif'; g.textAlign = 'center';
      ${JSON.stringify(columns)}.forEach((col, x) => [...col].forEach((glyph, y) => g.fillText(glyph, 547-x*43, 159+y*34)));
      const file = new File([await new Promise(r=>sample.toBlob(r))],'synthetic.png',{type:'image/png'});
      const dt = new DataTransfer(); dt.items.add(file);
      const input = document.querySelector('#fileInput'); input.files=dt.files; input.dispatchEvent(new Event('change'));
    })()`);
    await pause(350);
    const before = await evaluate("document.querySelector('#manga').toDataURL()");
    await evaluate("document.querySelector('#scanOnly').click()", true);
    for (let attempt=0; attempt<160; attempt++) {
      if (!await evaluate("document.querySelector('#scanOnly').disabled")) break;
      await pause(300);
    }
    const records = await evaluate("[...document.querySelectorAll('#reviewList pre')].map(el=>JSON.parse(el.textContent))");
    console.log(index + 1, columns.join(''), JSON.stringify(records), await evaluate("document.querySelector('#status').textContent"));
    if (records.length !== 1 || records[0].text !== columns.join('')) {
      throw Error('expected one candidate with the correct right-to-left transcription');
    }
    if (index === 0 && process.env.TEST_LOCAL_TRANSLATION === '1') {
      const button = await evaluate(`(() => {
        const el = document.querySelector('#auto');
        el.scrollIntoView({ block: 'center' });
        const r = el.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
      })()`);
      await send('Input.dispatchMouseEvent', { type: 'mousePressed', button: 'left', clickCount: 1, ...button }, sessionId);
      await send('Input.dispatchMouseEvent', { type: 'mouseReleased', button: 'left', clickCount: 1, ...button }, sessionId);
      for (let attempt = 0; attempt < 225; attempt++) {
        if (!await evaluate("document.querySelector('#auto').disabled")) break;
        await pause(300);
      }
      const result = await evaluate("JSON.parse(document.querySelector('#reviewList pre').textContent)");
      console.log('Automatic translation proposal:', JSON.stringify(result));
      if (!result.translation || result.reason) throw Error('expected a local translation proposal');
      if (await evaluate("document.querySelectorAll('.caption').length")) throw Error('automatic caption');
    }
    await evaluate(`(() => {
      const c = document.querySelector('#manga');
      c.setPointerCapture = () => {};
      const r = c.getBoundingClientRect();
      const event = (type, x, y) => c.dispatchEvent(new PointerEvent(type,{clientX:r.left+r.width*x,clientY:r.top+r.height*y}));
      event('pointerdown',0.64,0.12); event('pointerup',0.79,0.32);
      document.querySelector('#translated').value='Texto revisado';
      document.querySelector('#paint').click();
    })()`, true);
    if (await evaluate("document.querySelectorAll('.caption').length") !== 1) throw Error('caption missing');
    if (before !== await evaluate("document.querySelector('#manga').toDataURL()")) throw Error('image changed');
    await evaluate("document.querySelector('#undo').click()");
    if (await evaluate("document.querySelectorAll('.caption').length") !== 0) throw Error('undo failed');
  }
  console.log('Overlay/undo: 3/3, manga pixels identical.');
  console.log('Translator:', await evaluate(`(async()=> {
    if (!globalThis.Translator) return 'API absent';
    const options = { sourceLanguage: 'ja', targetLanguage: 'es' };
    const state = await Promise.race([Translator.availability(options),new Promise(r=>setTimeout(()=>r('timeout'),8000))]);
    if (state !== 'available') return state;
    const model = await Translator.create(options);
    try { return {state, output:await model.translate('今日は良い天気です。')}; } finally { model.destroy(); }
  })()`, true));
} finally {
  clearTimeout(limit);
  chrome.kill();
}
