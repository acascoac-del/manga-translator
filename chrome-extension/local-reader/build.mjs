import fs from 'node:fs/promises';

const root = new URL('./', import.meta.url);
const dist = new URL('dist/', root);
await fs.mkdir(new URL('vendor/core/', dist), { recursive: true });
for (const name of ['manifest.json', 'popup.html', 'popup.js', 'background.js', 'reader.html',
  'reader.js', 'style.css', 'detector.mjs', 'ocr.mjs', 'renderer.mjs', 'translator.mjs', 'icons', 'lang']) {
  await fs.cp(new URL(name, root), new URL(name, dist), { recursive: true });
}
await fs.copyFile(new URL('../local-ai/pipeline.mjs', root), new URL('pipeline.mjs', dist));
for (const name of ['worker.min.js', 'tesseract.min.js']) {
  await fs.copyFile(new URL('node_modules/tesseract.js/dist/' + name, root), new URL('vendor/' + name, dist));
}
for (const name of await fs.readdir(new URL('node_modules/tesseract.js-core/', root))) {
  if (!/^tesseract-core.*\.wasm(?:\.js)?$/.test(name)) continue;
  await fs.copyFile(new URL('node_modules/tesseract.js-core/' + name, root), new URL('vendor/core/' + name, dist));
}
console.log('Cargá chrome-extension/local-reader/dist en chrome://extensions.');
await fs.copyFile(new URL('node_modules/tesseract.js/LICENSE.md', root), new URL('vendor/LICENSE-tesseract.md', dist));
await fs.copyFile(new URL('node_modules/tesseract.js-core/LICENSE', root), new URL('vendor/LICENSE-tesseract-core', dist));
